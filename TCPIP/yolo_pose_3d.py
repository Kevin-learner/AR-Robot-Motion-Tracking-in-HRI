# yolo_pose_3d.py
import torch
import numpy as np
import threading
from ultralytics import YOLO
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"YOLO pose on: {device}")


_model_pose = None
_model_lock = threading.Lock()

def get_model_pose():
    global _model_pose
    with _model_lock:
        if _model_pose is None:
            print("Loading YOLOv11 model...")
            _model_pose = YOLO("yolo11n-pose.pt").to(device)
            #_model_pose = YOLO("yolo11n-pose.engine")

            _ = _model_pose.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        return _model_pose



def YOLOposeDetect(color_image):
    model_pose = get_model_pose()
    results = model_pose(color_image, verbose=False)

    pose_2d = np.zeros((17, 2), dtype=np.float32)
    if len(results[0].boxes) == 0 or results[0].keypoints is None:
        return pose_2d

    confidences = results[0].boxes.conf.data
    max_conf_index = torch.argmax(confidences)
    max_conf_value = confidences[max_conf_index].item()
    if max_conf_value < 0.75:
        return pose_2d

    keypoints_xy = results[0].keypoints.xy[max_conf_index].cpu().numpy()
    return keypoints_xy.astype(np.float32)

def convert_17YOLOpose_to_3d_with_fill(pose_2d, verts, image_shape, kernel_size=3, z_thresh=1.0):
    h, w = image_shape[:2]
    offset = kernel_size // 2
    pose_3d = []

    for i, (x, y) in enumerate(pose_2d):
        if x == 0 and y == 0:
            pose_3d.append([0.0, 0.0, 0.0])
            continue

        x = int(round(x))
        y = int(round(y))

        idx = y * w + x
        pt_valid = False
        pt = np.zeros(3)

        if 0 <= idx < verts.shape[0]:
            pt = verts[idx]
            if np.linalg.norm(pt) > 1e-5:
                pt_valid = True

        if not pt_valid:
            neighbor_points = []

            for dy in range(-offset, offset + 1):
                for dx in range(-offset, offset + 1):
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        n_idx = ny * w + nx
                        if 0 <= n_idx < verts.shape[0]:
                            npt = verts[n_idx]
                            if np.linalg.norm(npt) > 1e-5:
                                neighbor_points.append(npt)

            if neighbor_points:
                pt = np.median(np.array(neighbor_points), axis=0)
            else:
                pt = np.zeros(3)

        pose_3d.append(pt.tolist())

    pose_3d = np.array(pose_3d, dtype=np.float32)


    def is_valid(pt):
        return not np.allclose(pt, [0.0, 0.0, 0.0], atol=1e-6)

    def correct_joint(idx, ref_indices):
        ref_points = [pose_3d[j] for j in ref_indices if is_valid(pose_3d[j])]
        if is_valid(pose_3d[idx]) and ref_points:
            z_ref = np.mean([p[2] for p in ref_points])
            if abs(pose_3d[idx][2] - z_ref) > z_thresh:
                pose_3d[idx] = np.mean(ref_points, axis=0)


    correction_rules = {

        0: [1, 2],
        5: [3, 7],
        6: [4, 8],
        7: [5, 5],
        8: [6, 6],
        9: [5, 5],
        10: [6, 6],
        11: [5, 13],
        12: [6, 14],
        13: [11, 11],
        14: [12, 12],
        15: [11, 11],
        16: [12, 12],

    }

    for joint_idx, refs in correction_rules.items():
        correct_joint(joint_idx, refs)

    return pose_3d

def rotate_image_and_get_matrix(image, angle_deg):
    """
    将图像绕中心点旋转，并返回旋转后的图像和变换矩阵
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # 获取 2D 旋转矩阵 (尺度为 1.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # 执行旋转 (保持原图尺寸)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    
    return rotated_img, M

def inverse_rotate_keypoints(keypoints, M):
    """
    将 YOLO 在旋转图上找出的 2D 坐标，逆向旋转回原始相机画面
    """
    # 求旋转矩阵的逆矩阵
    M_inv = cv2.invertAffineTransform(M)
    
    restored_keypoints = []
    IMG_W = 640
    IMG_H = 480
    
    for pt in keypoints:
        # 如果是无效点 (0,0)，保持不变
        if pt[0] == 0 and pt[1] == 0:
            restored_keypoints.append([0.0, 0.0])
            continue
            
        # 组装齐次坐标 [u, v, 1]
        p_hom = np.array([pt[0], pt[1], 1.0])
        # 矩阵乘法还原坐标
        p_restored = M_inv @ p_hom

        x = np.clip(p_restored[0], 0, IMG_W - 1)
        y = np.clip(p_restored[1], 0, IMG_H - 1)

        restored_keypoints.append([x, y])        
    return np.array(restored_keypoints, dtype=np.float32)

def YOLOposeDetect_with_rotation(color_image, roll_angle=0.0):
    """
    带数字云台抗旋转的 YOLO 推理
    :param roll_angle: 相机绕光轴的旋转角度 (度数)。如果是正着，传 0。
    """
    # 1. 如果相机歪了，先把图像拧正
    if abs(roll_angle) > 5.0:  # 超过 5 度才触发旋转，节省算力
        img_to_infer, M = rotate_image_and_get_matrix(color_image, roll_angle)
    else:
        img_to_infer = color_image
        M = None

    # 2. 送入 YOLO 进行推理 (YOLO 现在看到的是正立的人)
    model_pose = get_model_pose()
    results = model_pose(img_to_infer, verbose=False)

    pose_2d = np.zeros((17, 2), dtype=np.float32)
    if len(results[0].boxes) == 0 or results[0].keypoints is None:
        return pose_2d

    confidences = results[0].boxes.conf.data
    max_conf_index = torch.argmax(confidences)
    if confidences[max_conf_index].item() < 0.75:
        return pose_2d

    # 拿到基于“正立图像”的 2D 坐标
    keypoints_xy = results[0].keypoints.xy[max_conf_index].cpu().numpy().astype(np.float32)

    # 3. 🌟 关键：如果之前旋转了图像，现在必须把点转回真实的歪斜状态！
    if M is not None:
        pose_2d = inverse_rotate_keypoints(keypoints_xy, M)
    else:
        pose_2d = keypoints_xy

    return pose_2d

