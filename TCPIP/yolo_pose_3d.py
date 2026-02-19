# yolo_pose_3d.py
import torch
import numpy as np
import threading
from ultralytics import YOLO

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



