import cv2
import pyrealsense2 as rs
import math
import time
import numpy as np
from segment_and_crop import segment_and_crop
from yolo_pose_3d import YOLOposeDetect, convert_17YOLOpose_to_3d_with_fill
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
import concurrent.futures
from datetime import datetime
from threading import current_thread
from global_config import pointcloud_display_queue
import queue
from kalman_filter import SimpleKalmanFilter
from MPhand_pose_3d import MPhandDetect2D, MPhand2D_to_3D
#from segment_and_crop_trt import segment_and_crop
import numpy as np
import yaml
# --------- Basic Configuration---------
def detect_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        serials.append(serial)
    return serials
class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        # self.decimate = 1
        self.scale = True
        self.color = True
        self.window_shape = (480, 640)
    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)
window_created = False
state = AppState()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
T_Stereo = np.loadtxt(config['alignment']['stereo_transform_path'])
#T_Stereo = np.loadtxt("G:/DualCameras/PythonProject/Matlab/T_icp_updated.txt")
R = T_Stereo[:3, :3]
T = T_Stereo[:3, 3]
print("📌 Transform matrix (Camera2 → Camera1), with translation in meters:")
print(T_Stereo)
skeleton_pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
kalman_filters = [SimpleKalmanFilter() for _ in range(59)]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('output_skeleton.mp4', fourcc, 30.0, (640, 480))
downsample_number = 800
serials = detect_realsense_devices()
print(f"🔍 Detected {len(serials)} RealSense devices: {serials}")
force_mono_mode = False
use_dual_camera = len(serials) >= 2 and not force_mono_mode

# --------- Pipeline 1 Configuration---------
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(serials[0])
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_1.start(config_1)
profile_1 = pipeline_1.get_active_profile()
depth_profile_1 = rs.video_stream_profile(profile_1.get_stream(rs.stream.depth))
depth_intrinsics_1 = depth_profile_1.get_intrinsics()
w_1, h_1 = depth_intrinsics_1.width, depth_intrinsics_1.height
pc_1 = rs.pointcloud()
colorizer_1 = rs.colorizer()

# --------- Pipeline 2 Configuration---------
if use_dual_camera:
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device(serials[1])
    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline_2.start(config_2)
    profile_2 = pipeline_2.get_active_profile()
    depth_profile_2 = rs.video_stream_profile(profile_2.get_stream(rs.stream.depth))
    depth_intrinsics_2 = depth_profile_2.get_intrinsics()
    w_2, h_2 = depth_intrinsics_2.width, depth_intrinsics_2.height
    pc_2 = rs.pointcloud()
    colorizer_2 = rs.colorizer()
else:
    pipeline_2 = None
    pc_2 = None
    print("⚠️ Only one camera detected. Switching to mono mode.")

# --------- 3D display---------
def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = state.window_shape
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)
def project(v, out_shape):
    """project 3d vector array to 2d"""
    h, w = out_shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj
def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation
def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3), out.shape)[0]
    p1 = project(pt2.reshape(-1, 3), out.shape)[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)
def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)
def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)
def pointcloud(out, verts, texcoords, color,
                   verts_extra=None, color_extra=(0, 255, 0), radius_extra=5,
                   skeleton_pairs=None, line_thickness=2, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s], out.shape)
    else:
        proj = project(view(verts), out.shape)

    # if state.scale:
    #     proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


    if verts_extra is not None and len(verts_extra) > 0:
        verts_extra = np.asarray(verts_extra, dtype=np.float32)
        v_extra = view(verts_extra)
        proj_extra = project(v_extra, out.shape)

        for pt in proj_extra:
            if np.any(np.isnan(pt)):
                continue
            x, y = pt[:2].astype(int)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (x, y), radius_extra, color_extra, thickness=-1)

        if skeleton_pairs is not None:
            for i, j in skeleton_pairs:
                if i >= 17 or j >= 17:
                    continue
                pt1, pt2 = verts_extra[i], verts_extra[j]
                if pt1[2] < 0.1 or pt2[2] < 0.1:
                    continue
                p0 = project(view(pt1.reshape(1, 3)), out.shape)[0]
                p1 = project(view(pt2.reshape(1, 3)), out.shape)[0]
                if np.isnan(p0).any() or np.isnan(p1).any():
                    continue
                x0, y0 = p0[:2].astype(int)
                x1, y1 = p1[:2].astype(int)
                if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                    cv2.line(out, (x0, y0), (x1, y1), color_extra, line_thickness, cv2.LINE_AA)



# 3DSkeleton
def transform_points(points, T):
    """
    Transform the Nx3 dot matrix using the 4x4 transformation matrix T and output Nx3.
    If a point is (0, 0, 0), it remains unchanged after the transformation.
    """
    transformed = np.zeros_like(points)

    # Find the point that is not (0,0,0)
    mask = ~np.all(points == 0, axis=1)

    # Transform only non-zero points
    if np.any(mask):
        points_nonzero = points[mask]
        ones = np.ones((points_nonzero.shape[0], 1))
        points_hom = np.hstack((points_nonzero, ones))
        transformed_nonzero = (T @ points_hom.T).T[:, :3]
        transformed[mask] = transformed_nonzero

    return transformed
def fuse_keypoints(send_list_1, send_list_2):
    fused = []

    for p1, p2 in zip(send_list_1, send_list_2):
        p1 = np.array(p1)
        p2 = np.array(p2)

        is_p1_valid = not np.allclose(p1, [0.0, 0.0, 0.0])
        is_p2_valid = not np.allclose(p2, [0.0, 0.0, 0.0])

        if is_p1_valid and is_p2_valid:
            fused_point = (p1 + p2) / 2.0
        elif is_p1_valid:
            fused_point = p1
        elif is_p2_valid:
            fused_point = p2
        else:
            fused_point = np.array([0.0, 0.0, 0.0])

        fused.append(tuple(fused_point))

    return fused
def optimize_pose3d(pose_3d, pose_2d, intrinsics):
    if np.all(pose_3d == 0):
        return pose_3d

    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    def project(p3d):
        x, y, z = p3d
        u = fx * x / z + cx
        v = fy * y / z + cy
        return np.array([u, v])

    optimized_pose = pose_3d.copy()
    lr = 0.01  # learning rate
    for _ in range(3):  # 只优化3步，快
        for i, p3d in enumerate(optimized_pose):
            if np.allclose(p3d, [0,0,0]):
                continue
            u_pred, v_pred = project(p3d)
            du = pose_2d[i][0] - u_pred
            dv = pose_2d[i][1] - v_pred

            delta = np.array([
                du * p3d[2] / fx,
                dv * p3d[2] / fy,
                -(du * p3d[0] / fx + dv * p3d[1] / fy)
            ])

            optimized_pose[i] += lr * delta

    return optimized_pose
def apply_kalman_filter_to_skeleton(send_coords_input, use_transform=True, T_M=None):

    filtered_send_coords = []

    for i, p in enumerate(send_coords_input):
        if not np.allclose(p, [0.0, 0.0, 0.0]):
            kalman_filters[i].predict()
            kalman_filters[i].update(np.array(p))
            filtered_send_coords.append(tuple(kalman_filters[i].get_state()))
        else:
            filtered_send_coords.append((0.0, 0.0, 0.0))

    if use_transform:
        if T_M is None:
            raise ValueError("Transformation matrix T_M must be provided when use_transform=True.")
        send_coords_filtered = [tuple(p) for p in transform_points(np.array(filtered_send_coords), T_M)]
    else:
        send_coords_filtered = filtered_send_coords

    return send_coords_filtered
def compute_depth_confidence(depth_image, keypoints_2d, window_size=5, alpha=0.05):

    h, w = depth_image.shape
    half = window_size // 2
    conf_list = []

    for u, v in keypoints_2d:
        u = int(u)
        v = int(v)
        if not (0 <= u < w and 0 <= v < h):
            conf_list.append(0.0)
            continue

        patch = depth_image[max(v - half, 0):min(v + half + 1, h), max(u - half, 0):min(u + half + 1, w)]

        if patch.size == 0 or np.all(patch == 0):
            conf_list.append(0.0)
            continue

        valid_patch = patch[patch > 0]
        if valid_patch.size == 0:
            conf_list.append(0.0)
            continue

        sigma = np.std(valid_patch)
        confidence = np.exp(-sigma ** 2 / alpha)
        conf_list.append(float(confidence))

    return conf_list
def fuse_keypoints_with_confidence(send_list_1, send_list_2, conf_list_1, conf_list_2):
    fused = []

    for p1, p2, c1, c2 in zip(send_list_1, send_list_2, conf_list_1, conf_list_2):
        p1 = np.array(p1)
        p2 = np.array(p2)

        is_p1_valid = not np.allclose(p1, [0.0, 0.0, 0.0])
        is_p2_valid = not np.allclose(p2, [0.0, 0.0, 0.0])

        if is_p1_valid and is_p2_valid:
            if (c1 + c2) > 1e-6:
                fused_point = (c1 * p1 + c2 * p2) / (c1 + c2)
            else:
                fused_point = (p1 + p2) / 2.0
        elif is_p1_valid:
            fused_point = p1
        elif is_p2_valid:
            fused_point = p2
        else:
            fused_point = np.array([0.0, 0.0, 0.0])

        fused.append(tuple(fused_point))

    return fused
def process_skeleton_single(pipeline, pc, color_intrinsics_out, R=None, T=None):

    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    color_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
    color_intrinsics_out.append(color_intrinsics)
    w, h = color_intrinsics.width, color_intrinsics.height
    color_image = np.asanyarray(color_frame.get_data())

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

    verts = verts.reshape(h * w, 3)
    texcoords = texcoords.reshape(h * w, 2)
    colors = color_image


    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
    verts = verts.reshape(h * w, 3)
    texcoords = texcoords.reshape(h * w, 2)

    start_time = time.time()  # 开始计时
    pose_2d_yolo = YOLOposeDetect(color_image)
    end_time = time.time()  # 结束计时
    capture_time = end_time - start_time
    # print(f"YOLOdelay: {capture_time:.4f} seconds")
    pose_3d_yolo = convert_17YOLOpose_to_3d_with_fill(pose_2d_yolo, verts, color_image.shape)
    pose_3d_yolo_OP = optimize_pose3d(pose_3d_yolo, pose_2d_yolo, color_intrinsics)

    pose_2d_hand = MPhandDetect2D(color_image)
    pose_3d_hand = MPhand2D_to_3D(pose_2d_hand, verts, color_image.shape)
    pose_3d_hand_OP = optimize_pose3d(pose_3d_hand, pose_2d_hand, color_intrinsics)

    pose_2d = np.concatenate([pose_2d_yolo, pose_2d_hand], axis=0)  # (59, 2)
    pose_3d = np.concatenate([pose_3d_yolo_OP, pose_3d_hand_OP], axis=0)  # (59, 3)


    process_time = time.time()
    pro_time = process_time - end_time
    #print(f"Process time AAA: {pro_time:.4f} seconds")

    if R is not None and T is not None:
        verts = (R @ verts.T).T + T
        pose_3d =(R @  pose_3d.T).T + T
    return pose_2d, pose_3d, verts, texcoords, colors, color_image, depth_intrinsics, color_frame,depth_frame, points
def Body3DSkeletonProcess_dual(T_M, use_dual_camera=False):
    color_intrinsics_list = []

    if use_dual_camera:

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(process_skeleton_single, pipeline_1, pc_1, color_intrinsics_list,None, None)
            future2 = executor.submit(process_skeleton_single, pipeline_2, pc_2, color_intrinsics_list, R, T)
            result1 = future1.result()
            result2 = future2.result()

            if result1[0] is None or result2[0] is None:
                print("🛑 [Dual] No person detected, sending zeros.")
                return [(0, 0, 0)] * 17, False


            pose_2d_1, pose_3d_1, verts_1, texcoords_1, colors_1, color_image_1, depth_intrinsics_1, mapped_frame_1, depth_frame_1, points_1 = result1
            pose_2d_2, pose_3d_2_in_1, verts_2, texcoords_2, colors_2, color_image_2, _, mapped_frame_2, depth_frame_2, points_2 = result2

            pose_3d_raw_1 = pose_3d_1.copy()
            pose_3d_raw_2 = pose_3d_2_in_1.copy()

            pose_3d_1[:, 2] *= -1
            pose_3d_2_in_1[:, 2] *= -1
            #send_coords_raw = fuse_keypoints(pose_3d_1, pose_3d_2_in_1)
            conf_list_1 = compute_depth_confidence(depth_frame_1, pose_2d_1)
            conf_list_2 = compute_depth_confidence(depth_frame_2, pose_2d_2)

            send_coords_raw = fuse_keypoints_with_confidence(pose_3d_1, pose_3d_2_in_1, conf_list_1, conf_list_2)
            #send_coords = [tuple(p) for p in transform_points(np.array(send_coords_raw), T_M)]
            send_coords = apply_kalman_filter_to_skeleton(send_coords_raw, use_transform=True, T_M=T_M)
            if not pointcloud_display_queue.full():
                pointcloud_display_queue.put_nowait((
                    verts_1, texcoords_1, colors_1,
                    verts_2, texcoords_2, colors_2,
                    depth_intrinsics_1,
                    pose_3d_raw_1, pose_3d_raw_2,
                    color_image_1, color_image_2,
                    pose_2d_1, pose_2d_2
                ))

    else:

        result = process_skeleton_single(pipeline_1, pc_1, color_intrinsics_list)
        if result[0] is None:
            print("🛑 [Mono] No person detected, sending zeros.")


            if not pointcloud_display_queue.full():
                pointcloud_display_queue.put_nowait((
                    np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 3)),
                    np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 3)),
                    rs.intrinsics(),
                    None, None
                ))

            return [(0, 0, 0)] * 17, False

       # pose_2d_1, pose_3d_1, verts_1, texcoords_1, colors_1, color_image_1, depth_intrinsics_1, mapped_frame_1, points_1 = result
        pose_2d_1, pose_3d_1, verts_1, texcoords_1, colors_1, color_image_1, depth_intrinsics_1, mapped_frame_1, depth_frame_1, points_1 = result

        pose_3d_raw_1 = pose_3d_1.copy()
        pose_3d_1[:, 2] *= -1
        pose_3d_transformed_1 = transform_points(pose_3d_1, T_M)
        send_coords = [tuple(p) for p in pose_3d_transformed_1]
        send_coords = apply_kalman_filter_to_skeleton(send_coords, use_transform=False)
        #verts_all = verts_1
        if not pointcloud_display_queue.full():
            pointcloud_display_queue.put_nowait((
                verts_1, texcoords_1, colors_1,
                np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 3)),
                depth_intrinsics_1,
                pose_3d_raw_1,
                None,
                color_image_1,
                np.zeros_like(color_image_1),  # dummy for camera 2
                pose_2d_1,
                [(0, 0)] * 17  # dummy 2d for camera 2
            ))



    return send_coords, False
def draw_pose_2d(image, keypoints, skeleton_pairs, color=(0, 255, 0), radius=4, thickness=2):

    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)

    for i, j in skeleton_pairs:
        if i >= len(keypoints) or j >= len(keypoints):
            continue

        if i >= 17 or j >= 17:
            continue
        pt1, pt2 = keypoints[i], keypoints[j]
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)
def main_3DSkeleton_visualizer_loop():
    from global_config import pointcloud_display_queue
    import cv2
    import numpy as np
    global state
    print("🖼️ Main-thread visualizer loop started.")
    cv2.namedWindow("Main Skeleton", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Main Skeleton", mouse_cb)


    while True:
        try:
            data = pointcloud_display_queue.get(timeout=1.0)
            if data is None:
                print("🛑 Exiting main visualizer loop.")
                break


            (verts_1, texcoords_1, colors_1,
             verts_2, texcoords_2, colors_2,
             intrinsics,
             pose_3d_1, pose_3d_2,
             color_image_1, color_image_2,
             pose_2d_1, pose_2d_2) = data

            if intrinsics.width == 0 or intrinsics.height == 0:
                print("❌ Invalid intrinsics, skipping frame.")
                continue

            frame_h, frame_w = intrinsics.height, intrinsics.width
            out = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            state.window_shape = (frame_h, frame_w)

            frustum(out, intrinsics)
            axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=2)


            pointcloud(out, verts_1, texcoords_1, colors_1,
                       verts_extra=pose_3d_1,
                       color_extra=(0, 255, 0),
                       skeleton_pairs=skeleton_pairs,
                       radius_extra=5, line_thickness=2)


            if verts_2 is not None and len(verts_2) > 0:
                # pointcloud(out, verts_2, texcoords_2, colors_2)
                pointcloud(out, verts_2, texcoords_2, colors_2,
                           verts_extra=pose_3d_2,
                           color_extra=(0,0,255),
                           skeleton_pairs=skeleton_pairs,
                           radius_extra=5, line_thickness=2)


            if pose_3d_1 is not None:
                image_2d_1 = color_image_1.copy()
                draw_pose_2d(image_2d_1, pose_2d_1, skeleton_pairs, color=(0, 255, 0))
                cv2.imshow("2D View - Camera 1", image_2d_1)

            if any(state.mouse_btns):
                axes(out, view(state.pivot), state.rotation, thickness=4)

            cv2.imshow("Main Skeleton", out)
            key = cv2.waitKey(1)

            if key == 27 or cv2.getWindowProperty("Main Skeleton", cv2.WND_PROP_AUTOSIZE) < 0:
                print("👋 Window closed by user.")
                break

            from BodyPointCloud_dual import state

            if key == ord("r"):
                print("🔄 Resetting view.")
                state.reset()

            if key == ord("p"):
                state.paused ^= True
                print(f"⏸️ Paused: {state.paused}")

            if key == ord("z"):
                state.scale ^= True
                print(f"🔍 Zoom mode toggled: {state.scale}")

            if key == ord("c"):
                state.color ^= True
                print(f"🎨 Color mode toggled: {state.color}")

            if key == ord("s"):
                cv2.imwrite('./out.png', out)
                print("💾 Screenshot saved to out.png")

            if key == ord("e"):
                try:
                    from BodyPointCloud_dual import points_1, mapped_frame_1
                    points_1.export_to_ply('./out.ply', mapped_frame_1)
                    print("💾 PointCloud exported to out.ply")
                except Exception as e:
                    print(f"❌ Failed to export PLY: {e}")
        except queue.Empty:
            continue



# PointCloud
from sklearn.decomposition import PCA
def pca_filter_ellipsoid(verts, colors, threshold=[0.5, 0.3, 0.3]):

    if verts.shape[0] < 10:
        return verts, colors

    pca = PCA(n_components=3)
    verts_pca = pca.fit_transform(verts)

    mask = (np.abs(verts_pca[:, 0]) < threshold[0]) & \
           (np.abs(verts_pca[:, 1]) < threshold[1]) & \
           (np.abs(verts_pca[:, 2]) < threshold[2])

    verts_filtered = verts[mask]
    colors_filtered = colors[mask]

    return verts_filtered, colors_filtered
def z_filter(verts, colors, z_range=(0.05, 4.5)):

    depth_mask = (verts[:, 2] > z_range[0]) & (verts[:, 2] < z_range[1])
    verts_filtered = verts[depth_mask]
    colors_filtered = colors[depth_mask]
    return verts_filtered, colors_filtered
def estimate_voxel_size(points, target_num):
    bbox = np.ptp(points, axis=0)
    volume = np.prod(bbox)
    voxel_volume = volume / target_num
    voxel_size = voxel_volume ** (1/3)
    return voxel_size
def voxel_down_sample(points, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return points[unique_indices]
def process_pointcloud_single(pipeline, pc, color_intrinsics_out, R=None, T=None,ero_para=1):
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    color_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
    color_intrinsics_out.append(color_intrinsics)
    w, h = color_intrinsics.width, color_intrinsics.height
    color_image = np.asanyarray(color_frame.get_data())

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
    indices = segment_and_crop(color_image, color_intrinsics,ero_para)

    if len(indices) == w * h:
        return None, None, None, color_image, depth_intrinsics, color_frame, points  # ⚠️ 特殊返回表示“没检测到人”

    verts = verts.reshape(h * w, 3)[indices]
    texcoords = texcoords.reshape(h * w, 2)[indices]
    colors = color_image[indices // w, indices % w]

    if R is not None and T is not None:
        verts = (R @ verts.T).T + T


    #verts, colors = z_filter(verts, colors, z_range=(0.05, 4.5))

    #verts, colors = pca_filter_ellipsoid(verts, colors, threshold=[0.5, 0.3, 0.3])

    return verts, texcoords, colors, color_image, depth_intrinsics, color_frame, points
def BodyPointCloudProcess_dual(T_M, max_points=4000, use_dual_camera=False,ero_para=1):
    color_intrinsics_list = []

    if use_dual_camera:


        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(process_pointcloud_single, pipeline_1, pc_1, color_intrinsics_list,None, None,ero_para=ero_para)
            future2 = executor.submit(process_pointcloud_single, pipeline_2, pc_2, color_intrinsics_list, R, T,ero_para=ero_para)
            result1 = future1.result()
            result2 = future2.result()

            if result1[0] is None or result2[0] is None:
                print("🛑 [Dual] No person detected, sending zeros.")
                return [(0, 0, 0)] * max_points, False

            verts_1, texcoords_1, colors_1, color_image_1, depth_intrinsics_1, mapped_frame_1, points_1 = result1
            verts_2, texcoords_2, colors_2, color_image_2, _, _, _ = result2

        verts_all = np.vstack([verts_1, verts_2])
        tex_all = np.vstack([texcoords_1, texcoords_2])
        color_all = np.vstack([colors_1, colors_2])
    else:

        result = process_pointcloud_single(pipeline_1, pc_1, color_intrinsics_list,ero_para=ero_para)
        if result[0] is None:
            print("🛑 [Mono] No person detected, sending zeros.")
            return [(0, 0, 0)] * max_points, False

        verts_1, texcoords_1, colors_1, color_image_1, depth_intrinsics_1, mapped_frame_1, points_1 = result
        verts_all = verts_1


    verts_flipped = verts_all.copy()
    verts_flipped[:, 2] *= -1
    coords = transform_points(verts_flipped, T_M)
    coords_mm = coords * 1000
    valid_mask = np.all(np.abs(coords_mm) <= 32767, axis=1)
    coords_valid = coords[valid_mask]
    print(f"Before voxel downsample: {coords_valid.shape[0]} points")
    if coords_valid.shape[0] > max_points:
        sampled = np.random.choice(len(coords_valid), max_points, replace=False)
        coords_valid = coords_valid[sampled]

        print(f"After voxel downsample: {coords_valid.shape[0]} points")

    send_coords = [tuple(p) for p in coords_valid]



    if not pointcloud_display_queue.full():
        pointcloud_display_queue.put_nowait((
            verts_1, texcoords_1, color_image_1,
            verts_2 if use_dual_camera else np.empty((0, 3)),
            texcoords_2 if use_dual_camera else np.empty((0, 2)),
            color_image_2 if use_dual_camera else np.empty((0, 3)),
            depth_intrinsics_1
        ))

    return send_coords, False
def main_pointcloud_visualizer_loop():
    from global_config import pointcloud_display_queue
    import cv2
    import numpy as np
    global state
    print("🖼️ Main-thread visualizer loop started.")
    cv2.namedWindow("Main PointCloud", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Main PointCloud", mouse_cb)


    while True:
        try:
            data = pointcloud_display_queue.get(timeout=1.0)
            if data is None:
                print("🛑 Exiting main visualizer loop.")
                break

            verts_1, texcoords_1, colors_1, verts_2, texcoords_2, colors_2, intrinsics = data

            if intrinsics.width == 0 or intrinsics.height == 0:
                print("❌ Invalid intrinsics, skipping frame.")
                continue

            frame_h, frame_w = intrinsics.height, intrinsics.width
            out = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            state.window_shape = (frame_h, frame_w)
            # TODO: 在这里画 pointcloud、frustum 等，可逐步恢复逻辑

            frustum(out, intrinsics)
            axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=2)

            if verts_1 is not None and texcoords_1 is not None and colors_1 is not None:
                pointcloud(out, verts_1, texcoords_1, colors_1)

            if verts_2 is not None and texcoords_2 is not None and colors_2 is not None and len(verts_2) > 0:
                pointcloud(out, verts_2, texcoords_2, colors_2)

            if any(state.mouse_btns):
                axes(out, view(state.pivot), state.rotation, thickness=4)

            cv2.imshow("Main PointCloud", out)
            key = cv2.waitKey(1)

            if key == 27 or cv2.getWindowProperty("Main PointCloud", cv2.WND_PROP_AUTOSIZE) < 0:
                print("👋 Window closed by user.")
                break

            from BodyPointCloud_dual import state

            if key == ord("r"):
                print("🔄 Resetting view.")
                state.reset()

            if key == ord("p"):
                state.paused ^= True
                print(f"⏸️ Paused: {state.paused}")

            if key == ord("z"):
                state.scale ^= True
                print(f"🔍 Zoom mode toggled: {state.scale}")

            if key == ord("c"):
                state.color ^= True
                print(f"🎨 Color mode toggled: {state.color}")

            if key == ord("s"):
                cv2.imwrite('./out.png', out)
                print("💾 Screenshot saved to out.png")

            if key == ord("e"):
                try:
                    from BodyPointCloud_dual import points_1, mapped_frame_1
                    points_1.export_to_ply('./out.ply', mapped_frame_1)
                    print("💾 PointCloud exported to out.ply")
                except Exception as e:
                    print(f"❌ Failed to export PLY: {e}")
        except queue.Empty:
            continue


if __name__ == "__main__":
    while True:
        T_M = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

        send_coords, should_quit = BodyPointCloudProcess_dual(T_M, max_points=4000, use_dual_camera=False, ero_para=1)

        if should_quit:
            print("⚠️ No valid person detected in this frame, retrying...")
            continue

        print(f"✅ Frame OK, {len(send_coords)} points captured.")



