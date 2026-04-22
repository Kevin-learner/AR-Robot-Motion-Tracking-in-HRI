import cv2
import pyrealsense2 as rs
import numpy as np
import math

# ==========================================
# 1. 3D 可视化基础设置
# ==========================================
class AppState:
    def __init__(self):
        self.pitch, self.yaw = math.radians(-15), math.radians(0)
        self.translation = np.array([0, 0, -1.5], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
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

state = AppState()

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: state.mouse_btns[0] = True
    if event == cv2.EVENT_LBUTTONUP: state.mouse_btns[0] = False
    if event == cv2.EVENT_RBUTTONDOWN: state.mouse_btns[1] = True
    if event == cv2.EVENT_RBUTTONUP: state.mouse_btns[1] = False
    if event == cv2.EVENT_MOUSEMOVE:
        h, w = state.window_shape
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]
        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2
        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)
    state.prev_mouse = (x, y)

def project(v, out_shape):
    h, w = out_shape[:2]
    view_aspect = float(h) / w
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (w / 2.0, h / 2.0)
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj

def view(v):
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=2):
    p0 = project(pt1.reshape(-1, 3), out.shape)[0]
    p1 = project(pt2.reshape(-1, 3), out.shape)[0]
    if np.isnan(p0).any() or np.isnan(p1).any(): return
    p0, p1 = tuple(p0.astype(int)), tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside: cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

def pointcloud(out, verts, texcoords, color, verts_extra=None, color_extra=(0, 255, 255), radius_extra=10):
    v = view(verts)
    s = v[:, 2].argsort()[::-1]
    proj = project(v[s], out.shape)
    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T
    m = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    cw, ch = color.shape[:2][::-1]
    v_tc, u_tc = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    np.clip(u_tc, 0, ch - 1, out=u_tc)
    np.clip(v_tc, 0, cw - 1, out=v_tc)
    out[i[m], j[m]] = color[u_tc[m], v_tc[m]]

    # 绘制高亮点 (如交点)
    if verts_extra is not None and len(verts_extra) > 0:
        verts_extra = np.asarray(verts_extra, dtype=np.float32)
        proj_extra = project(view(verts_extra), out.shape)
        for pt in proj_extra:
            if not np.any(np.isnan(pt)):
                x, y = pt[:2].astype(int)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(out, (x, y), radius_extra, color_extra, thickness=-1)

# ==========================================
# 2. 全量点云获取 (移除 Segment & Crop)
# ==========================================
def process_pointcloud_single(pipeline, pc, color_intrinsics_out, R=None, T=None):
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    color_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()
    color_intrinsics_out.append(color_intrinsics)
    
    color_image = np.asanyarray(color_frame.get_data())

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    v, t = points.get_vertices(), points.get_texture_coordinates()
    
    # 直接获取全量点云和纹理，不需要裁剪
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
    colors = color_image.reshape(-1, 3) # 将颜色图像拉平以对应全量 verts

    # 只保留有效深度点 (Z > 0)
    valid_mask = verts[:, 2] > 0
    verts = verts[valid_mask]
    texcoords = texcoords[valid_mask]
    colors = colors[valid_mask]

    if R is not None and T is not None:
        verts = (R @ verts.T).T + T

    return verts, texcoords, colors, color_image, depth_intrinsics, color_frame, points


# ==========================================
# 3. 核心算法：射线与点云求交
# ==========================================
def get_gaze_point_cloud_intersection(ray_origin, ray_hit_pos, point_cloud, radius=0.03):
    if point_cloud is None or len(point_cloud) == 0:
        return None

    # 计算射线方向向量
    ray_vector = ray_hit_pos - ray_origin
    ray_dir = ray_vector / np.linalg.norm(ray_vector)
    
    # 计算点云到起点的向量
    vec_points = point_cloud - ray_origin
    
    # 计算在射线方向上的投影 t
    t = np.dot(vec_points, ray_dir)
    
    # 过滤掉射线背后的点
    front_mask = t > 0
    if not np.any(front_mask):
        return None
        
    vec_points_front = vec_points[front_mask]
    t_front = t[front_mask]
    points_front = point_cloud[front_mask]
    
    # 计算垂直距离 d (使用勾股定理 d^2 = R^2 - t^2)
    dist_sq = np.sum(vec_points_front**2, axis=1) - t_front**2
    dist_sq = np.maximum(dist_sq, 0)
    dist = np.sqrt(dist_sq)
    
    # 半径过滤 (寻找落在“激光管”内的点)
    cylinder_mask = dist < radius
    if not np.any(cylinder_mask):
        return None
        
    valid_points = points_front[cylinder_mask]
    valid_t = t_front[cylinder_mask]
    
    # 找最近的交点
    closest_idx = np.argmin(valid_t)
    return valid_points[closest_idx]


# ==========================================
# 4. 主循环：启动相机与 Demo
# ==========================================
if __name__ == "__main__":
    print("🚀 启动全景射线求交 Demo (全量点云)...")

    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    pc = rs.pointcloud()

    cv2.namedWindow("Gaze Intersection Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Gaze Intersection Demo", mouse_cb)

    # 模拟眼动数据
    simulated_origin = np.array([-0.2, -0.2, 0.0], dtype=np.float32)  
    simulated_hit_target = np.array([0.1, 0.1, 1.0], dtype=np.float32) 

    try:
        while True:
            color_intrinsics_list = []
            
            # 1. 获取全量点云
            verts, texcoords, colors, color_img, depth_intr, _, _ = process_pointcloud_single(
                pipeline, pc, color_intrinsics_list
            )

            # 准备画布
            out = np.zeros((480, 640, 3), dtype=np.uint8)
            state.window_shape = (480, 640)

            if verts is not None and len(verts) > 0:
                # 2. 执行求交算法 (容差半径 3cm)
                intersection_pt = get_gaze_point_cloud_intersection(
                    simulated_origin, 
                    simulated_hit_target, 
                    verts, 
                    radius=0.03
                )

# 3. 渲染点云
                if intersection_pt is not None:
                    # 画点云并在交点画一个黄色的球 (注意这里把 colors 改成了 color_img)
                    pointcloud(out, verts, texcoords, color_img, verts_extra=[intersection_pt], color_extra=(0, 255, 255))
                    # 让视线红线连接起点和交点
                    line3d(out, view(simulated_origin), view(intersection_pt), color=(0, 0, 255), thickness=3)
                    print(f"🎯 击中目标! 交点坐标: {np.round(intersection_pt, 3)}")
                else:
                    # 没击中时，画正常红线 (注意这里把 colors 改成了 color_img)
                    pointcloud(out, verts, texcoords, color_img)
                    line3d(out, view(simulated_origin), view(simulated_hit_target), color=(0, 0, 255), thickness=1)
            else:
                cv2.putText(out, "No Point Cloud Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Gaze Intersection Demo", out)
            
            # 按键交互：W/A/S/D 移动视线
            key = cv2.waitKey(1)
            if key == 27: # ESC 退出
                break
            elif key == ord('w'): simulated_hit_target[1] -= 0.05
            elif key == ord('s'): simulated_hit_target[1] += 0.05
            elif key == ord('a'): simulated_hit_target[0] -= 0.05
            elif key == ord('d'): simulated_hit_target[0] += 0.05

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("🛑 程序已安全退出")