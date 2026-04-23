import numpy as np

def get_gaze_point_cloud_intersection(ray_origin, ray_hit_pos, point_cloud, radius=0.03):
    """
    计算射线与点云的交点
    输入:
        ray_origin (np.array): 射线起点 (统一坐标系)
        ray_hit_pos (np.array): 射线终点/方向点 (统一坐标系)
        point_cloud (np.array): 点云数据 (N, 3) (统一坐标系)
        radius (float): 射线的容差半径 (米)，默认 3cm
    输出:
        np.array: 最近的交点坐标 [x, y, z]，若无交点返回 None
    """
    if point_cloud is None or len(point_cloud) == 0:
        return None
        
    if ray_origin is None or ray_hit_pos is None:
        return None

    # 1. 计算射线方向向量
    ray_vector = np.array(ray_hit_pos) - np.array(ray_origin)
    ray_dir = ray_vector / np.linalg.norm(ray_vector)
    
    # 2. 计算点云到起点的向量
    vec_points = point_cloud - np.array(ray_origin)
    
    # 3. 投影距离 t
    t = np.dot(vec_points, ray_dir)
    
    # 4. 过滤背后点
    front_mask = t > 0
    if not np.any(front_mask):
        return None
        
    vec_points_front = vec_points[front_mask]
    t_front = t[front_mask]
    points_front = point_cloud[front_mask]
    
    # 5. 垂直距离 d
    dist_sq = np.sum(vec_points_front**2, axis=1) - t_front**2
    dist_sq = np.maximum(dist_sq, 0)
    dist = np.sqrt(dist_sq)
    
    # 6. 半径过滤
    cylinder_mask = dist < radius
    if not np.any(cylinder_mask):
        return None
        
    valid_points = points_front[cylinder_mask]
    valid_t = t_front[cylinder_mask]
    
    # 7. 找最近点
    closest_idx = np.argmin(valid_t)
    return valid_points[closest_idx]

import open3d as o3d

class PointCloudAccumulator:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size
        self.global_pcd = o3d.geometry.PointCloud()
        self.is_empty = True
        self.frame_count = 0

        # === 🌟 初始化 Open3D 实时可视化窗口 ===
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Real-time 3D Map (Press 'C' to Capture)", width=800, height=600)
        
        # 创建一个专用于显示的代理点云
        self.display_pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.display_pcd)
        self.first_render = True

    # 注意这里多加了一个 point_colors 参数
    def add_point_cloud(self, points_robot_base, point_colors):
        """将新视角的点云融入全局地图并刷新画面"""
        if points_robot_base is None or len(points_robot_base) == 0: return
        
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points_robot_base)
        
        # ==========================================
        # 🌟 直接赋予真实世界的 RGB 颜色！
        # ==========================================
        if point_colors is not None:
            new_pcd.colors = o3d.utility.Vector3dVector(point_colors)

        if self.is_empty:
            self.global_pcd = new_pcd
            self.is_empty = False
        else:
            self.global_pcd += new_pcd

        # 体素降采样：Open3D 会自动把重叠点的颜色“平滑混合”，拼缝会非常自然！
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 刷新渲染窗口
        self.display_pcd.points = self.global_pcd.points
        self.display_pcd.colors = self.global_pcd.colors
        self.vis.update_geometry(self.display_pcd)
        
        if self.first_render:
            self.vis.reset_view_point(True)
            self.first_render = False
            
        self.frame_count += 1

    def update_window(self):
        """放在主循环中，维持窗口不卡死"""
        self.vis.poll_events()
        self.vis.update_renderer()

    def get_merged_points_numpy(self):
        if self.is_empty: return None
        return np.asarray(self.global_pcd.points)

    def clear(self):
        self.global_pcd = o3d.geometry.PointCloud()
        self.display_pcd.clear()
        self.vis.update_geometry(self.display_pcd)
        self.is_empty = True
        self.frame_count = 0