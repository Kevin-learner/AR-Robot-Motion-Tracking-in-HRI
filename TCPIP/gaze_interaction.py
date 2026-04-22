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