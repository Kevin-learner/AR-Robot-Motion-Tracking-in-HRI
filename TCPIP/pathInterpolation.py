import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def generate_smooth_path_with_orientation(path_data, resolution=30):
    """
    支持 位置(B-Spline) + 姿态(Slerp) + 力(Linear) 的平滑插值
    """
    if len(path_data) < 2:
        return path_data

    # --- 1. 提取原始数据 ---
    raw_pos = [np.array(p['pos']) for p in path_data]
    raw_rot = [p['rot'] for p in path_data] 
    
    # 检查是否包含 force 字段 (兼容纯位置模式 'p')
    has_force = 'force' in path_data[0]
    if has_force:
        raw_force = [p['force'] for p in path_data]

    # --- 2. 构造 B-Spline 控制点 (外推法) ---
    v_start = raw_pos[0] + (raw_pos[0] - raw_pos[1])
    v_end = raw_pos[-1] + (raw_pos[-1] - raw_pos[-2])
    b_pts = [v_start] + raw_pos + [v_end]

    final_path = []

    def b_spline_compute(t, p0, p1, p2, p3):
        t2 = t * t
        t3 = t2 * t
        f1 = (-t3 + 3.0 * t2 - 3.0 * t + 1.0) / 6.0
        f2 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        f3 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        f4 = t3 / 6.0
        return p0 * f1 + p1 * f2 + p2 * f3 + p3 * f4

    num_segments = len(raw_pos) - 1

    for i in range(num_segments):
        p0, p1, p2, p3 = b_pts[i], b_pts[i+1], b_pts[i+2], b_pts[i+3]
        
        r_start = raw_rot[i]
        r_end = raw_rot[i+1]
        
        key_rots = R.from_quat([r_start, r_end])
        slerp = Slerp([0, 1], key_rots)
        
        # 提取当前段起止的力
        if has_force:
            f_start = raw_force[i]
            f_end = raw_force[i+1]

        for j in range(resolution):
            t = j / resolution
            
            curr_pos = b_spline_compute(t, p0, p1, p2, p3)
            curr_rot = slerp([t]).as_quat()[0]
            
            point_dict = {'pos': curr_pos, 'rot': curr_rot}
            
            # 对力进行简单的线性插值
            if has_force:
                curr_force = f_start + (f_end - f_start) * t
                point_dict['force'] = curr_force
                
            final_path.append(point_dict)

    # 补入最后一个点
    last_dict = {'pos': raw_pos[-1], 'rot': raw_rot[-1]}
    if has_force:
        last_dict['force'] = raw_force[-1]
    final_path.append(last_dict)
    
    return final_path