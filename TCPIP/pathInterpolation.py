import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def generate_smooth_path_with_orientation(path_with_orientations, resolution=30):
    """
    修正版：使用外推法 (Extrapolation) 替代重复法 (Clamping)
    解决了首尾速度为0导致的“旋转堆积”问题。
    """
    if len(path_with_orientations) < 2:
        return path_with_orientations

    # --- 1. 提取原始数据 ---
    raw_pos = [np.array(p['pos']) for p in path_with_orientations]
    raw_rot = [p['rot'] for p in path_with_orientations] # [qx, qy, qz, qw]

    # --- 2. 构造 B-Spline 控制点 (外推法) ---
    # 目的：让曲线以恒定速度穿过首尾，保证位移和旋转的步调一致
    # 虚拟起点 = P0 + (P0 - P1)
    v_start = raw_pos[0] + (raw_pos[0] - raw_pos[1])
    # 虚拟终点 = Pn + (Pn - Pn-1)
    v_end = raw_pos[-1] + (raw_pos[-1] - raw_pos[-2])

    # B-Spline 专用点列表：[VirtualStart, P0, P1, ..., Pn, VirtualEnd]
    b_pts = [v_start] + raw_pos + [v_end]

    final_path = []

    # --- 3. B-Spline 基函数 ---
    def b_spline_compute(t, p0, p1, p2, p3):
        t2 = t * t
        t3 = t2 * t
        # 标准 Uniform B-Spline 矩阵形式
        f1 = (-t3 + 3.0 * t2 - 3.0 * t + 1.0) / 6.0
        f2 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        f3 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        f4 = t3 / 6.0
        return p0 * f1 + p1 * f2 + p2 * f3 + p3 * f4

    # --- 4. 遍历每一段 (Segment) ---
    # 原始点有 N 个，我们就生成 N-1 段曲线
    # i 从 0 到 len-2
    num_segments = len(raw_pos) - 1

    for i in range(num_segments):
        # 位置控制点：取 b_pts 的滑动窗口 [i, i+1, i+2, i+3]
        # 当 i=0 时，取的是 [vStart, P0, P1, P2]，曲线将精确从 P0 走到 P1
        p0, p1, p2, p3 = b_pts[i], b_pts[i+1], b_pts[i+2], b_pts[i+3]
        
        # 姿态控制点：直接对应当前段的起点和终点
        r_start = raw_rot[i]
        r_end = raw_rot[i+1]
        
        # 构造 Slerp 插值器
        # Scipy 的 Slerp 自动处理最短路径 (Shortest Path)，无需手动点积检查
        key_rots = R.from_quat([r_start, r_end])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)

        for j in range(resolution):
            t = j / resolution
            
            # 位置插值
            curr_pos = b_spline_compute(t, p0, p1, p2, p3)
            
            # 姿态插值
            curr_rot = slerp([t]).as_quat()[0] # as_quat 返回是 (N,4)，取第0个
            
            final_path.append({'pos': curr_pos, 'rot': curr_rot})

    # 5. 补入最后一个点 (确保终点绝对精确)
    final_path.append({'pos': raw_pos[-1], 'rot': raw_rot[-1]})
    
    return final_path