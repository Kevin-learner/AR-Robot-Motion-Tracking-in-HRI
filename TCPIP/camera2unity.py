import numpy as np
from scipy.spatial.transform import Rotation as R
def transform_points(points, T):
    """
    Transform the Nx3 dot matrix using the 4x4 transformation matrix T and output Nx3.
    If a point is (0, 0, 0), it remains unchanged after the transformation.
    """
    # 🌟 1. 最关键的一步：强制将列表转为 NumPy 数组！
    points_arr = np.asarray(points, dtype=np.float64)
    
    # 🌟 2. 判空保护
    if points_arr.size == 0:
        return points_arr

    # 如果传进来的是一维数组，转成二维
    if points_arr.ndim == 1:
        points_arr = points_arr.reshape(-1, 3)

    transformed = np.zeros_like(points_arr)

    # 3. 此时 points_arr 是数组，==0 会进行逐元素的矩阵比较，不再报错！
    mask = ~np.all(points_arr == 0, axis=1)

    # 4. 只转换非 (0,0,0) 的点
    if np.any(mask):
        points_nonzero = points_arr[mask]
        ones = np.ones((points_nonzero.shape[0], 1))
        points_hom = np.hstack((points_nonzero, ones))
        
        # 矩阵乘法 T @ points
        transformed_nonzero = (T @ points_hom.T).T[:, :3]
        transformed[mask] = transformed_nonzero

    return transformed

def get_camera_to_robot_matrix(ee_pos, ee_quat, EE_T_C):
    """
    计算从相机坐标系到 ROBOT 坐标系的 4x4 变换矩阵
    :param ee_pos: 机械臂末端位置 [x, y, z]
    :param ee_quat: 机械臂末端四元数 [qx, qy, qz, qw] (注意 scipy 默认是 xyzw 顺序)
    :param EE_T_C: 4x4 末端到相机的变换矩阵
    :return: 4x4 相机到 robot 的变换矩阵 Robot_T_C
    """
    
    # 1. 
    EE_T_C = EE_T_C

    # 2.
    ROBOT_T_EE = np.eye(4)
    # 将四元数转为 3x3 旋转矩阵 (注意你的四元数排列顺序，这里假设是 [x, y, z, w])
    rot_matrix = R.from_quat(ee_quat).as_matrix() 
    ROBOT_T_EE[:3, :3] = rot_matrix
    ROBOT_T_EE[:3, 3] = ee_pos

    # 3. ROBOT_T_C
    ROBOT_T_C = ROBOT_T_EE @ EE_T_C

    return ROBOT_T_C

def points_camera_to_robot(points_camera, ee_pos, ee_quat, EE_T_C):
    ROBOT_T_C = get_camera_to_robot_matrix(ee_pos, ee_quat, EE_T_C)
    points_robot = transform_points(points_camera, ROBOT_T_C)
    return points_robot.tolist()

def points_camera_to_unity(points_camera, ee_pos, ee_quat, EE_T_C, UNITY_T_ROBOT):
    ROBOT_T_C = get_camera_to_robot_matrix(ee_pos, ee_quat, EE_T_C)

    F_z = np.eye(4)
    F_z[2, 2] = -1.0

    UNITY_T_C = UNITY_T_ROBOT @ F_z @ ROBOT_T_C

    points_unity = transform_points(points_camera, UNITY_T_C)

    return points_unity.tolist()

def get_rotation_matrix_scipy(rx_deg, ry_deg, rz_deg, seq='xyz'):
    """
    输入绕 X, Y, Z 轴的旋转角度 (度数)，输出 3x3 旋转矩阵
    :param rx_deg: 绕 X 轴旋转角度 (Degree)
    :param ry_deg: 绕 Y 轴旋转角度 (Degree)
    :param rz_deg: 绕 Z 轴旋转角度 (Degree)
    :param seq: 旋转顺序，默认 'xyz' (外旋) 或 'XYZ' (内旋)。机器人常用 'xyz' 或 'zyx'
    :return: 3x3 numpy 旋转矩阵
    """
    # 1. 初始化一个 4x4 的单位矩阵 (对角线为1，其余为0)
    T = np.eye(4)
    
    # 2. 计算 3x3 旋转矩阵
    rot_matrix = R.from_euler(seq, [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    
    # 3. 把旋转矩阵塞进左上角 3x3 区域
    T[:3, :3] = rot_matrix
    
    
    return T