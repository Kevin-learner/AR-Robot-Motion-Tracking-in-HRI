import numpy as np
from scipy.spatial.transform import Rotation as R

def get_camera_to_unity_matrix(ee_pos, ee_quat, EE_T_C, T_M):
    """
    计算从相机坐标系到 Unity 坐标系的 4x4 变换矩阵
    :param ee_pos: 机械臂末端位置 [x, y, z]
    :param ee_quat: 机械臂末端四元数 [qx, qy, qz, qw] (注意 scipy 默认是 xyzw 顺序)
    :param EE_T_C: 4x4 末端到相机的变换矩阵
    :param T_M: 4x4 机械臂基座到 Unity 的变换矩阵
    :return: 4x4 相机到 Unity 的变换矩阵 C_T_unity
    """
    
    # 1. 计算 C_T_EE (求逆)
    # 假设 EE_T_C 是把末端转相机的矩阵，如果是反过来的，去掉 np.linalg.inv 即可
    C_T_EE = np.linalg.inv(EE_T_C) 

    # 2. 计算 ee_T_robot (把末端位姿转成 4x4 矩阵)
    ee_T_robot = np.eye(4)
    # 将四元数转为 3x3 旋转矩阵 (注意你的四元数排列顺序，这里假设是 [x, y, z, w])
    rot_matrix = R.from_quat(ee_quat).as_matrix() 
    ee_T_robot[:3, :3] = rot_matrix
    ee_T_robot[:3, 3] = ee_pos

    # 3. 机械臂到 Unity 的矩阵 (已知)
    robot_T_unity = T_M

    # 4. 链式相乘得到最终矩阵 C_T_unity
    # 公式：P_unity = robot_T_unity * ee_T_robot * C_T_EE * P_cam
    C_T_unity = robot_T_unity @ ee_T_robot @ C_T_EE

    return C_T_unity