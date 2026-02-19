import numpy as np
from scipy.spatial.transform import Rotation as R

R_base_down = R.from_euler('x', 180, degrees=True)

def unity2robot_transform(unity_pos, T_M):
    """
    将 Unity 坐标转换为机器人坐标
    
    Args:
        unity_pos: (x, y, z) 元组或列表, Unity 中的目标坐标 (单位: 米)
        T_M: 4x4 numpy 数组, 之前校准得到的变换矩阵 (Robot -> Unity)
        
    Returns:
        np.array([x, y, z]): 机器人的目标坐标
    """
    if T_M is None:
        print("❌ T_M is None. Cannot transform.")
        return None

    try:
        # 1. 计算 T_M 的逆矩阵 (从 Unity -> Robot)
        # T_M 是 Robot->Unity，所以它的逆矩阵就是 Unity->Robot
        T_unity_to_robot = np.linalg.inv(T_M)
        
        # 2. 将 Unity 坐标转换为齐次坐标 [x, y, z, 1]
        p_unity_homogeneous = np.array([unity_pos[0], unity_pos[1], unity_pos[2], 1.0])
        
        # 3. 矩阵乘法: P_robot = T_inv * P_unity
        p_robot_homogeneous = T_unity_to_robot @ p_unity_homogeneous
        
        # 4. 取出前三位 [x, y, z]
        robot_pos = p_robot_homogeneous[:3]
        
        # 5. z=-z
        robot_pos[2] = -robot_pos[2]
        return robot_pos

    except np.linalg.LinAlgError:
        print("❌ Matrix inversion failed. T_M might be singular.")
        return None
    except Exception as e:
        print(f"❌ Transformation error: {e}")
        return None
    
def robot2unity_transform(point_3d, transformation_matrix):
    """应用变换矩阵 P_new = T * P_old"""
    p_homogeneous = np.append(point_3d, 1)
    return (transformation_matrix @ p_homogeneous)[:3]

def transform_unity_rot_to_robot(u_quat, T_M):
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    # 1. 提取 Unity 的原始欧拉角 (XYZ 顺序)
    # 这一步得到 Unity Inspector 面板里的数值概念
    r_unity = R.from_quat(u_quat)
    u_euler = r_unity.as_euler('xyz', degrees=True)
    u_x, u_y, u_z = u_euler[0], u_euler[1], u_euler[2]

    # 2. 【暴力映射】根据你的要求直接赋值
    # 你的规则：
    # Unity +y  ->  Robot -z (自旋)
    # Unity +x  ->  Robot +y (左右摆)
    # Unity +z  ->  Robot -x (前后摆)
    
    # 构造机器人的相对旋转量 (Delta)
    delta_x =  u_z  # u_z -> +r_x
    delta_y = -u_x  # u_x -> -r_y
    delta_z =  u_y  # u_y -> +r_z 

    # 3. 生成相对旋转对象
    # 注意：这里我们认为这些旋转是相对于“当前末端坐标系”的
    R_delta = R.from_euler('xyz', [delta_x, delta_y, delta_z], degrees=True)

    # 4. 定义基准姿态 (垂直向下)
    # Franka 标准：[180, 0, 0]
    R_base = R.from_euler('xyz', [180, 0, 0], degrees=True)

    # 5. 组合旋转
    # 逻辑：基准姿态 * 相对旋转 (让旋转发生在末端本地坐标系上)
    # 这样当你绕 Unity 的 Y 轴转时，机器人就会绕它自己的 Z 轴(指向地面的轴)转
    R_final = R_base * R_delta
    
    # 6. 导出结果
    robot_quat = R_final.as_quat()
    robot_rpy = R_final.as_euler('xyz', degrees=True)

    return {
        'robot_quat': robot_quat,
        'robot_rpy': robot_rpy,
        'raw_rpy': u_euler
    }