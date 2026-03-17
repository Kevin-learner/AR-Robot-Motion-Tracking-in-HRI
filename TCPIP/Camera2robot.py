import numpy as np

def transform_camera_to_robot_base(points_camera, EE_T_C, Base_T_EE):
    """
    将相机捕捉到的 3D 骨骼关键点，转换到机器人基座(Base)坐标系。
    
    参数:
        points_camera (list/ndarray): Nx3 的骨骼关键点数组 (Camera坐标系)。
        EE_T_C (ndarray): 4x4 变换矩阵，从 Camera 到 EE (手眼校准矩阵)。
        Base_T_EE (ndarray): 4x4 变换矩阵，从 EE 到 Robot Base (机械臂当前位姿矩阵)。
                                
    返回:
        ndarray: Nx3 的骨骼关键点数组 (Robot Base坐标系)。
    """
    points = np.array(points_camera)
    
    # 防御性检查：如果没有数据，直接返回
    if len(points) == 0:
        return points
        
    # 1. 过滤无效点 (避免未检测到的 [0.0, 0.0, 0.0] 被平移到奇怪的地方)
    valid_mask = ~np.all(np.isclose(points, 0.0), axis=1)
    valid_points = points[valid_mask]
    
    # 如果当前帧全是无效点，直接返回原数组
    if len(valid_points) == 0:
        return points 
        
    # 2. 转换为齐次坐标：给 Nx3 的矩阵加上一列 1，变成 Nx4
    ones = np.ones((valid_points.shape[0], 1))
    points_homo = np.hstack((valid_points, ones))
    
    # 3. 计算核心变换矩阵
    # 链式乘法：Base_T_Camera = Base_T_EE * EE_T_Camera
    Base_T_C = np.dot(Base_T_EE, EE_T_C)
    
    # 4. 执行坐标系转换
    # 矩阵运算：P_base = Base_T_C * P_camera^T
    points_base_homo = np.dot(Base_T_C, points_homo.T).T
    
    # 5. 还原回 3D 笛卡尔坐标 (切片去掉最后那一列 1)
    points_base = points_base_homo[:, :3]
    
    # 6. 把计算好的有效坐标安全地塞回原格式中
    result_points = np.zeros_like(points)
    result_points[valid_mask] = points_base
    
    return result_points