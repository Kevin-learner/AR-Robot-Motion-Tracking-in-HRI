import numpy as np
from tf.transformations import quaternion_matrix

# ==========================================
# 工具中心点 (TCP) 补偿工具包
# ==========================================
TOOL_LENGTH = 0.1  # ⚠️ 请用尺子测量法兰盘到笔尖的真实距离（单位：米），比如 15cm = 0.15

def ee_to_tool_tip(ee_pos, ee_quat, tool_length=TOOL_LENGTH):
    """
    正向补偿：已知法兰盘 (EE) 位姿，推算笔尖 (Tool) 坐标。
    用于：读取真实机器人位置，发给 Unity 或录制保存。
    """
    ee_pos = np.array(ee_pos)
    # 1. 四元数转旋转矩阵
    rot_mat = quaternion_matrix(ee_quat)[:3, :3]
    # 2. 提取局部 Z 轴在世界坐标系下的朝向 (假设笔是沿着法兰 Z 轴安装的)
    local_z_axis = rot_mat[:, 2]
    # 3. 笔尖位置 = 法兰位置 + Z轴方向 * 笔长
    tool_pos = ee_pos + local_z_axis * tool_length
    
    return tool_pos, ee_quat  # 姿态通常与法兰盘保持一致

def tool_tip_to_ee(tool_pos, tool_quat, tool_length=TOOL_LENGTH):
    """
    逆向补偿：已知期望的笔尖 (Tool) 位姿，推算底层的法兰盘 (EE) 坐标。
    用于：接收 Unity 发来的笔尖轨迹，转换为底层的电机执行目标。
    """
    tool_pos = np.array(tool_pos)
    # 1. 四元数转旋转矩阵
    rot_mat = quaternion_matrix(tool_quat)[:3, :3]
    # 2. 提取局部 Z 轴方向
    local_z_axis = rot_mat[:, 2]
    # 3. 法兰位置 = 笔尖位置 - Z轴方向 * 笔长
    ee_pos = tool_pos - local_z_axis * tool_length
    
    return ee_pos, tool_quat