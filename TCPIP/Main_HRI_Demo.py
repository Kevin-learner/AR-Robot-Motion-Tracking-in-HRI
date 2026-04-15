import socket
import struct
import numpy as np
import yaml
import os
import sys
import time
import cv2
from scipy.spatial.transform import Rotation as R

# netsh interface portproxy add v4tov4 listenaddress=192.168.137.1 listenport=[本地监听端口] connectaddress=[机械臂的Tailscale IP] connectport=[机械臂服务端口]

# -------------------------------------------------
# 1. 基础配置与加载
# -------------------------------------------------
def load_config(path="config.yaml"):
    default_config = {
        'tcp': {'host': '0.0.0.0', 'port': 8848},
        'alignment': {'aruco_path': 'Realsense_Aruco.txt'},
        'robot': {'position_file': 'robotPosition.txt'},
        'recording': {'output_file': 'calibration_data_recorded.txt'}, # 默认保存路径
        'handeyecalibration': {
            'EE_T_C': [  # 默认的末端到相机的变换矩阵 (T_EE_C)，可以在 config.yaml 中覆盖
                [0.7029200982,  -0.7111337457,  0.01386113997,  0.0115],
                [0.7112536662,  0.7028988485,  -0.007171549375,  -0.065],
                [-0.004643048551,  0.01489981281,  0.9998782114,  0.032],
                [0,  0,  0,  1]
            ]
        }
    }
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                user_config = yaml.safe_load(file)
                # 简单的合并逻辑，确保 key 存在
                for section in user_config:
                    if section in default_config:
                        default_config[section].update(user_config[section])
            return default_config
    except Exception as e:
        print(f"⚠️ Config load failed: {e}, using default.")
    return default_config

# 加载配置
config = load_config()
HOST = config['tcp']['host']
PORT = config['tcp']['port']
ROBOT_POS_FILE = config['robot']['position_file']
RECORD_FILE = config['recording']['output_file'] #  保存路径
EE_T_C = np.array(config['handeyecalibration']['EE_T_C']) # 从配置加载 EE_T_C

# import calibration module
try:
    from compute_alignment import align_with_realsense
except ImportError:
    print("⚠️ Warning: compute_alignment.py not found. Calculation will be skipped.")
    align_with_realsense = None

# import robot position listener
try:
    import robotPositionListener
except ImportError:
    print("⚠️ Warning: robotPostionListener.py not found. Listening will be skipped.")
    robotPositionListener = None

try:
    import robot_unity_transformation as rut
except ImportError:
    print("⚠️ Warning: robot_unity_transformation.py not found. No way to transform.")
    rut = None

try:
    from robotController import RobotController
except ImportError:
    print("⚠️ Warning: robotController.py not found. No way to move robot.")
    robotController = None

try:
    from forceController import ForceController
except ImportError:
    print("⚠️ Warning: forceController.py not found. No way to apply forces.")
    forceController = None

try:
    import pathInterpolation
except ImportError:
    print("⚠️ Warning: pathInterpolation.py not found. No way to generate paths.")
    pathInterpolation = None

try:
    from videoSender import VideoSender
except ImportError:
    print("⚠️ Warning: videoSender.py not found. No way to send video.")
    VideoSender = None

try:
    from BodyPointCloud_dual import Body3DSkeletonProcess_dual
    print("✅ BodyPointCloud 导入完毕！")
except ImportError:
    print("⚠️ Warning: BodyPointCloud_dual.py not found. Skeleton tracking disabled.")
    Body3DSkeletonProcess_dual = None

try:
    from tool_tip_ee_transformation import ee_to_tool_tip
except ImportError:
    print("⚠️ Warning: tool_tip_ee_transformation.py not found. No way to compensate TCP.")
    ee_to_tool_tip = None

try:
    from tool_tip_ee_transformation import tool_tip_to_ee
except ImportError:
    print("⚠️ Warning: tool_tip_ee_transformation.py not found. No way to compensate TCP.")
    tool_tip_to_ee = None

try:
    import camera2unity
except ImportError:
    print("⚠️ Warning: camera2unity.py not found. ")
    camera2unity = None

try:
    import RvisSkeletonBroacaster
except ImportError:
    print("⚠️ Warning: RvisSkeletonBroacaster.py not found. ")
    camera2unity = None
# -------------------------------------------------
# 2. 核心与辅助函数
# -------------------------------------------------
def recv_exact(conn, num_bytes):
    """
    工业级防 TCP 拆包/粘包接收函数
    """
    buffer = bytearray()
    while len(buffer) < num_bytes:
        # 还要收多少，就去水管里读多少
        packet = conn.recv(num_bytes - len(buffer))
        if not packet:
            return None # 如果中间断网了，返回 None
        buffer.extend(packet)
    return bytes(buffer)

def send_T_M(conn, T_M):
    """发送变换矩阵 T_M"""
    try:
        flat_T_M = T_M.astype(np.float32).flatten()
        conn.sendall(b't' + flat_T_M.tobytes())
        print("[TCP] ✅ Sent Transformation Matrix (T_M).")
    except Exception as e:
        print(f"[TCP] ❌ Error sending T_M: {e}")

def read_and_parse_robot_txt(file_path):
    """解析机器人坐标文件"""
    if not os.path.exists(file_path):
        print(f"⚠️ Robot file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            if 'transform' in data and 'translation' in data['transform']:
                t = data['transform']['translation']
                return np.array([float(t['x']), float(t['y']), float(t['z'])])
            elif 'translation' in data:
                t = data['translation']
                return np.array([float(t['x']), float(t['y']), float(t['z'])])
    except Exception:
        pass 
    return None



def send_robot_ball_position(conn, robot_raw_pos, T_M):
    """计算并发送小球坐标"""
    try:
        unity_pos = rut.robot2unity_transform(robot_raw_pos, T_M)

        header = b'b'
        payload = struct.pack('<fff', unity_pos[0], unity_pos[1], unity_pos[2])
        full_packet = header + payload

        conn.sendall(full_packet)
        print(f"   -> Sent Ball Pos: {unity_pos}")

        # ==================== 调试打印区 ====================
        print("-" * 50)
        print(f"DEBUG: 发送球坐标数据")
        print(f"  -> 逻辑坐标 (Unity): X={unity_pos[0]:.4f}, Y={unity_pos[1]:.4f}, Z={unity_pos[2]:.4f}")
        
        # 打印十六进制，方便与 Unity 端逐字节比对
        # b'b' 的十六进制是 62
        hex_data = full_packet.hex(' ')
        print(f"  -> 原始字节流 (Hex): {hex_data}")
        print("-" * 50)
        # ===================================================
    except Exception as e:
        print(f"Error sending ball pos: {e}")

def save_recorded_point(filename, index, raw_pos):
    """
    [优化版] 读取 -> 更新 -> 重写
    确保每个 Index 只存在一份数据（最新的那份）
    """
    data_map = {}
    
    # 1. 如果文件存在，先读取旧数据
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    data_map = loaded
        except Exception:
            pass # 如果文件损坏或为空，就从头开始

    # 2. 更新当前点的坐标 (直接覆盖旧的 key)
    # raw_pos 是 numpy array，转为 list
    data_map[index] = raw_pos.tolist()

    try:
        # 3. 重新写入整个文件 (覆盖模式 'w')
        with open(filename, 'w', encoding='utf-8') as f:
            # 为了美观，我们按序号排序写入
            for key in sorted(data_map.keys()):
                f.write(f"{key}: {data_map[key]}\n")
                
        print(f"   💾 Point {index} Updated/Saved to {filename}")
        
    except Exception as e:
        print(f"   ❌ Save failed: {e}")


def save_tm_matrix(tm, path="tm_matrix.txt"):
    """将 T_M 矩阵保存为易读的文本文件"""
    try:
        # 保存为文本，保持 6 位小数，方便查看
        np.savetxt(path, tm, fmt='%.6f')
        print(f"💾 [System] T_M matrix saved (TXT) to {path}")
    except Exception as e:
        print(f"❌ [System] Failed to save T_M: {e}")

def load_tm_matrix(path="tm_matrix.txt"):
    """从本地文本文件加载 T_M 矩阵"""
    if os.path.exists(path):
        try:
            tm = np.loadtxt(path)
            print(f"✅ [System] T_M matrix loaded from {path}")
            # 打印一下确认加载的内容
            print(tm)
            return tm
        except Exception as e:
            print(f"⚠️ [System] Found T_M file but failed to load: {e}")
    return None


def send_skeleton_data(conn, send_coords):
    """
    将 17 个骨骼关键点 (51个float) 打包发送给客户端
    数据格式: b's' + 51个小端序float (共 1 + 204 = 205 bytes)
    """
    try:
        # 1. 展平列表: 把 [(x1,y1,z1), (x2,y2,z2)...] 变成 [x1, y1, z1, x2, y2, z2...]
        flat_coords = [coord for pt in send_coords for coord in pt]
        
        # 2. 检查数据长度是否正确 (17 * 3 = 51)
        if len(flat_coords) != 51:
            # 如果进来的数据是59个点，我们自动截断到前17个
            if len(flat_coords) >= 51:
                flat_coords = flat_coords[:51]
            else:
                print(f"⚠️ Skeleton data length mismatch. Expected >=51, got {len(flat_coords)}")
                return

        # 3. 打包: 's' 是包头，'<51f' 表示 51 个小端序的单精度浮点数
        header = b's'
        payload = struct.pack('<' + 'f' * len(flat_coords), *flat_coords)
        
        # 4. 发送
        conn.sendall(header + payload)
        # print(f"  -> Sent Skeleton Frame: 17 joints.") # 频率太高可以注释掉这行
        
    except Exception as e:
        print(f"❌ Error sending skeleton pos: {e}")


def check_reach_intent(skeleton_coord_camera):
    """
    通过 3D 骨骼坐标判断人是否在“伸手指向桌面”
    """
    if skeleton_coord_camera is None or len(skeleton_coord_camera) < 8:
        return False
        
    # 假设你的骨骼点索引：1-脖子, 2-右肩, 3-右肘, 4-右手腕, 5-左肩, 6-左肘, 7-左手腕
    # 请根据你实际的模型 (比如 COCO 17点 或 MediaPipe) 调整 index！
    idx_neck = 1
    idx_r_shoulder = 2
    idx_r_wrist = 4
    idx_l_shoulder = 5
    idx_l_wrist = 7
    
    neck = skeleton_coord_camera[idx_neck]
    r_shoulder = skeleton_coord_camera[idx_r_shoulder]
    r_wrist = skeleton_coord_camera[idx_r_wrist]
    l_shoulder = skeleton_coord_camera[idx_l_shoulder]
    l_wrist = skeleton_coord_camera[idx_l_wrist]
    
    # 1. 计算手腕到对应肩膀的三维欧氏距离 (代表手臂伸展程度)
    r_arm_extend = np.linalg.norm(r_wrist - r_shoulder)
    l_arm_extend = np.linalg.norm(l_wrist - l_shoulder)
    
    # 2. 判断手腕的深度 (Z轴) 是否明显比身体更往前
    # 你的 F_z 设定可能导致相机的 Z 是指向人体的（Z越小越靠近相机），请根据实际输出的符号测试
    # 这里假设：伸手时，手腕的 Z 会比肩膀的 Z 更小（更靠近镜头）或者具有显著差异
    z_diff_r = r_shoulder[2] - r_wrist[2] 
    z_diff_l = l_shoulder[2] - l_wrist[2]
    
    # === 判定条件 (阈值需要你在实际运行时稍微调试一下) ===
    # 假设：手臂伸展长度大于 0.45 米，且 Z 轴往前伸出超过 0.3 米
    EXTEND_THRESHOLD = 0.45 
    Z_FORWARD_THRESHOLD = 0.30
    
    right_reaching = (r_arm_extend > EXTEND_THRESHOLD) and (abs(z_diff_r) > Z_FORWARD_THRESHOLD)
    left_reaching = (l_arm_extend > EXTEND_THRESHOLD) and (abs(z_diff_l) > Z_FORWARD_THRESHOLD)
    
    return right_reaching or left_reaching

# -------------------------------------------------
# 3. 主循环
# -------------------------------------------------
def main():
    print("🚀 Starting TCP/IP Server..")
    #Initialize the video sender
    sender = VideoSender(port=8849)

    #Initialize the robot controller
    robot = None
    # robot_force = None

    #Initialize the robot listener
    robot_listener = robotPositionListener.RobotPositionListener(port=5006)

    #Initialize the RVIZ visualizer
    rviz_broadcaster = RvisSkeletonBroacaster.RVizSkeletonBroadcaster(frame_id="panda_link0")


    sSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    T_M = None 

    is_skeleton_streaming = False

    is_robot_state_streaming = False
    robot_stream_rate = 10.0  # 30Hz
    last_robot_stream_time = 0.0

    # [新增] 获取缓存文件路径
    TM_CACHE_PATH = config['alignment'].get('tm_cache_file', 'tm_matrix.npy')
    
    # [新增] 启动时尝试自动加载
    T_M = load_tm_matrix(TM_CACHE_PATH) 
    if T_M is not None:
        print("🚀 [System] Ready to go without manual calibration.")
    else:
        print("ℹ️ [System] No calibration cache found. Calibration required.")

    # ===  HRI 状态机初始化 ===
    is_HRI_Demo = False  # 是否进入 Demo 模式的总开关
    STATE_IDLE = -1          # 待机状态
    STATE_INIT = 0           # 0. 机械臂前往初始姿态
    STATE_TRACKING = 1       # 1. 跟踪人体，维持在视野中央
    STATE_CHECK_INTENT = 2   # 2. 识别意图 
    STATE_SCAN_OBJECTS = 3   # 3. 扫描桌面物品
    
    current_hri_state = STATE_IDLE
    hri_start_time = 0.0     # 记录状态切换的时间，用于非阻塞等待
    
    # 定义机械臂初始观测位姿 (请替换为你实际的关节角或笛卡尔坐标)
    INIT_POSE = [0.4, 0.0, 0.4, 3.14, 0.0, 0.0] 
    
    # 1. State Tracking 的 PID 参数 (需要调试)
    KP_X = 0.5 
    KP_Y = 0.5
    DEADZONE_M = 0.05 # 5cm死区，人在画面中心 5cm 内机械臂不动

    LOOK_AT_TABLE_POSE = [0.45, 0.0, 0.35, 3.14, 0.5, 0.0]
    # ===============================

    try:
        sSock.bind((HOST, PORT))
        print(f'[TCP] Listening on {HOST}:{PORT}...')
        sSock.listen(1)
        
        conn, addr = sSock.accept()
        print(f'[TCP] ✅ HoloLens Connected: {addr[0]}:{addr[1]}')

        conn.setblocking(False)

        while True:
            header = None

            try:
                header_byte = conn.recv(1)
                if header_byte:
                    header = header_byte.decode('utf-8', errors='ignore')
            except BlockingIOError:
                pass # 没数据，跳过
            except Exception as e:
                print(f"连接异常: {e}")
                break
            
            if header in ['d', 'r', 'b', 'm', 'p', 'v', 'f']:
                print(f"\n[TCP] Received Header: '{header}'")
                conn.setblocking(True)
            # ===============================================
            # CASE 'd': 校准数据 (Calibration)
            # ===============================================
                if header == 'd':

                    print("[TCP] Header 'd': Receiving calibration points...")
                    count_bytes = recv_exact(conn, 4)
                    if not count_bytes: break
                    num_points = struct.unpack("<i", count_bytes)[0]
                    
                    total_bytes = num_points * 3 * 4
                    data_bytes = recv_exact(conn, total_bytes)
                    if not data_bytes: break

                    if num_points == 5 and align_with_realsense:
                        float_data = np.frombuffer(data_bytes, dtype='<f4')
                        points3d = float_data.reshape((num_points, 3))
                        
                        print(f"   -> Received 5 points. Calculating T_M...")
                        
                        print(f"📍 Received {num_points} Calibration Points (HoloLens Ground Truth):")
                        for i, pt in enumerate(points3d):
                            print(f"   Index {i}: [{pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}]")

                        try:
                            T_M = align_with_realsense(points3d, RECORD_FILE)
                            print("\n" + "="*30)
                            print(f"🎉 T_M Calculated!\n{T_M}")
                            print("="*30 + "\n")
                            send_T_M(conn, T_M)

                            # [修改点] 计算成功后立即保存到本地
                            save_tm_matrix(T_M, TM_CACHE_PATH)

                        except Exception as e:
                            print(f"❌ Calculation Error: {e}")
                    else:
                        print(f"⚠️ Expected 5 points, got {num_points}. Skipping.")


                # ===============================================
                # CASE 'r': 记录数据 (Record)
                # ===============================================
                elif header == 'r':
                    print("[TCP] Header 'r': Recording Point...")
                    
                    # 读取 4 字节整数 (Unity 发来的 Index)
                    idx_bytes = recv_exact(conn, 4)
                    if idx_bytes:
                        point_index = struct.unpack('<i', idx_bytes)[0]
                        
                        # 🌟 1. 获取机器人 EE 的绝对位姿 (位置 + 姿态)
                        # 注意：你需要确保你的 get_position() 能同时返回 pos 和 quat
                        # 假设返回格式为 (ee_pos, ee_quat)，如果你现有的函数只返回 pos，你需要修改它以获取 TF 的 rotation
                        ee_pos, ee_quat = robot_listener.get_current_pose() 
                        
                        save_recorded_point(RECORD_FILE, point_index, ee_pos)
                        
                        # if ee_pos is not None and ee_quat is not None:
                        #     # 🌟 2. 正向补偿：推算真实笔尖坐标
                        #     #tool_pos, _ = ee_to_tool_tip(ee_pos, ee_quat)
                            
                        #     # 3. 保存到文件 (保存笔尖坐标)
                        #     save_recorded_point(RECORD_FILE, point_index, tool_pos)
                        #     print(f"   ✅ 已记录笔尖坐标 {tool_pos} 到索引 {point_index}")
                        # else:
                        #     print("   ❌ Failed to read robot position for recording.")
                    else:
                        print("   ⚠️ Received 'r' header but failed to read index.")

                # ===============================================
                # CASE 'b': 请求小球位置 (Ball)
                # ===============================================
                elif header == 'b':
                    if robot == None:
                        print(">> 正在连接机器人...")
                        robot = RobotController()

                    _ = recv_exact(conn, 1) # 读掉 Unity 的补位字节
                    print("[TCP] Header 'b': Requesting Robot Position...")
                    
                    if T_M is not None:
                        robot_pos = robot_listener.get_position()
                        if robot_pos is not None:
                            send_robot_ball_position(conn, robot_pos, T_M)
                        else:
                            print("   ❌ Failed to read robotPosition.txt")
                    else:
                        print("   ⚠️ T_M is None. Please calibrate first.")


                # ===============================================
                # CASE 'm': 移动目标 (Move Target) 
                # ===============================================
                # elif header == 'm':
                #     data = recv_exact(conn, 12) # 再读 12 个字节 (3个float)
                #     if len(data) == 12:
                #         # '<fff' 表示：小端序 (Little Endian), 3个 float
                #         ux, uy, uz = struct.unpack('<fff', data)
                #         print(f"Unity target received: X={ux}, Y={uy}, Z={uz}")

                #         if T_M is not None:
                #             # === 调用转换函数 ===
                #             target_robot_pos = rut.unity2robot_transform((ux, uy, uz), T_M)
                            
                #             if target_robot_pos is not None:
                #                 print(f"target in robot coordinate frame{target_robot_pos}")
                                
                #                 robot.move_to(target_robot_pos, speed=0.02)
                                
                #             else:
                #                 print("   ❌ 转换失败")
                #         else:
                #             print("   ⚠️ T_M 尚未计算，无法转换坐标。请先进行校准 (Header 'd')。")

                # ===============================================
                # CASE 'p': 接收位姿点序列 (Position + Orientation)
                # ===============================================
                elif header == 'p':
                    print("\n" + "="*50)
                    print("[TCP] 检测到 Header 'p'，开始解析位姿序列...")
                    
                    # 1. 读取点数 (4字节 Int)
                    count_bytes = recv_exact(conn, 4)
                    if not count_bytes: 
                        print("   ❌ 未收到点数数据")
                        break
                    
                    num_points = struct.unpack('<i', count_bytes)[0]
                    print(f"   -> 计划接收关键点数: {num_points}")

                    # 2. 读取数据包 (每个点 28 字节: 3 float pos + 4 float rot)
                    bytes_per_point = 28
                    total_bytes = num_points * bytes_per_point
                    data_bytes = recv_exact(conn, total_bytes)
                    if not data_bytes: 
                        print("   ❌ 未收到完整位姿数据")
                        break

                    # 3. 解析与转换数据
                    if T_M is not None:
                        # 使用 numpy 将 buffer 转换为 (N, 7) 的矩阵
                        raw_payload = np.frombuffer(data_bytes, dtype='<f4').reshape((num_points, 7))
                        
                        path_with_orientations = []
                        
                        for i in range(num_points):
                            # 提取位置
                            u_pos = raw_payload[i, 0:3]
                            # 提取四元数姿态 [qx, qy, qz, qw]
                            u_rot_quat = raw_payload[i, 3:7] 

                            # --- A. 位置转换 (Unity 坐标 -> 机器人坐标) ---
                            r_pos = rut.unity2robot_transform(u_pos, T_M) #tooltip
                            
                            # --- B. 姿态转换 (调用封装好的函数，内部处理镜像、TM及RPY计算) ---
                            # 该函数应返回一个字典，包含转换后的四元数和用于显示的 RPY
                            res = rut.transform_unity_rot_to_robot(u_rot_quat, T_M)

                            if r_pos is not None:
                                path_with_orientations.append({
                                    'pos': r_pos,
                                    'rot': res['robot_quat']  # 存入用于执行的机器人系四元数
                                })

                                # --- C. 日志打印：对比原始角度与变换后角度 ---
                                print(f"   [{i}] ---------------------------------------")
                                print(f"       位置: Unity {np.round(u_pos, 2)} -> 机器人 {np.round(r_pos, 3)}")
                                print(f"       姿态: 原始RPY(Unity): {np.round(res['raw_rpy'], 1)}°")
                                print(f"       姿态: 变换RPY(Robot): {np.round(res['robot_rpy'], 1)}°")
                                # 如果需要调试 XYZ 单位向量，可以打印 res['rhs_axes']

                        # 4. 生成带姿态插值的平滑路径
                        if len(path_with_orientations) >= 2:
                            print(f"\n   [Interpolation] 正在生成平滑路径...")
                            final_smooth_path = pathInterpolation.generate_smooth_path_with_orientation(
                                path_with_orientations, 
                                resolution=3
                            )
                            final_smooth_path.append(path_with_orientations[-1])
                            
                            # 5. 执行机器人运动
                            if robot is None: 
                                robot = RobotController()
                            
                            print(f"   🚀 开始执行，总插值点数: {len(final_smooth_path)}")
                            # 发送给机械臂执行
                            robot.execute_path(final_smooth_path, speed=0.02)

                            # 2. 🌟 关键：增加等待逻辑
                            # 假设目标点是路径的最后一个点
                            target_pos = final_smooth_path[-1]['pos'] 

                            print(" ⏳ 等待机械臂到达目标位置...")
                            while True:
                                # 获取机械臂当前的实时位置 (你可以从你的 robot_listener 获取)
                                current_pos = robot_listener.get_position() 
                                
                                if current_pos is not None:
                                    # 计算当前位置与目标位置的欧氏距离
                                    dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                                    
                                    # 如果距离小于 1cm (0.01m)，认为已到达
                                    if dist < 0.01: 
                                        break
                                        
                                time.sleep(0.1) # 每 100ms 检查一次，避免占用过多 CPU

                            try:
                                conn.sendall(b'm')
                                print("   ✅ [TCP] 机械臂运动完毕，已向 HoloLens 发送 'm' 解锁信号")
                            except Exception as e:
                                print(f"   ❌ 发送完成信号失败: {e}")
                    else:
                        print("   ⚠️ T_M 矩阵为空，请先进行校准发送 'c'！")
                    print("="*50 + "\n")

                # ===============================================
                # CASE 'f': 接收力控点序列 (Position + Orientation + Force)
                # ===============================================
                elif header == 'f':
                    print("\n" + "="*50)
                    print("[TCP] 检测到 Header 'f'，开始解析【力控】位姿序列...")
                    
                    # 1. 读取点数 (4字节 Int)
                    count_bytes = recv_exact(conn, 4)
                    if not count_bytes: 
                        print("   ❌ 未收到点数数据")
                        break
                    
                    num_points = struct.unpack('<i', count_bytes)[0]
                    print(f"   -> 计划接收力控关键点数: {num_points}")

                    # 2. 读取数据包 (每个点 32 字节: 3 pos + 4 rot + 1 force)
                    bytes_per_point = 32
                    total_bytes = num_points * bytes_per_point
                    data_bytes = recv_exact(conn, total_bytes)
                    if not data_bytes: 
                        print("   ❌ 未收到完整力控数据")
                        break

                    # 3. 解析与转换数据
                    if T_M is not None:
                        # 每一行: [x, y, z, qx, qy, qz, qw, force]
                        raw_payload = np.frombuffer(data_bytes, dtype='<f4').reshape((num_points, 8))
                        
                        path_with_force = []
                        
                        for i in range(num_points):
                            u_pos = raw_payload[i, 0:3]
                            u_rot_quat = raw_payload[i, 3:7] 
                            target_force = raw_payload[i, 7] # 🌟 提取目标力

                            # 坐标与姿态转换
                            r_pos = rut.unity2robot_transform(u_pos, T_M)
                            res = rut.transform_unity_rot_to_robot(u_rot_quat, T_M)

                            if r_pos is not None:
                                path_with_force.append({
                                    'pos': r_pos,
                                    'rot': res['robot_quat'],
                                    'force': float(target_force)  # 加入 force 字段
                                })

                                print(f"   [{i}] 位置:{np.round(r_pos, 3)} | 姿态RPY:{np.round(res['robot_rpy'], 1)}° | 目标力:{target_force:.1f}N")

                        # 4. 生成带姿态和【力】的平滑路径
                        if len(path_with_force) >= 2:
                            print(f"\n   [Interpolation] 正在生成力控平滑路径...")
                            
                            # 注意：你的插值函数需要更新，以支持 force 字段（见下文）
                            final_smooth_path = pathInterpolation.generate_smooth_path_with_orientation(
                                path_with_force, 
                                resolution=3
                            )
                            
                            final_smooth_path.append(path_with_force[-1])
                            # 5. 执行机器人运动
                            if robot is None: 
                                robot = RobotController()
                            
                            print(f"   🚀 开始执行力控轨迹，总插值点数: {len(final_smooth_path)}")
                            # 统一使用 execute_path，不再需要传 mode 参数
                            robot.execute_path(final_smooth_path, speed=0.02)

                            # 2. 🌟 关键：增加等待逻辑
                            # 假设目标点是路径的最后一个点
                            target_pos = final_smooth_path[-1]['pos'] 

                            print(" ⏳ 等待机械臂到达目标位置...")
                            while True:
                                # 获取机械臂当前的实时位置 (你可以从你的 robot_listener 获取)
                                current_pos = robot_listener.get_position() 
                                
                                if current_pos is not None:
                                    # 计算当前位置与目标位置的欧氏距离
                                    dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                                    
                                    # 如果距离小于 1cm (0.01m)，认为已到达
                                    if dist < 0.01: 
                                        break
                                        
                                time.sleep(0.1) # 每 100ms 检查一次，避免占用过多 CPU

                            try:
                                conn.sendall(b'm')
                                print("   ✅ [TCP] 机械臂力控运动完毕，已向 HoloLens 发送 'm' 解锁信号")
                            except Exception as e:
                                print(f"   ❌ 发送完成信号失败: {e}")
                    else:
                        print("   ⚠️ T_M 矩阵为空，请先进行校准发送 'c'！")
                    print("="*50 + "\n")

                # ===============================================
                # CASE 'v': 接收视频流与食指坐标 (Video + Finger)
                # ===============================================
                elif header == 'v':
                    print("\n" + "="*50)
                    print("[TCP] Header 'v': Receiving Video Stream + Finger Data...")
                    # 1. 读取传感器类型 (1 byte, 例如 'i' 表示红外)
                    sensor_type_bytes = recv_exact(conn, 1)
                    if not sensor_type_bytes: break
                    sensor_type = sensor_type_bytes.decode('utf-8', errors='ignore')

                    # 2. 读取食指坐标 (12 bytes -> 3 个 float)
                    pos_bytes = recv_exact(conn, 12)
                    if not pos_bytes: break
                    finger_x, finger_y, finger_z = struct.unpack('<fff', pos_bytes)

                    # 3. 读取图像数据长度 (4 bytes -> int)
                    len_bytes = recv_exact(conn, 4)
                    if not len_bytes: break
                    img_len = struct.unpack('>i', len_bytes)[0]

                    # 4. 读取全部图像数据
                    img_data = recv_exact(conn, img_len)
                    if not img_data: break

                    # 5. 解析图像并显示
                    img_array = np.frombuffer(img_data, dtype=np.uint8).copy() 
                    
                    frame = None
                    # 判断：如果我们发的是 Raw Bytes，长度会严格等于像素数
                    if img_len == 512 * 512:        # 短距相机分辨率 (Short-throw)
                        frame = img_array.reshape((512, 512))
                    elif img_len == 320 * 288:      # 长距相机分辨率 (Long-throw)
                        frame = img_array.reshape((288, 320))
                    else:
                        # 如果后续你改成了发 JPG 压缩流，就用 imdecode 解压
                        frame = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)

                    if frame is not None:
                        # 将接收到的食指坐标打印在画面上（便于直观调试）
                        text = f"Finger: X={finger_x:.2f} Y={finger_y:.2f} Z={finger_z:.2f}"
                        # 避免文字太暗看不清，针对灰度图用白色(255)显示
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
                        
                        cv2.imshow(f"HoloLens Stream ({sensor_type})", frame)
                        cv2.waitKey(1) # 必须有这一句，OpenCV 才能刷新窗口
                    else:
                        print(f"   ⚠️ 图像解码失败，接收长度: {img_len}")
                
                conn.setblocking(False) 
                print(f"[TCP] {header} 处理完毕，切回视频模式")

            elif header == 'S': 
                sender.is_streaming = True
                print("▶️ 开始传输")
            elif header == 'E': 
                sender.is_streaming = False
                print("⏹️ 停止传输")

            elif header == 'K': 
                is_skeleton_streaming = True
                print("▶️ 开始传输【骨架数据】")
            elif header == 'L': 
                is_skeleton_streaming = False
                print("⏹️ 停止传输【骨架数据】")
            
            elif header == 'J': 
                is_robot_state_streaming = True
                print("▶️ 开始以 30Hz 传输【全关节实时位姿】(Python端已转换)")
            elif header == 'H': 
                is_robot_state_streaming = False
                print("⏹️ 停止传输【全关节实时位姿】")

            elif header == 'O': 
                is_HRI_Demo = True
                print("Demo模式已开")
            elif header == 'P': 
                is_HRI_Demo = False
                print("Demo模式已关")

            # ===============   ================================
            # CASE 'x': 退出 (Exit)
            # ===============================================
            elif header == 'x':
                print("[TCP] Received Exit signal. Cleaning up...")
                
                TM_CACHE_PATH = config['alignment'].get('tm_cache_file', 'tm_matrix.txt')
                if os.path.exists(TM_CACHE_PATH):
                    os.remove(TM_CACHE_PATH)
                    print("🧹 Cache cleared.")
                break
        
            else:
                pass

            # --- 只有在开关打开时才执行发送函数 ---
            # --- 重点：视频转发逻辑 ---
            if sender.is_streaming:
                # 这个函数现在会自动找 A 电脑要图并转给 conn (HoloLens)
                sender.send_frame(conn, sensor_type='c')

            if is_skeleton_streaming and Body3DSkeletonProcess_dual:
                F_z = np.eye(4)
                F_z[2, 2] = -1.0
                
                    
                if robot_listener is not None:
                    # geting skeleton_coord_robot
                    ee_pos, ee_quat = robot_listener.get_current_pose()

                    # 注意：如果你的机械臂不是绕 X 轴旋转导致画面倾斜，请把 [0] 改成 [2] (Z轴) 或 [1] (Y轴)
                    from scipy.spatial.transform import Rotation as R
                    euler_angles = R.from_quat(ee_quat).as_euler('xyz', degrees=True)
                    current_ry = euler_angles[1]  # [1] 代表取出 Ry
                    
                    # 💡 设定基准角度：即相机画面正立时的 Ry 角度
                    BASELINE_RY = -47.6 
                    
                    # 💡 计算图像到底歪了多少度
                    roll_angle = current_ry - BASELINE_RY

                    skeleton_coord_camera, should_quit = Body3DSkeletonProcess_dual(
                        F_z,
                        use_dual_camera=False,
                        roll_angle = roll_angle
                        ) #remain right-handed
                    #print("gained skeleton data from camera")

                    skeleton_coord_robot = camera2unity.points_camera_to_robot(skeleton_coord_camera, ee_pos, ee_quat, EE_T_C)
                    #print("transformed to robot coord")
                    rviz_broadcaster.publish(skeleton_coord_robot)

                    if T_M is not None:
                        skeleton_coord_unity = camera2unity.points_camera_to_unity(skeleton_coord_camera, ee_pos, ee_quat, EE_T_C, T_M)
                        #print("transformed to unity")

                        if not should_quit and skeleton_coord_unity:
                            # 6. 截取前 17 个身体关键点 (丢掉后面的手部点)
                            body_only_coords = skeleton_coord_unity[:17]
                            
                            send_skeleton_data(conn, body_only_coords)
                            
                    else:
                        # 防止疯狂打印，可以加个简单的限流，或者只提醒一次
                        # print("⚠️ 骨架开关已开，但尚未进行相机校准 (T_M is None)！")
                        pass
            
            # --- 机械臂全关节实时状态流 (30Hz) ---
            if is_robot_state_streaming and robot_listener is not None:
                current_time = time.time()
                if current_time - last_robot_stream_time >= (1.0 / robot_stream_rate):
                    last_robot_stream_time = current_time
                    try:
                        # 假设 robot_listener 提供获取所有关节状态的方法
                        # 返回值应该是包含7个元素的列表，每个元素包含(angle, pos, quat)
                        joints_angles = robot_listener.get_joints() 
                        joints_positions = robot_listener.get_all_joint_positions() 
                        joints_quats = robot_listener.get_all_joint_quats()
                        
                        flat_data = []
                        
                        # 遍历 7 个关节，在 Python 端完成所有矩阵转换
                        for i in range(8):
                            angle = joints_angles[i]
                            raw_pos = joints_positions[i]
                            raw_quat = joints_quats[i]
                            
                            # 【核心】在这里提前应用 T_M 矩阵，并转换为 Unity 左手系
                            if T_M is not None:
                                unity_pos = rut.robot2unity_transform(raw_pos, T_M)
                                # 假设你的 rut 模块有这个对应姿态的转换函数
                                res = rut.transform_robot_rot_to_unity(raw_quat, T_M) 
                                unity_quat = res['unity_quat'] 
                            else:
                                # 如果还没校准，就先发原数据或全0
                                unity_pos = raw_pos
                                unity_quat = raw_quat
                                
                            flat_data.append(float(angle))
                            flat_data.extend([float(x) for x in unity_pos])
                            flat_data.extend([float(x) for x in unity_quat])
                            
                        # 打包发送: 'j' + 56个小端序 float (共 225 bytes)
                        header_j = b'j'
                        payload_j = struct.pack('<64f', *flat_data)

                        conn.setblocking(True)
                        conn.sendall(header_j + payload_j)
                        conn.setblocking(False)

                        
                    except Exception as e:
                        pass
            
            if is_HRI_Demo:
                # -1. 待机状态：等待 'O' 信号启动
                if current_hri_state == STATE_IDLE:
                    if robot is None:
                        robot = RobotController()
                    print("\n" + "="*50)
                    print("🤖 [HRI] 收到指令，启动人机交互流程！")
                    current_hri_state = STATE_INIT # 切入状态 0
                    print("="*50 + "\n")
                
                # 0. 初始位姿状态：机械臂前往预设的初始位置，等待 5 秒让用户观察
                elif current_hri_state == STATE_INIT:
                    # 刚进入 STATE_INIT 时，触发一次运动并记录时间
                    if hri_start_time == 0.0:
                        print("🔄 [HRI] 状态 0: 机械臂前往初始观测位姿...")
                        if robot is not None:
                            robot.move_to(INIT_POSE, speed=0.02)
                        hri_start_time = time.time() # 🌟 开始计时！
                        
                    # 每次循环检查时间是否够了，如果没够就直接跳过，继续发视频
                    else:
                        if time.time() - hri_start_time > 5.0:
                            print(" ✅ [HRI] 5秒已过，假设已到达初始位置，切入人体跟踪状态！")
                            current_hri_state = STATE_TRACKING
                            hri_start_time = 0.0 # 重置计时器，给以后的状态用
                
                # 1. 跟踪状态：持续接收骨架数据，计算误差并喂给 PID 控制器
                elif current_hri_state == STATE_TRACKING:
                    
                    # 确保在前面的逻辑中，skeleton_coord_camera 已经获取到了
                    if skeleton_coord_camera is not None and len(skeleton_coord_camera) > 1:
                        
                        # 【新增】：在这里高频检测“伸手”意图
                        if check_reach_intent(skeleton_coord_camera):
                            print(" 🎯 [HRI] 检测到伸手意图！准备看向桌面...")
                            # 1. 停止 PID 跟踪
                            if robot is not None:
                                robot.stop_tracking()
                            # 2. 切入状态 2
                            current_hri_state = STATE_CHECK_INTENT
                            hri_start_time = 0.0 # 重置计时器
                            continue # 直接跳过本次循环后续的跟踪代码
                        
                        # 提取胸口或脖子的 3D 坐标 (假设 Index 1 是脖子)
                        # 注意：这是在相机坐标系下的 3D 坐标，单位通常是米 (m)
                        target_cam = skeleton_coord_camera[1] 
                        
                        cam_x = target_cam[0] # 相机画面左右偏差
                        cam_y = target_cam[1] # 相机画面上下偏差
                        # cam_z = target_cam[2] # 离相机的远近距离
                        
                        # 💡【极其关键：轴向映射】
                        # 你的机械臂 PID 是在 Base (基座) 坐标系下移动的。
                        # 我们必须把相机的左右上下，映射成基座的前后左右。
                        # 假设初始姿态下，相机正视前方 (Base X正向)，那么：
                        # - 人在画面偏右 (cam_x > 0) -> 机械臂需要向右移动 -> 对应 Base 的 -Y 方向
                        # - 人在画面偏下 (cam_y > 0) -> 机械臂需要向下移动 -> 对应 Base 的 -Z 方向
                        # (⚠️ 注意：这里的映射完全取决于你的实际硬件安装方向，请根据测试情况修改正负号和 XYZ 对应关系)
                        
                        base_err_x = 0.0       # 前后距离暂不跟踪，保持为0
                        base_err_y = -cam_x    # 将相机的左右映射给 Base 的 Y
                        base_err_z = -cam_y    # 将相机的上下映射给 Base 的 Z
                        
                        # 设置死区：如果人在画面中心 5cm (0.05m) 范围内，不要乱动防抖
                        if abs(base_err_y) < 0.05: base_err_y = 0.0
                        if abs(base_err_z) < 0.05: base_err_z = 0.0
                        
                        # 🚀 关键调用 2：高频喂给底层的 PID 误差
                        if robot is not None:
                            robot.update_tracking_error(base_err_x, base_err_y, base_err_z)
                            
                    else:
                        # 视野里没有识别到人，喂入 0 误差，机械臂会在原地悬停
                        if robot is not None:
                            robot.update_tracking_error(0.0, 0.0, 0.0)
                
    except Exception as e:
        print(f"[TCP] Server Error: {e}")

    finally:
        try:
            sender.release()
            conn.close()
            sSock.close()
        except:
            pass

if __name__ == "__main__":
    #print("=== Main_Calibration_Only.py ===")
    main()