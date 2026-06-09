import socket
import struct
import numpy as np
import yaml
import os
import sys
import time
import cv2
from scipy.spatial.transform import Rotation as R
from collections import deque
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor

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

executor = ThreadPoolExecutor(max_workers=1)

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
    import BodyPointCloud_dual
    from BodyPointCloud_dual import Body3DSkeletonProcess_dual, K_1
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

try:
    from gaze_interaction import get_gaze_point_cloud_intersection
    from gaze_interaction import PointCloudAccumulator
except ImportError:
    print("⚠️ Warning: gaze_interaction.py not found. No way to compute gaze interaction.")
    get_gaze_point_cloud_intersection = None
    PointCloudAccumulator = None

try:
    from request_grasps import request_grasps_from_graspnet
except ImportError:
    print("⚠️ Warning: request_grasps.py not found. No way to compute grasp pose.")


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
    通过 3D 骨骼坐标判断人是否在“伸手” (适配 YOLO 17点格式)
    """
    if skeleton_coord_camera is None or len(skeleton_coord_camera) < 11:
        return False
        
    pts = np.array(skeleton_coord_camera)
    
    # YOLO COCO 索引
    idx_nose = 0
    idx_l_shoulder = 5
    idx_r_shoulder = 6
    idx_l_wrist = 9
    idx_r_wrist = 10
    
    nose = pts[idx_nose]
    l_shoulder = pts[idx_l_shoulder]
    r_shoulder = pts[idx_r_shoulder]
    l_wrist = pts[idx_l_wrist]
    r_wrist = pts[idx_r_wrist]
    
    # 过滤掉丢失的点
    if np.allclose(l_shoulder, [0,0,0]) or np.allclose(r_shoulder, [0,0,0]):
        return False

    # 1. 计算手腕到肩膀的三维长度
    l_arm_extend = np.linalg.norm(l_wrist - l_shoulder) if not np.allclose(l_wrist, [0,0,0]) else 0.0
    r_arm_extend = np.linalg.norm(r_wrist - r_shoulder) if not np.allclose(r_wrist, [0,0,0]) else 0.0
    
    # 2. 判断手腕的 Z 深度 (手是否伸向了相机前方)
    # 伸得越前，Z 值通常越小或者呈现特定方向的差值
    z_diff_l = l_shoulder[2] - l_wrist[2]
    z_diff_r = r_shoulder[2] - r_wrist[2] 
    
    # 阈值：手臂伸长超过 0.35m，且向前伸超过 0.2m
    EXTEND_THRESHOLD = 0.35 
    Z_FORWARD_THRESHOLD = 0.20
    
    left_reaching = (l_arm_extend > EXTEND_THRESHOLD) and (abs(z_diff_l) > Z_FORWARD_THRESHOLD)
    right_reaching = (r_arm_extend > EXTEND_THRESHOLD) and (abs(z_diff_r) > Z_FORWARD_THRESHOLD)
    
    return left_reaching or right_reaching

def send_path_to_hololens(conn, points, T_M):
    """
    发送动态预测路径给 HoloLens 进行实时渲染 (Header: 'w')
    :param conn: TCP socket 连接对象
    :param points: 机器人坐标系下的三维点列表 (List of [x, y, z] 或 np.ndarray)
    :param T_M: 机器人到 Unity 左手系的对齐转换矩阵
    """
    try:
        # 1. 如果路径为空，发送 0 个点，通知 HoloLens 清空屏幕上的线
        if not points or len(points) == 0:
            header = b'w'
            payload = struct.pack('<i', 0)
            conn.setblocking(True)
            conn.sendall(header + payload)
            conn.setblocking(False)
            return

        # 2. 坐标系转换 (Robot -> Unity)
        unity_points = []
        for pt in points:
            # 使用你的辅助函数转为 Unity 坐标
            u_pt = rut.robot2unity_transform(pt, T_M)
            if u_pt is not None:
                unity_points.append(u_pt)

        # 3. 打包 Header 和 点的数量
        header = b'w'
        num_pts = len(unity_points)
        payload = struct.pack('<i', num_pts) # <i 代表小端序的 4字节整型(int)

        # 4. 打包所有的 XYZ 坐标点
        flat_coords = []
        for u_pt in unity_points:
            flat_coords.extend([float(u_pt[0]), float(u_pt[1]), float(u_pt[2])])
        
        # 将展平的坐标列表一次性打包成 float 字节流 (小端序)
        payload += struct.pack(f'<{len(flat_coords)}f', *flat_coords)

        # 5. 发送数据 (极其重要：发数据前开启阻塞，发完关闭，这是非阻塞TCP的核心稳健写法)
        conn.setblocking(True)
        conn.sendall(header + payload)
        conn.setblocking(False)

    except BlockingIOError:
        pass # 非阻塞模式下的系统级等待，可以直接忽略
    except Exception as e:
        print(f"⚠️ [TCP] 发送动态路径至 HoloLens 失败: {e}")

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
    sSock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    sSock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 20 * 1024 * 1024)
    sSock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

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
    STATE_GAZE_INTERSECTION = 4 # 4. gaze selection
    STATE_GRAB_OBJECT = 5       # 5. 抓取物品
    STATE_LOOKING_FOR_USER = 6     # 6. 寻找用户来拿
    STATE_TRACKING_AND_PASS = 7     # 7. 跟踪并递给用户
    
    has_completed_initial_scan = False

    current_hri_state = STATE_IDLE
    hri_start_time = 0.0     # 记录状态切换的时间，用于非阻塞等待
    handover_dwell_start = 0.0
    
    # 定义机械臂初始观测位姿 (请替换为你实际的关节角或笛卡尔坐标)
    INIT_POSE = [0.3572, 0.0073, 0.5586, -0.68, 0.27, -0.63, 0.26]    # 1. State Tracking 的 PID 参数 (需要调试)
    KP_X = 0.5 
    KP_Y = 0.5
    DEADZONE_M = 0.05 # 5cm死区，人在画面中心 5cm 内机械臂不动

    intent_history = deque(maxlen=30)
    LOOK_AT_TABLE_POSE = [0.6321, 0.2207, 0.5051, 0.92, -0.40, 0.01, 0.00] #[INFO] [1777634020.049898]: sent #1 UPDATED d=0.000000 xyz=(0.5792, 0.2653, 0.5375) Euler[Deg]=(Rx:-175.8, Ry:-3.4, Rz:-49.6) q=(0.91, -0.42, 0.04, -0.02)

    # 存放最新的全局坐标系下的眼动射线
    global_ray_origin = None 
    global_ray_hit = None
    current_point_cloud = None

    global_holo_hand_pos = None
    
    # ====== Multipointcloud test ======
    # 这里使用的是 [x, y, z, qx, qy, qz, qw] 或关节角，只要符合你的 robot.move_to 格式即可

    SCAN_START_POSE = [0.5792, 0.2653, 0.5375, 0.91, -0.42, 0.04, -0.02]
    SCAN_END_POSE   = [0.5621, -0.1739, 0.5375, -0.92, 0.38, -0.04, -0.01] #[INFO] [1777634168.026776]: sent #149 UPDATED d=0.000223 xyz=(0.5621, -0.1739, 0.5610) Euler[Deg]=(Rx:-179.5, Ry:-4.6, Rz:-45.0) q=(-0.92, 0.38, -0.04, -0.01)
    SCAN_STEPS = 6  # 直线上拍 4 张照片（起点、2个中间点、终点）

    SCAN_ARC_HEIGHT = 0.04

    scan_waypoints = []
    for i in range(SCAN_STEPS):
        ratio = i / (SCAN_STEPS - 1)
        
        # 1. 正常的线性插值
        interpolated_pose = [
            SCAN_START_POSE[j] + ratio * (SCAN_END_POSE[j] - SCAN_START_POSE[j])
            for j in range(len(SCAN_START_POSE))
        ]
        
        # 2. 🌟 核心魔法：将抛物线高度直接“烘焙”到 Z 轴坐标里！
        if SCAN_ARC_HEIGHT > 0.0:
            # 公式：4 * h * x * (1-x)，两端为 0，中间最高
            interpolated_pose[2] += SCAN_ARC_HEIGHT * 4.0 * ratio * (1.0 - ratio)
            
        scan_waypoints.append(interpolated_pose)

    scan_current_step = 0 # 记录走到第几个点了
  
    # 实例化建图累加器

    scene_mapper = PointCloudAccumulator(voxel_size=0.01) 
    # 🌟 2. 绑定按键回调：在 Open3D 窗口按 'C' 触发拍照标志位
    capture_flag = [False] # 使用列表包装，方便在回调函数中修改
    def key_callback_capture(vis):
        capture_flag[0] = True
        return False
        
    scene_mapper.vis.register_key_callback(ord('C'), key_callback_capture)
    scene_mapper.vis.register_key_callback(ord('c'), key_callback_capture)

    LOOK_USER_POSE = [0.6097, 0.0584, 0.5640, 0.69, -0.26, 0.61, -0.28]
    # [INFO] [1778077176.377578]: sent #8199 UPDATED d=0.000063 xyz=(0.6097, 0.0584, 0.5640) Euler[Deg]=(Rx:-98.2, Ry:-44.1, Rz:-80.3) q=(0.69, -0.26, 0.61, -0.28)
    READY_FOR_PASSING_POSE = [0.3319, -0.0112, 0.5835, -0.92, 0.39, -0.04, 0.02]
    #[INFO] [1779192781.952631]: sent #567 UPDATED d=0.000005 xyz=(0.3319, -0.0112, 0.5835) Euler[Deg]=(Rx:-176.0, Ry:-3.7, Rz:-46.4) q=(-0.92, 0.39, -0.04, 0.02)

    # ===============================

    try:
        sSock.bind((HOST, PORT))
        print(f'[TCP] Listening on {HOST}:{PORT}...')
        sSock.listen(1)
        
        conn, addr = sSock.accept()
        print(f'[TCP] ✅ HoloLens Connected: {addr[0]}:{addr[1]}')

        conn.setblocking(False)

        # ==========================================
        # 🛠️ 新增：在这里初始化打印计时器（放在 while True 外面）
        # ==========================================
        last_e_print_time = 0.0
        last_ray_print_time = 0.0
        last_hit_print_time = 0.0
        # ==========================================

        last_yolo_time = 0.0
        while True:
            while True:
                header = None

                try:
                    header_byte = conn.recv(1)
                    if header_byte:
                        header = header_byte.decode('utf-8', errors='ignore')
                except BlockingIOError:
                    break # 没数据，跳过
                except Exception as e:
                    print(f"连接异常: {e}")
                    break
                
                if header in ['d', 'r', 'b', 'm', 'p', 'v', 'f', 'e', 'h', 'O', 'P']:
                    #print(f"\n[TCP] Received Header: '{header}'")
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

                    # ===============================================
                    # CASE 'e': 接收眼动追踪坐标 (Eye Tracking)
                    # ===============================================
                    elif header == 'e':
                        #print("receiving gazing position")
                        eye_data_bytes = recv_exact(conn, 24)
                        if not eye_data_bytes: break
                        
                        eye_data = struct.unpack('<6f', eye_data_bytes)
                        u_origin_pos = np.array(eye_data[0:3])
                        u_hit_pos = np.array(eye_data[3:6])
                        
                        if T_M is not None:
                            # 使用 T_M 将 HoloLens 坐标转换到相机/全局坐标系
                            r_origin_pos = rut.unity2robot_transform(u_origin_pos, T_M)
                            r_hit_pos = rut.unity2robot_transform(u_hit_pos, T_M)
                            
                            global_ray_origin = r_origin_pos
                            global_ray_hit = r_hit_pos
                            
                            # ==========================================
                            # Debug: Print Origin and End Point (Every 1 second)
                            # ==========================================
                            current_time = time.time() # 获取当前时间
                            
                            # ==========================================
                            # 🛠️ 纯英文打印起点和终点 (每 1 秒打印一次)
                            # ==========================================
                            if current_time - last_ray_print_time > 1.0:
                                print(f"[TCP] Gaze Origin: {np.round(r_origin_pos, 3)} | Gaze End: {np.round(r_hit_pos, 3)}")
                            last_ray_print_time = current_time
                            # ==========================================
                                
                            #print(f"👁️ [TCP] 眼动坐标已更新...")
                        else:
                            print("   ⚠️ T_M 矩阵为空，无法转换眼动坐标！")

                    # ===============================================
                    # 🌟 CASE 'h': 接收原生手掌跟踪坐标 (Header 'h')
                    # ===============================================
                    elif header == 'h':
                        pos_bytes = recv_exact(conn, 12)
                        if pos_bytes:
                            hx, hy, hz = struct.unpack('<fff', pos_bytes)
                            u_hand_pos = np.array([hx, hy, hz])
                            
                            if T_M is not None:
                                # 转换为机器人坐标并存入全局变量
                                global_holo_hand_pos = rut.unity2robot_transform(u_hand_pos, T_M)
                            else:
                                print("  ⚠️ T_M 矩阵为空，无法转换手掌坐标！")

                    elif header == 'O': 
                        is_HRI_Demo = True
                        print("Demo模式已开")
                    elif header == 'P': 
                        is_HRI_Demo = False
                        print("Demo模式已关")
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
                conn.setblocking(False) 
                #print(f"[TCP] {header} 处理完毕，切回视频模式")

           



            # --- 只有在开关打开时才执行发送函数 ---
            # --- 重点：视频转发逻辑 ---
            if sender.is_streaming:
                # 这个函数现在会自动找 A 电脑要图并转给 conn (HoloLens)
                sender.send_frame(conn, sensor_type='c')

            if is_skeleton_streaming and Body3DSkeletonProcess_dual:
                if time.time() - last_yolo_time > 0.1:
                    last_yolo_time = time.time() # 记录运行时间
                        
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

                    # print("\n" + "="*50)
                    # print("🤖 [HRI] 测试模式：跳过跟踪，直接进入状态 7 (Looking for user)！")
                    
                    # current_hri_state = STATE_TRACKING_AND_PASS  # 🌟 直接空降状态 7！
                    # hri_start_time = 0.0                    # 确保重置计时器
                    
                    # print("="*50 + "\n")
                
                # 0. 初始位姿状态：机械臂前往预设的初始位置，等待 5 秒让用户观察
                elif current_hri_state == STATE_INIT:
                    # 刚进入 STATE_INIT 时，触发一次运动并记录时间
                    if hri_start_time == 0.0:
                        print("🔄 [HRI] 状态 0: 机械臂前往初始观测位姿...")
                        if robot is not None:
                            robot.move_to(INIT_POSE, speed=0.05)
                        hri_start_time = time.time() # 🌟 开始计时！
                        
                    # 每次循环检查时间是否够了，如果没够就直接跳过，继续发视频
                    else:
                        if time.time() - hri_start_time > 20.0:
                            print(" ✅ [HRI] 5秒已过，假设已到达初始位置，切入人体跟踪状态！")
                            if robot is not None:
                                robot.start_tracking()
                            current_hri_state = STATE_TRACKING
                            hri_start_time = 0.0 # 重置计时器，给以后的状态用
                    
                # -----------------------------------
                # 状态 1：视觉伺服跟踪人体
                # -----------------------------------
                elif current_hri_state == STATE_TRACKING:
                    
                    # YOLO 至少需要 11 个点才能拿到左右手腕 (Index 9, 10)
                    if skeleton_coord_camera is not None and len(skeleton_coord_camera) > 10:
                        
            # 🌟【全新防抖逻辑】：不再一帧定生死，而是塞入时间窗口
                        current_intent = check_reach_intent(skeleton_coord_camera)
                        intent_history.append(current_intent)
                        
                        # 只有当队列填满了（时间窗口满了），才开始计算比例
                        if len(intent_history) == intent_history.maxlen:
                            # 统计窗口内 True 的个数
                            true_count = sum(intent_history)
                            true_ratio = true_count / len(intent_history)
                            
                            # 如果超过 80% 的帧都判定为伸手
                            if true_ratio >= 0.8:
                                print(f" 🎯 [HRI] 连续确认伸手意图 (置信度: {true_ratio*100:.1f}%)！停止跟踪，准备看向桌面...")
                                
                                if robot is not None:
                                    robot.stop_tracking() 
                                
                                current_hri_state = STATE_CHECK_INTENT 
                                hri_start_time = 0.0 
                                
                                # ⚠️ 非常重要：切入下一个状态前，清空历史队列！
                                # 防止未来如果切回状态 1 时，旧的残留数据导致瞬间误触发
                                intent_history.clear() 
                                
                                continue # 跳过本次循环的 PID 跟踪

                        # --- 以下是正常的 YOLO 物理 3D 坐标 PID 跟踪 ---
                        l_shoulder = np.array(skeleton_coord_camera[5])
                        r_shoulder = np.array(skeleton_coord_camera[6])
                        
                        # 过滤无效点 (0,0,0)
                        if np.linalg.norm(l_shoulder) > 0.1 and np.linalg.norm(r_shoulder) > 0.1:
                            # 计算胸口中心点真实的 3D 坐标 (单位：米)
                            chest_3d = (l_shoulder + r_shoulder) / 2.0
                            
                            cam_x = chest_3d[0] # 单位：米
                            cam_y = chest_3d[1] # 单位：米

                            # 🐞 【核心 DEBUG 1】：看看相机到底输出了什么级别的坐标
                            print(f"👀 [视觉测距] 胸口物理坐标: X={cam_x:.4f}m, Y={cam_y:.4f}m, 深度Z={chest_3d[2]:.4f}m")
                            
                            # 轴向映射
                            base_err_x = 0.0       
                            base_err_y = -cam_x    
                            base_err_z = 0.0 #-cam_y    
                            
                            # 死区改成物理尺寸 (0.03代表3厘米)
                            if abs(base_err_y) < 0.03: base_err_y = 0.0
                            if abs(base_err_z) < 0.03: base_err_z = 0.0
                            
                            if robot is not None:
                                robot.update_tracking_error(base_err_x, base_err_y, base_err_z)
                        else:
                            if robot is not None: robot.update_tracking_error(0.0, 0.0, 0.0)
                    else:
                        if robot is not None: robot.update_tracking_error(0.0, 0.0, 0.0)
                        intent_history.append(False)
                


                # -----------------------------------
                # 状态 2：机器人看向手伸出的方向（桌面）
                # -----------------------------------
                elif current_hri_state == STATE_CHECK_INTENT:
                    
                    if hri_start_time == 0.0:
                        print("👀 [HRI] 状态 2: 机械臂低头，看向桌面目标区域...")
                        if robot is not None:
                            # 前往我们预设好的桌面观测点
                            # 请确保在 main() 开头定义了 LOOK_AT_TABLE_POSE = [x, y, z, qx, qy, qz, qw]
                            #pass
                            robot.move_to(LOOK_AT_TABLE_POSE, speed=0.05)
                            
                        hri_start_time = time.time()
                        
                    else:
                        # 非阻塞等待机械臂走到位 (这里给 4 秒时间，可根据实际距离调整)
                        if time.time() - hri_start_time > 10.0:
                            print(" ✅ [HRI] 视线已锁定桌面！准备进入物品识别 (State 3)...")
                            current_hri_state = STATE_SCAN_OBJECTS
                            hri_start_time = 0.0

                # -----------------------------------
                # 状态 3：全自动直线巡航建图 (一次扫描，多次复用版)
                # -----------------------------------
                elif current_hri_state == STATE_SCAN_OBJECTS:
                    
                    if hri_start_time == 0.0:
                        if not has_completed_initial_scan:
                            print(f"\n🚀 [HRI] 状态 3: 启动首次全自动物理扫描。计划点数: {SCAN_STEPS}")
                            scan_current_step = 0 
                            if robot is not None:
                                robot.move_to(scan_waypoints[scan_current_step], speed=0.05)
                        else:
                            print(f"\n🚀 [HRI] 状态 3: 跳过物理扫描，直接复用并刷新已有环境记忆...")
                            # 🌟 核心魔法：直接设为满值，跳过底下的所有拍照和移动！
                            scan_current_step = SCAN_STEPS 
                            
                        hri_start_time = time.time()
                        last_capture_time = 0.0 
                        
                    else:
                        scene_mapper.update_window()
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True
                        ee_pos, ee_quat = robot_listener.get_current_pose()
                        
                        if ee_pos is not None:
                            # =========================================================
                            # 【未走完步数时】：执行物理移动与采集
                            # =========================================================
                            if scan_current_step < SCAN_STEPS:
                                target_pos = scan_waypoints[scan_current_step][0:3]
                                dist = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
                                
                                if last_capture_time == 0.0:
                                    if dist < 0.02 and is_robot_idle: 
                                        print(f"📍 已到达节点 {scan_current_step + 1}！停顿 0.8 秒防抖...")
                                        last_capture_time = time.time()
                                else:
                                    if time.time() - last_capture_time > 0.5: 
                                        print(f"📸 正在获取第 {scan_current_step + 1}/{SCAN_STEPS} 个视角的点云...")
                                        
                                        current_point_cloud = BodyPointCloud_dual.global_latest_verts
                                        current_colors = BodyPointCloud_dual.global_latest_colors 
                                        
                                        if current_point_cloud is not None:
                                            bbox_mask = (
                                                (current_point_cloud[:, 2] > 0.1) & (current_point_cloud[:, 2] < 1.5) & 
                                                (current_point_cloud[:, 0] > -0.8) & (current_point_cloud[:, 0] < 0.8)
                                            )
                                            p_crop = current_point_cloud[bbox_mask]
                                            c_crop = current_colors[bbox_mask]

                                            verts_robot_base = camera2unity.point_cloud_camera_to_robot(
                                                p_crop, ee_pos, ee_quat, EE_T_C
                                            )
                                            scene_mapper.add_point_cloud(verts_robot_base, c_crop)
                                            print(f"   ✅ 节点 {scan_current_step + 1} 拼合成功。")
                                        else:
                                            print(f"   ⚠️ 节点 {scan_current_step + 1} 数据获取失败，跳过。")
                                            
                                        scan_current_step += 1
                                        if scan_current_step < SCAN_STEPS:
                                            if robot is not None:
                                                robot.move_to(scan_waypoints[scan_current_step], speed=0.05)
                                            last_capture_time = 0.0 
                            
                            # =========================================================
                            # 【已走完或直接跳过时】：进入安全后处理与退回阶段
                            # =========================================================
                            else:
                                if getattr(scene_mapper, 'scan_post_process', 0) == 0:
                                    print("\n🎉 [HRI] 开始处理/刷新点云地图数据...")
                                    
                                    # 🌟 标记：第一次完整的物理扫描已经完成了！以后都不用再扫了
                                    has_completed_initial_scan = True 
                                    
                                    final_pcd = scene_mapper.global_pcd
                                    
                                    # 防崩保护：如果桌上的东西全抓完了，点云没了，直接跳回待机
                                    if final_pcd is None or len(final_pcd.points) < 50:
                                        print("⚠️ 桌面已空，没有足够的点云数据！自动切回待机状态...")
                                        current_hri_state = STATE_IDLE
                                        hri_start_time = 0.0
                                        scene_mapper.scan_post_process = 0
                                        continue

                                    print("   ✂️ 0. 正在进行 Z 轴高度直通滤波...")
                                    points = np.asarray(final_pcd.points)
                                    valid_z_indices = np.where(points[:, 2] >= -0.25)[0] 
                                    final_pcd = final_pcd.select_by_index(valid_z_indices)

                                    print("   🧹 1. 正在执行统计学滤波降噪...")
                                    cleaned_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                                    
                                    scene_mapper.global_pcd = cleaned_pcd
                                    scene_mapper.display_pcd.points = cleaned_pcd.points
                                    scene_mapper.display_pcd.colors = cleaned_pcd.colors
                                    scene_mapper.update_window()

                                    try:
                                        # 保存为 .pcd 格式 (你可以换成 .ply 格式)
                                        o3d.io.write_point_cloud("scanned_scene.pcd", cleaned_pcd)
                                        print("   💾 [系统] 已将当前桌面点云成功保存为 scanned_scene.pcd")
                                    except Exception as e:
                                        print(f"   ❌ [系统] 保存点云失败: {e}")
                                    
                                    if request_grasps_from_graspnet is not None:
                                        print("\n   📦 准备打包真实点云与相机内参 K_1...")
                                        
                                        # 1. 获取当前拍照位置的相机到机器人基座的变换矩阵
                                        ee_pos, ee_quat = robot_listener.get_current_pose()
                                        ROBOT_T_C = camera2unity.get_camera_to_robot_matrix(ee_pos, ee_quat, EE_T_C)

                                        # 2. 计算逆矩阵（从机器人基座转回相机坐标系）
                                        C_T_ROBOT = np.linalg.inv(ROBOT_T_C)

                                        # 3. 提取去噪后的干净点云，并将其【逆变换】回相机坐标系
                                        real_points_base = np.asarray(cleaned_pcd.points)
                                        ones = np.ones((real_points_base.shape[0], 1))
                                        points_hom = np.hstack((real_points_base, ones))
                                        # 关键：转回相机系，网络就能看懂了！
                                        real_points_camera = (C_T_ROBOT @ points_hom.T).T[:, :3] 

                                        real_colors = np.asarray(cleaned_pcd.colors) if cleaned_pcd.has_colors() else np.zeros_like(real_points_base)

                                        # 4. 把【相机系下的点云】发给后台线程
                                        def ai_task(points, colors, K, ip):
                                            return request_grasps_from_graspnet(points, colors, K, server_ip=ip)

                                        # 注意：这里传入保存当前位置的 ROBOT_T_C，用于后面把结果转回来
                                        future = executor.submit(ai_task, real_points_camera, real_colors, K_1, "100.116.99.44")
                                        scene_mapper.current_robot_t_c = ROBOT_T_C # 暂存这个矩阵
                                        scene_mapper.ai_future = future
                                        scene_mapper.scan_post_process = 2
                                    
                                    print("   🪚 2. 正在识别桌面 (RANSAC)...")
                                    plane_model, inliers = cleaned_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                                    objects_pcd = cleaned_pcd.select_by_index(inliers, invert=True)

                                    table_points = np.asarray(cleaned_pcd.select_by_index(inliers).points)
                                    scene_mapper.table_z = np.mean(table_points[:, 2]) 
                                    print(f"   📏 测得桌面绝对物理高度: Z = {scene_mapper.table_z:.4f} 米")
                                    
                                    print("   📦 3. 正在聚类剩余物体...")
                                    labels = np.array(objects_pcd.cluster_dbscan(eps=0.03, min_points=30))
                                    
                                    valid_mask = labels >= 0  
                                    clean_objects_pcd = objects_pcd.select_by_index(np.where(valid_mask)[0])
                                    clean_labels = labels[valid_mask]
                                    
                                    import matplotlib.pyplot as plt 
                                    max_label = clean_labels.max() if len(clean_labels) > 0 else -1
                                    if max_label >= 0:
                                        cmap = plt.get_cmap("tab20")
                                        colors = cmap(clean_labels / (max_label if max_label > 0 else 1))[:, :3]
                                        clean_objects_pcd.colors = o3d.utility.Vector3dVector(colors)
                                        
                                        scene_mapper.display_pcd.points = clean_objects_pcd.points
                                        scene_mapper.display_pcd.colors = clean_objects_pcd.colors
                                        scene_mapper.update_window()

                                    scene_mapper.objects_pcd = clean_objects_pcd
                                    scene_mapper.object_labels = clean_labels
                                    print(f"   ✅ 环境刷新完毕！桌面剩余 {clean_labels.max() + 1 if len(clean_labels)>0 else 0} 个目标。")
                                    
                                    print(f"   🤖 正在前往安全观测/递送准备位...")
                                    if robot is not None:
                                        robot.move_to(READY_FOR_PASSING_POSE, speed=0.05)
                                        
                                    #scene_mapper.scan_post_process = 1  
                                    hri_start_time = time.time()
                                    

                                
                                elif getattr(scene_mapper, 'scan_post_process', 0) == 2:
                                    if scene_mapper.ai_future.done():
                                        # 🌟 AI 在相机系下预测出来的原始结果
                                        ai_grasps_camera = scene_mapper.ai_future.result() 
                                        
                                        if ai_grasps_camera is not None:
                                            print(f"🧠 AI 在相机系下预测了 {ai_grasps_camera.shape[0]} 个姿态，正在正变换至机器人基座...")
                                            
                                            # 拿出刚才状态 3 存下来的那一瞬间的 ROBOT_T_C
                                            ROBOT_T_C = scene_mapper.current_robot_t_c
                                            ai_grasps_base = np.zeros_like(ai_grasps_camera)
                                            
                                            # 🌟 矩阵左乘变换：将相机系抓取转为机器人基座系
                                            for i in range(ai_grasps_camera.shape[0]):
                                                ai_grasps_base[i] = ROBOT_T_C @ ai_grasps_camera[i]
                                                
                                            scene_mapper.ai_grasps = ai_grasps_base
                                            
                                            # 自动保存到本地文件
                                            np.save("test_output_grasps.npy", ai_grasps_base)
                                            print("🎉 [AI] 异步推理并坐标对齐成功完成！文件已保存。")
                                            
                                        scene_mapper.scan_post_process = 1
                                        
                                elif getattr(scene_mapper, 'scan_post_process', 0) == 1:
                                    if is_robot_idle and (time.time() - hri_start_time > 1.0):
                                        print("\n👀 切换至状态 4: 请通过眼动凝视选择目标物体...")
                                        current_hri_state = STATE_GAZE_INTERSECTION
                                        hri_start_time = 0.0
                                        scene_mapper.scan_post_process = 0 # 重置标记

                # -----------------------------------
                # 状态 4：全景眼动求交 + 纯净物品吸附
                # -----------------------------------
                elif current_hri_state == STATE_GAZE_INTERSECTION:
                    
                    if hri_start_time == 0.0:
                        print("👀 [HRI] 状态 4: 请凝视你要抓取的物体 (持续 2 秒)...")
                        hri_start_time = time.time()
                        debug_print_time = time.time() 
                        
                        fixation_point = None       
                        fixation_start_time = 0.0   
                        FIXATION_TOLERANCE = 0.05   
                        FIXATION_TIME_REQUIRED = 2.0 
                        
                    else:
                        scene_mapper.update_window() 
                        current_pt = None
                        
                        # 🎯 1. 射线跟【完整大地图（含桌子）】求交！保证红球绝不消失！
                        verts_robot_base = scene_mapper.get_merged_points_numpy()
                        if global_ray_origin is not None and global_ray_hit is not None and verts_robot_base is not None:
                            current_pt = get_gaze_point_cloud_intersection(
                                ray_origin=global_ray_origin, 
                                ray_hit_pos=global_ray_hit, 
                                point_cloud=verts_robot_base, 
                                radius=0.02 
                            )
                            
                        scene_mapper.update_marker(current_pt)
                        
                        # ==========================================
                        if current_pt is not None and T_M is not None:
                            try:
                                u_pt = rut.robot2unity_transform(current_pt, T_M)
                                
                                packet_p = b't' + struct.pack('<fff', u_pt[0], u_pt[1], u_pt[2])
                                
                                # ⚠️ TCP 关键：发送前必须开启阻塞，发完再关掉！
                                conn.setblocking(True)
                                conn.sendall(packet_p)
                                conn.setblocking(False)
                            except Exception as e:
                                print(f"⚠️ [TCP] 发送实时光标失败: {e}")
                        # ==========================================

                        if time.time() - debug_print_time > 0.5:
                            if current_pt is not None:
                                print(f"🎯 [Debug] 视线落点 : {np.round(current_pt, 3)}")
                            debug_print_time = time.time()

                        # 🎯 2. 凝视确认与“智能吸附”
                        if current_pt is not None:
                            if fixation_point is None:
                                fixation_point = current_pt
                                fixation_start_time = time.time()
                            else:
                                dist = np.linalg.norm(current_pt - fixation_point)
                                if dist < FIXATION_TOLERANCE:
                                    dwell_time = time.time() - fixation_start_time
                                    
                                    # 盯住 2 秒了！开始吸附盒子！
                                    if dwell_time >= FIXATION_TIME_REQUIRED:
                                        
                                        if hasattr(scene_mapper, 'objects_pcd') and not scene_mapper.objects_pcd.is_empty():
                                            # 🧲 去【纯净盒子地图】里找离当前落点最近的点
                                            kdtree = o3d.geometry.KDTreeFlann(scene_mapper.objects_pcd)
                                            _, idx, sq_dist = kdtree.search_knn_vector_3d(fixation_point, 1)
                                            
                                            distance_to_box = np.sqrt(sq_dist[0])
                                            
                                            # 如果你看的地方方圆 20 厘米内根本没有盒子，说明你在看空桌子，放弃抓取
                                            if distance_to_box > 0.3:
                                                print(f"\n ⚠️ 视线落点周围没有盒子，请看向盒子！ distance to box : {np.round(distance_to_box, 3)}")
                                                fixation_point = None
                                                fixation_start_time = time.time()
                                                continue
                                                
                                            # 成功吸附！获取盒子 ID（因为噪点被删了，这里必定是有效盒子）
                                            box_id = scene_mapper.object_labels[idx[0]]
                                            
                                            # 📦 提取这整个盒子的所有点，计算几何中心
                                            box_indices = np.where(scene_mapper.object_labels == box_id)[0]
                                            box_points = np.asarray(scene_mapper.objects_pcd.points)[box_indices]

                                           # 1. 将 numpy 数组转为 Open3D 点云对象
                                            single_box_pcd = o3d.geometry.PointCloud()
                                            single_box_pcd.points = o3d.utility.Vector3dVector(box_points)

                                            # 2. 计算有向包围盒 (OBB)
                                            obb = single_box_pcd.get_oriented_bounding_box()

                                            # =======================================================
                                            # 🌟【修改 2】残影消除：把这个盒子从全局记忆地图中抹除！
                                            # =======================================================
                                            # 稍微放大一下 OBB (比如 1.2 倍)，确保边缘和底部的噪点也能被包裹进去
                                            eraser_obb = o3d.geometry.OrientedBoundingBox(obb)
                                            eraser_obb.scale(1.2, eraser_obb.center)
                                            
                                            # 获取全局地图中，落在这个“橡皮擦盒子”里的所有点的索引
                                            ghost_indices = eraser_obb.get_point_indices_within_bounding_box(scene_mapper.global_pcd.points)
                                            
                                            # 使用 invert=True 进行反向提取，相当于把这些点“抠除”
                                            scene_mapper.global_pcd = scene_mapper.global_pcd.select_by_index(ghost_indices, invert=True)
                                            
                                            # 同步把这个物体从“磁铁”点云中剔除，防止接下来发生重复吸附
                                            ghost_indices_obj = eraser_obb.get_point_indices_within_bounding_box(scene_mapper.objects_pcd.points)
                                            scene_mapper.objects_pcd = scene_mapper.objects_pcd.select_by_index(ghost_indices_obj, invert=True)
                                            
                                            print(f"🧹 已将 [盒子 {box_id}] 的残影从全局点云地图中永久擦除！")
                                            # =======================================================

                                            # 3. 获取真正的几何中心 (这比 np.mean 准得多！)
                                            box_center = obb.center

                                            # 4. 获取盒子的长宽高尺寸 (用于决定夹爪张开多大)
                                            box_size = obb.extent 
                                            print(f"📦 盒子尺寸: 长宽高 {np.round(box_size, 3)} 米")

                                            # 5. 获取盒子的旋转矩阵 (用于对齐夹爪姿态！)
                                            box_rotation = obb.R
                                            
                                            print(f"\n 🎉 成功吸附！目标锁定为 [盒子 {box_id}]")
                                            print(f" 🎯 盒子几何中心坐标: {np.round(box_center, 3)}")
                                            
                                            # =======================================================
                                            # 智能匹配：在 AI 预测的多个抓取位姿中，选出一个距离盒子中心最近的唯一最优姿态！
                                            # =======================================================
                                            if hasattr(scene_mapper, 'ai_grasps') and scene_mapper.ai_grasps is not None and len(scene_mapper.ai_grasps) > 0:
                                                min_grasp_dist = float('inf')
                                                closest_grasp = None
                                                closest_idx = -1
                                                
                                                # 遍历 4060 传回来并转好基座系的所有 4x4 抓取矩阵
                                                for idx_g, grasp_matrix in enumerate(scene_mapper.ai_grasps):
                                                    # 提取矩阵中前 3 行第 4 列的 [X, Y, Z] 空间平移坐标
                                                    grasp_xyz = grasp_matrix[:3, 3]
                                                    
                                                    # 计算该抓取点到当前凝聚盒子中心的欧氏距离
                                                    dist = np.linalg.norm(grasp_xyz - box_center)
                                                    
                                                    if dist < min_grasp_dist:
                                                        min_grasp_dist = dist
                                                        closest_grasp = grasp_matrix
                                                        closest_idx = idx_g
                                                
                                                # 将筛选出的唯一最优姿态保存至 scene_mapper 供下一步使用
                                                scene_mapper.selected_grasp = closest_grasp
                                                print(f"🎯 [智能匹配] 成功锁定最佳抓取位姿！")
                                                print(f"   👉 抓取索引: {closest_idx} / {len(scene_mapper.ai_grasps)}")
                                                print(f"   👉 距离物品中心: {np.round(min_grasp_dist, 3)} 米")
                                            else:
                                                print("⚠️ [警告] 内存中未检测到 AI 的抓取姿态列表 (scene_mapper.ai_grasps 为空)！")
                                                scene_mapper.selected_grasp = None
                                            # =======================================================
                                            # if T_M is not None:
                                            #     try:
                                            #         print("box position sent to unity")
                                            #         u_target = rut.robot2unity_transform(box_center, T_M)
                                            #         packet = b't' + struct.pack('<fff', u_target[0], u_target[1], u_target[2])
                                            #         conn.sendall(packet)
                                            #         time.sleep(0.01)
                                            #     except Exception:
                                            #         pass
                                            
                                            scene_mapper.target_box_points = box_points
                                            
                                            current_hri_state = STATE_GRAB_OBJECT 
                                            hri_start_time = 0.0
                                            
                                        else:
                                            print("⚠️ 场景中没有找到任何盒子！")
                                            fixation_point = None
                                            fixation_start_time = time.time()
                                else:
                                    fixation_point = current_pt
                                    fixation_start_time = time.time()
                        else:
                            fixation_point = None
                            fixation_start_time = 0.0

                # # -----------------------------------
                # # 状态 5：高精度纯几何抓取 (Kinematic Grasp)
                # # -----------------------------------
                # elif current_hri_state == STATE_GRAB_OBJECT:
                    
                #     if hri_start_time == 0.0:
                #         print("\n" + "="*50)
                #         print("🦾 [HRI] 状态 5: 开始执行纯视觉几何抓取...")
                        
                #         # 1. 获取目标盒子点云，计算 OBB 包围盒
                #         box_pcd = o3d.geometry.PointCloud()
                #         box_pcd.points = o3d.utility.Vector3dVector(scene_mapper.target_box_points)
                #         obb = box_pcd.get_oriented_bounding_box()
                        
                #         # 2. 🌟 降维打击计算坐标：X/Y 相信 OBB，Z 相信桌面！
                #         center_x, center_y, _ = obb.center
                        
                #         # 盒子的最高点 Z (结合桌面高度，算出真实的物理最高点)
                #         box_top_z = np.max(scene_mapper.target_box_points[:, 2])
                #         box_actual_height = box_top_z - getattr(scene_mapper, 'table_z', 0.0)
                        
                #         # 设定抓取点 (比如从盒子最高点往下探 2 厘米)
                #         GRASP_DEPTH = -0.02
                #         target_z = box_top_z - GRASP_DEPTH
                        
                #         print(f"   📦 目标中心: X={center_x:.3f}, Y={center_y:.3f}")
                #         print(f"   📏 目标高度: {box_actual_height*100:.1f} cm (抓取深度 Z={target_z:.3f})")
                        
                #         # 3. 定义夹爪朝下的固定四元数 (请替换为你实际夹爪垂直朝下的位姿！)
                #         GRASP_ROT = [-0.92, 0.38, -0.01, -0.00] 
                        
                #         # 4. 生成三段式关键点，并安全地挂载到 scene_mapper 上！
                #         scene_mapper.hover_pose = [center_x, center_y, target_z + 0.15] + GRASP_ROT # 悬停点
                #         scene_mapper.grasp_pose = [center_x, center_y, target_z ] + GRASP_ROT        # 抓取点
                #         scene_mapper.lift_pose  = [center_x, center_y, target_z + 0.20] + GRASP_ROT # 提拉点
                        
                #         # 5. 立刻派发第一阶段：飞往悬停点
                #         print("   -> 🛫 阶段 1：飞往正上方悬停...")
                #         if robot is not None:
                #             robot.move_to(scene_mapper.hover_pose, speed=0.03) 
                            
                #         scene_mapper.grasp_step = 1       # 控制抓取阶段的变量
                #         hri_start_time = time.time()
                        
                #     else:
                #         # 🌟 新增：获取机械臂是否处于空闲状态 (动作是否执行完毕)
                #         is_robot_idle = robot.path_queue.empty() if robot is not None else True

                #         # -----------------------------------
                #         # 阶段 2：飞到上方后，张开夹爪，然后缓慢下探
                #         # -----------------------------------
                #         # 修改条件：等待至少 1 秒（防止队列还没塞进去就误判）并且 机械臂动作已走完！
                #         if getattr(scene_mapper, 'grasp_step', 0) == 1 and time.time() - hri_start_time > 1.0 and is_robot_idle:
                            
                #             # 🌟 先张开夹爪！
                #             if robot is not None:
                #                 robot.open_gripper(width=0.08) # 张开 8 厘米
                                
                #             print("   -> 🛬 阶段 2：夹爪已张开，开始直线缓慢下压...")
                #             if robot is not None:
                #                 robot.move_to(scene_mapper.grasp_pose, speed=0.02) 
                                
                #             scene_mapper.grasp_step = 2
                #             hri_start_time = time.time()
                            
                #         # -----------------------------------
                #         # 阶段 3：到达底部，闭合夹爪
                #         # -----------------------------------
                #         # 修改条件：下探的动作必须完全走完，才允许闭合夹爪！
                #         elif getattr(scene_mapper, 'grasp_step', 0) == 2 and time.time() - hri_start_time > 1.5 and is_robot_idle:
                #             print("   -> ✊ 阶段 3：接触目标，正在闭合夹爪...")
                            
                #             # 🌟 抓紧盒子！
                #             if robot is not None:
                #                 robot.close_gripper(force=15.0, speed=0.05) # 施加 30N 的抓取力
                            
                #             scene_mapper.grasp_step = 3
                #             hri_start_time = time.time()
                            
                #         # -----------------------------------
                #         # 阶段 4：抓稳后，向上提拉
                #         # -----------------------------------
                #         # 这个阶段不用等 is_robot_idle，因为夹爪闭合(close_gripper)本身有 sleep(1.5) 阻塞，
                #         # 当代码走到这里时，夹爪肯定已经闭合完毕了。
                #         elif getattr(scene_mapper, 'grasp_step', 0) == 3 and time.time() - hri_start_time > 0.5: 
                #             print("   -> 🚀 阶段 4：提拉物品...")
                #             if robot is not None:
                #                 robot.move_to(scene_mapper.lift_pose, speed=0.05) # 提拉稍微快一点点
                                
                #             scene_mapper.grasp_step = 4
                #             hri_start_time = time.time()
                            
                #         # -----------------------------------
                #         # 任务完成，重置状态
                #         # -----------------------------------
                #         elif getattr(scene_mapper, 'grasp_step', 0) == 4 and time.time() - hri_start_time > 1.0 and is_robot_idle:
                #             print("\n🎉 [HRI] Grab success! Passing the object to user")
                #             current_hri_state = STATE_LOOKING_FOR_USER
                #             hri_start_time = 0.0
                
                # -----------------------------------
                # 状态 5：高精度 6-DoF AI 神经网络抓取 (Selected AI Grasp)
                # -----------------------------------
                elif current_hri_state == STATE_GRAB_OBJECT:
                    
                    if hri_start_time == 0.0:
                        print("\n" + "="*50)
                        print("🦾 [HRI] 状态 5: 开始执行眼动过滤后的全新 AI 6-DoF 抓取...")
                        
                        # 🌟 1. 核心提取：检查眼动阶段是否成功截获了唯一的最佳抓取矩阵
                        if hasattr(scene_mapper, 'selected_grasp') and scene_mapper.selected_grasp is not None:
                            grasp_matrix = scene_mapper.selected_grasp
                            
                            # 从 4x4 矩阵中直接提取 3D 精准目标位置 [X, Y, Z]
                            center_x, center_y, target_z = grasp_matrix[:3, 3]
                            
                            # 使用 scipy 将 3x3 旋转矩阵转换为机械臂所需要的四元数格式 [qx, qy, qz, qw]
                            from scipy.spatial.transform import Rotation as Rot
                            grasp_rot_mat = grasp_matrix[:3, :3]
                            grasp_quat = Rot.from_matrix(grasp_rot_mat).as_quat().tolist()
                            
                            print(f"   🧠 [AI 姿态激活] 坐标与角度已完全对齐神经网络输出！")
                            print(f"      📍 目标位置: X={center_x:.3f}, Y={center_y:.3f}, Z={target_z:.3f}")
                            print(f"      🔄 真实姿态四元数: {np.round(grasp_quat, 3)}")
                        
                        else:
                            # 🚨 安全降级备用：如果发生了意外丢失，自动回退到你的经典 OBB 方案
                            print("⚠️ [警告] 未能找到眼动筛选的 selected_grasp，自动降级为传统 OBB 启发式抓取...")
                            box_pcd = o3d.geometry.PointCloud()
                            box_pcd.points = o3d.utility.Vector3dVector(scene_mapper.target_box_points)
                            obb = box_pcd.get_oriented_bounding_box()
                            center_x, center_y, _ = obb.center
                            box_top_z = np.max(scene_mapper.target_box_points[:, 2])
                            target_z = box_top_z - (-0.02) 
                            grasp_quat = [-0.92, 0.38, -0.01, -0.00] # 固定的垂直朝下姿态
                        
                        # 🌟 2. 生成全新三段式航点
                        # 姿态部分统一采用 AI 预测的真实旋转姿态(grasp_quat)
                        # 悬停点与提拉点继续保持世界坐标系 Z 轴垂直升降，这是最安全的防碰撞策略！
                        scene_mapper.hover_pose = [center_x, center_y, target_z + 0.15] + grasp_quat # 悬停点
                        scene_mapper.grasp_pose = [center_x, center_y, target_z]        + grasp_quat # 真实抓取点
                        scene_mapper.lift_pose  = [center_x, center_y, target_z + 0.20] + grasp_quat # 提拉点
                        
                        # 3. 立刻派发第一阶段：飞往悬停点
                        print("   -> 🛫 阶段 1：飞往目标正上方悬停...")
                        if robot is not None:
                            robot.move_to(scene_mapper.hover_pose, speed=0.03) 
                            
                        scene_mapper.grasp_step = 1       # 控制抓取阶段的变量
                        hri_start_time = time.time()
                        
                    else:
                        # 获取机械臂动作是否执行完毕（保持你原有的安全队列判定）
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True

                        # -----------------------------------
                        # 阶段 2：飞到上方后，张开夹爪，然后缓慢下探
                        # -----------------------------------
                        if getattr(scene_mapper, 'grasp_step', 0) == 1 and time.time() - hri_start_time > 1.0 and is_robot_idle:
                            if robot is not None:
                                robot.open_gripper(width=0.08) # 张开 8 厘米
                                
                            print("   -> 🛬 阶段 2：夹爪已张开，开始直线缓慢下压...")
                            if robot is not None:
                                robot.move_to(scene_mapper.grasp_pose, speed=0.02) 
                                
                            scene_mapper.grasp_step = 2
                            hri_start_time = time.time()
                            
                        # -----------------------------------
                        # 阶段 3：到达底部，闭合夹爪
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 2 and time.time() - hri_start_time > 1.5 and is_robot_idle:
                            print("   -> ✊ 阶段 3：接触目标，正在闭合夹爪...")
                            if robot is not None:
                                robot.close_gripper(force=15.0, speed=0.05) 
                                
                            scene_mapper.grasp_step = 3
                            hri_start_time = time.time()
                            
                        # -----------------------------------
                        # 阶段 4：抓稳后，向上提拉
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 3 and time.time() - hri_start_time > 0.5: 
                            print("   -> 🚀 阶段 4：提拉物品...")
                            if robot is not None:
                                robot.move_to(scene_mapper.lift_pose, speed=0.05) 
                                
                            scene_mapper.grasp_step = 4
                            hri_start_time = time.time()
                            
                        # -----------------------------------
                        # 任务完成，重置状态
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 4 and time.time() - hri_start_time > 1.0 and is_robot_idle:
                            print("\n🎉 [HRI] Grab success! Passing the object to user")
                            current_hri_state = STATE_LOOKING_FOR_USER
                            hri_start_time = 0.0
                # # -----------------------------------
                # # 状态 6：寻找用户 -> 确认伸手 -> 翻转为递送姿势
                # # -----------------------------------
                # elif current_hri_state == STATE_LOOKING_FOR_USER:
                    
                #     if hri_start_time == 0.0:
                #         print("\n" + "="*50)
                #         print("🤖 [HRI] 状态 6: 抓取成功！正在抬头寻找用户...")
                        
                #         if robot is not None:
                #             robot.move_to(LOOK_USER_POSE, speed=0.05)
                        
                #         hri_start_time = time.time()
                #         intent_history.clear() 
                #         # 🌟 新增：阶段标记，1 为等待伸手，2 为翻转姿态
                #         scene_mapper.looking_step = 1 

                #     else:
                #         is_robot_idle = robot.path_queue.empty() if robot is not None else True
                        
                #         # -----------------------------------
                #         # 阶段 1：等待机械臂抬头完毕，并用相机检测伸手意图
                #         # -----------------------------------
                #         if getattr(scene_mapper, 'looking_step', 1) == 1:
                #             if is_robot_idle and (time.time() - hri_start_time > 2.0):
                #                 if skeleton_coord_camera is not None and len(skeleton_coord_camera) > 10:
                #                     l_shoulder = np.array(skeleton_coord_camera[5])
                                    
                #                     if np.linalg.norm(l_shoulder) > 0.1:
                #                         current_intent = check_reach_intent(skeleton_coord_camera)
                #                         intent_history.append(current_intent)
                                        
                #                         if len(intent_history) == intent_history.maxlen:
                #                             true_ratio = sum(intent_history) / len(intent_history)
                                            
                #                             # 如果确认用户伸手
                #                             if true_ratio >= 0.8:
                #                                 print(f"🎯 [HRI] 确认用户已伸手！正在翻转夹爪至向下递送姿态...")
                                                
                #                                 # 🌟 触发移动到夹爪向下的准备姿态
                #                                 if robot is not None:
                #                                     robot.move_to(READY_FOR_PASSING_POSE, speed=0.05)
                                                
                #                                 intent_history.clear()
                #                                 scene_mapper.looking_step = 2 # 切换到阶段 2
                #                                 hri_start_time = time.time()  # 重置计时器
                                                
                #                 else:
                #                     if time.time() - hri_start_time > 15.0:
                #                         print("⚠️ [HRI] 等待用户超时，请出现在相机视野内并伸手！")
                #                         hri_start_time = time.time()
                                        
                #         # -----------------------------------
                #         # 阶段 2：等待机械臂翻转到递送准备姿势
                #         # -----------------------------------
                #         elif getattr(scene_mapper, 'looking_step', 1) == 2:
                #             # 确保翻转动作已经走完，并且稍微给 1 秒冗余时间防抖
                #             if is_robot_idle and (time.time() - hri_start_time > 1.0):
                #                 print("✅ [HRI] 已到达向下递送姿态，切入动态追踪 (状态 7)！")
                #                 current_hri_state = STATE_TRACKING_AND_PASS
                #                 hri_start_time = 0.0
                

                # -----------------------------------
                # 状态 6：等待手掌进入工作空间 -> 翻转为递送姿势
                # -----------------------------------
                elif current_hri_state == STATE_LOOKING_FOR_USER:
                    
                    if hri_start_time == 0.0:
                        print("\n" + "="*50)
                        print("🤖 [HRI] 状态 6: 抓取成功！等待用户将手伸入工作空间...")
                        
                        hri_start_time = time.time()
                        scene_mapper.looking_step = 1 # 阶段1：等待手掌；阶段2：等待姿态翻转

                    else:
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True
                        
                        # -----------------------------------
                        # 阶段 1：静静等待有效的手掌坐标进入判定区
                        # -----------------------------------
                        if getattr(scene_mapper, 'looking_step', 1) == 1:
                            
                            # 直接读取 HoloLens 发来的原生手掌坐标
                            if global_holo_hand_pos is not None:
                                
                                # 计算手掌距离机械臂基座 [0,0,0] 的直线距离
                                dist_to_base = np.linalg.norm(global_holo_hand_pos)
                                
                                # 设定机械臂的安全工作空间半径 (例如 0.75 米)
                                WORKSPACE_RADIUS = 1.0
                                
                                if dist_to_base < WORKSPACE_RADIUS:
                                    print(f"🎯 [HRI] 检测到手掌已进入工作空间 (距离: {dist_to_base:.2f}m)！正在翻转夹爪至向下递送姿态...")
                                    
                                    # 🌟 触发移动到夹爪向下的准备姿态
                                    if robot is not None:
                                        robot.move_to(READY_FOR_PASSING_POSE, speed=0.08)
                                    
                                    scene_mapper.looking_step = 2 # 切换到阶段 2
                                    hri_start_time = time.time()  # 重置计时器
                                    
                            else:
                                # 如果一直没收到坐标，每 10 秒提醒一次
                                if time.time() - hri_start_time > 10.0:
                                    print("⚠️ [HRI] 请在 HoloLens 中开启手掌追踪，并将手伸向机械臂...")
                                    hri_start_time = time.time()
                                        
                        # -----------------------------------
                        # 阶段 2：等待机械臂翻转到递送准备姿势
                        # -----------------------------------
                        elif getattr(scene_mapper, 'looking_step', 1) == 2:
                            # 确保翻转动作已经走完，并且稍微给 1 秒冗余时间防抖
                            if is_robot_idle and (time.time() - hri_start_time > 1.0):
                                print("✅ [HRI] 已到达向下递送姿态，切入动态追踪 (状态 7)！")
                                current_hri_state = STATE_TRACKING_AND_PASS
                                hri_start_time = 0.0

                # -----------------------------------
                # 状态 7：平滑追踪 -> 稳定 -> 下降力控递送 
                # -----------------------------------
                elif current_hri_state == STATE_TRACKING_AND_PASS:
                    # 刚进入状态，初始化
                    if hri_start_time == 0.0:
                        if robot is not None:
                            robot.start_servoing()
                        hri_start_time = time.time()
                        handover_dwell_start = 0.0
                        scene_mapper.pass_step = 0  
                        print("🚀 [HRI] 正在向您的手掌上方移动...")

                    # =========================================================
                    # 阶段 0：3D 伺服追踪，直到在手掌上方稳定停留
                    # =========================================================
                    if getattr(scene_mapper, 'pass_step', 0) == 0:
                        if global_holo_hand_pos is not None:
                            ee_pos, ee_quat = robot_listener.get_current_pose() # 🌟 确保获取 ee_quat
                            
                            if ee_pos is not None:
                                final_target_pos = global_holo_hand_pos + np.array([0, 0, 0.25]) 
                                vec_to_target = final_target_pos - np.array(ee_pos)
                                dist_to_target = np.linalg.norm(vec_to_target)
                                
                                if robot is not None:
                                    robot.update_servo_target(final_target_pos)
                                
                                if T_M is not None and (time.time() - last_ray_print_time > 0.1):
                                    full_visual_path = []
                                    num_visual_points = 10
                                    for i in range(num_visual_points + 1):
                                        ratio = i / float(num_visual_points)
                                        pt = np.array(ee_pos) + ratio * vec_to_target
                                        full_visual_path.append(pt)
                                    send_path_to_hololens(conn, full_visual_path, T_M)
                                    last_ray_print_time = time.time()

                                # 带迟滞区间的防抖读秒
                                if handover_dwell_start == 0.0:
                                    if dist_to_target < 0.06:
                                        handover_dwell_start = time.time()
                                        print("⏳ [HRI] 机械臂已就位，请保持手部稳定 1 秒钟...")
                                else:
                                    if dist_to_target > 0.10:
                                        print("⚠️ [HRI] 目标大幅移动，打断稳定计时，重新追踪...")
                                        handover_dwell_start = 0.0
                                        
                                    elif time.time() - handover_dwell_start >= 0.5:
                                        print("⬇️ [HRI] 追踪稳定！停止伺服，开始缓慢下降并检测触碰...")
                                        
                                        if robot is not None:
                                            robot.stop_servoing() 
                                            
                                        # 🌟【核心修复】：将 3维位置 与 4维姿态 拼装成 7维 Pose 列表
                                        descend_target_p = np.array(ee_pos) - np.array([0, 0, 0.20])
                                        descend_full_pose = list(descend_target_p) + list(ee_quat)
                                        
                                        if robot is not None:
                                            robot.move_to(descend_full_pose, speed=0.03)
                                        
                                        scene_mapper.pass_step = 1
                                        
                    # =========================================================
                    # 阶段 1：缓慢下降，同时高频检测 Z 轴受力 (修复起步惯性误触)
                    # =========================================================
                    elif getattr(scene_mapper, 'pass_step', 0) == 1:
                        
                        force_triggered = False
                        
                        # 🌟 新增：起步动力学缓冲计时器
                        if not hasattr(scene_mapper, 'descent_start_time'):
                            scene_mapper.descent_start_time = time.time()
                        
                        # 🌟 核心：给机械臂 0.5 秒钟的时间加速到匀速，这期间不测力！
                        if time.time() - scene_mapper.descent_start_time > 0.5:
                            try:
                                wrench = robot.get_wrench() 
                                current_fz = wrench[2] 
                                
                                # 在匀速阶段记录基准力，此时受力极其稳定
                                if not hasattr(scene_mapper, 'baseline_fz'):
                                    scene_mapper.baseline_fz = current_fz
                                    print(f"⚖️ [HRI] 运动已平稳，记录基准受力: Fz = {current_fz:.2f}N")
                                
                                delta_fz = abs(current_fz - scene_mapper.baseline_fz)
                                
                                # 稍微提高一点点阈值增加鲁棒性 (3.5N)
                                FORCE_THRESHOLD = 3.5 
                                if delta_fz > FORCE_THRESHOLD:
                                    force_triggered = True
                                    print(f"🖐️ [HRI] 检测到接触力变化 (ΔFz={delta_fz:.2f}N)！用户已接稳！")
                                    
                            except Exception as e:
                                pass
                            
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True
                        
                        # 情况 A：成功摸到了手
                        if force_triggered:
                            if robot is not None:
                                robot.path_queue.queue.clear() 
                                
                            print("🤝 [HRI] 成功移交，释放物品！")
                            time.sleep(0.2) 
                            
                            if robot is not None:
                                robot.open_gripper(width=0.08) 
                            
                            if T_M is not None:
                                send_path_to_hololens(conn, [], T_M)
                                
                            current_hri_state = STATE_SCAN_OBJECTS
                            hri_start_time = 0.0
                            handover_dwell_start = 0.0
                            global_holo_hand_pos = None
                            scene_mapper.pass_step = 0
                            
                            # 🧹 清理内存
                            if hasattr(scene_mapper, 'baseline_fz'): del scene_mapper.baseline_fz
                            if hasattr(scene_mapper, 'descent_start_time'): del scene_mapper.descent_start_time
                                
                        # 情况 B：走到底了没摸到力 (扑空回收)
                        elif is_robot_idle:
                            # 注意：如果是前 0.5s 还没有路径进入队列，可能会误判 idle，所以要确保过了缓冲期
                            if time.time() - getattr(scene_mapper, 'descent_start_time', 0) > 0.5:
                                print("⚠️ [HRI] 递送扑空！未检测到手部支撑。")
                                print("🔄 [HRI] 取消释放，正在退回安全高度...")
                                
                                ee_pos, ee_quat = robot_listener.get_current_pose()
                                if ee_pos is not None:
                                    retreat_target_p = np.array(ee_pos) + np.array([0, 0, 0.15])
                                    retreat_full_pose = list(retreat_target_p) + list(ee_quat)
                                    
                                    if robot is not None:
                                        robot.move_to(retreat_full_pose, speed=0.05)
                                
                                scene_mapper.pass_step = 0
                                handover_dwell_start = 0.0
                                
                                # 🧹 清理内存
                                if hasattr(scene_mapper, 'baseline_fz'): del scene_mapper.baseline_fz
                                if hasattr(scene_mapper, 'descent_start_time'): del scene_mapper.descent_start_time


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