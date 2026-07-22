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
import threading
import select
import traceback

# netsh interface portproxy add v4tov4 listenaddress=192.168.137.1 listenport=[本地监听端口] connectaddress=[机械臂的Tailscale IP] connectport=[机械臂服务端口]

tcp_send_lock = threading.Lock()
def safe_send_packet(conn, packet, timeout_sec=1.0):
    """
    带熔断机制的绝对安全发送：
    1. 保证不会发生粘包和碎包。
    2. 如果对方停止接收超过 timeout_sec 秒，直接抛出异常熔断，绝不卡死主线程！
    """
    global tcp_send_lock
    with tcp_send_lock:
        conn.setblocking(False) # 始终保持非阻塞
        total_sent = 0
        packet_len = len(packet)
        start_time = time.time()

        while total_sent < packet_len:
            # 使用 select 监听网卡，每次最多等 0.1 秒
            _, writable, _ = select.select([], [conn], [], 0.1)
            
            # 🌟 熔断机制：如果等了 1 秒对方还没收走数据，直接抛弃对方！
            if time.time() - start_time > timeout_sec:
                raise ConnectionError("⏱️ [TCP 熔断] 发送超时！HoloLens 疑似卡死，强制丢弃包裹以保全机器人控制！")
            
            if writable:
                try:
                    # 能发多少发多少
                    sent = conn.send(packet[total_sent:])
                    total_sent += sent
                except BlockingIOError:
                    continue # 缓冲区满，下一圈接着试
                except Exception as e:
                    raise e # 网络真断了，向外抛出
                

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

OFFSET_CONFIG_FILE = "pointcloud_offset.yaml"

def load_z_offset():
    """读取本地存储的 Z 轴偏置量，默认 -0.06"""
    try:
        if os.path.exists(OFFSET_CONFIG_FILE):
            with open(OFFSET_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and 'z_offset' in data:
                    return float(data['z_offset'])
    except Exception as e:
        print(f"⚠️ Load Z offset failed: {e}, using default 0")
    return 0  # 默认值

def save_z_offset(offset):
    """保存 Z 轴偏置量到本地"""
    try:
        with open(OFFSET_CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump({'z_offset': float(offset)}, f)
    except Exception as e:
        print(f"❌ Save Z offset failed: {e}")
# =========================================================

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

try:
    from yolo_detector import YoloGraspDetector
except ImportError:
    print("⚠️ Warning: yolo_detector.py not found. No way to compute grasp pose with YOLO.")
    YoloGraspDetector = None


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
        try:
            safe_send_packet(conn, b't' + flat_T_M.tobytes())
        except ConnectionError as e:
            print(e)
            # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
        except Exception:
            pass
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
        
        # 新的安全代码
        try:
            safe_send_packet(conn, full_packet)
        except ConnectionError as e:
            print(e)
            # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
            # 跳出内层 while 循环，重新等待 HoloLens 连接
        except Exception:
            pass
        print(f"   -> Sent Ball Pos: {unity_pos}")

        # ==================== 调试打印区 ====================
        print("-" * 50)
        print(f"DEBUG: Send Robot Ball Position:")
        print(f"  -> Coordination (Unity): X={unity_pos[0]:.4f}, Y={unity_pos[1]:.4f}, Z={unity_pos[2]:.4f}")
        
        # 打印十六进制，方便与 Unity 端逐字节比对
        # b'b' 的十六进制是 62
        hex_data = full_packet.hex(' ')
        print(f"  -> Original Data Stream (Hex): {hex_data}")
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
        # 新的安全代码
        try:
            safe_send_packet(conn, header+payload)
        except ConnectionError as e:
            print(e)
            # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
            # 跳出内层 while 循环，重新等待 HoloLens 连接
        except Exception:
            pass
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
            # 新的安全代码
            try:
                safe_send_packet(conn, header+payload)
            except ConnectionError as e:
                print(e)
                # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
                # 跳出内层 while 循环，重新等待 HoloLens 连接
            except Exception:
                pass
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
        # 新的安全代码
        try:
            safe_send_packet(conn, header+payload)
        except ConnectionError as e:
            print(e)
            # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
            # 跳出内层 while 循环，重新等待 HoloLens 连接
        except Exception:
            pass

    except BlockingIOError:
        pass # 非阻塞模式下的系统级等待，可以直接忽略
    except Exception as e:
        print(f"⚠️ [TCP] 发送动态路径至 HoloLens 失败: {e}")


def send_hri_status_packet(conn, state_id, sub_state_id):
    """
    [升级版] 同时同步 HRI 主状态和子状态到 HoloLens
    Header: 'I' (1 byte) + payload (8 bytes: 2个小端序 int)
    """
    try:
        header = b'I'
        # 打包两个 int：主状态 ID 和 子状态/步骤 ID
        payload = struct.pack('<ii', state_id, sub_state_id)
        
        conn.setblocking(True)
        # 新的安全代码
        try:
            safe_send_packet(conn, header+payload)
        except ConnectionError as e:
            print(e)
            # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
            # 跳出内层 while 循环，重新等待 HoloLens 连接
        except Exception:
            pass

        print(f"📢 [TCP] 状态全同步 -> Main: {state_id} | Sub: {sub_state_id}")
    except BlockingIOError:
        pass
    except Exception as e:
        print(f"❌ [TCP] 同步 HRI 状态失败: {e}")


def send_point_cloud_to_hololens(conn, pcd, T_M):
    """
    高性能点云发送：降采样、坐标系转换、二进制打包
    """
    try:
        if pcd is None or pcd.is_empty(): return
        
        # 1. 体素降采样 (极其重要！设置 1cm，既能看清形状，又能把点数压缩到 1万~3万点以内)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        points = np.asarray(downsampled_pcd.points)
        # 如果有点云颜色则提取，没有就默认白色
        colors = np.asarray(downsampled_pcd.colors) if downsampled_pcd.has_colors() else np.ones_like(points)
        
        num_points = len(points)
        if num_points == 0: return
        
        # 2. 坐标转换 (Robot -> Unity)
        # 因为 rut 可能没有向量化，我们用列表推导式批量转换，1万个点大概只需要几毫秒
        unity_points = []
        valid_indices = []
        for i, pt in enumerate(points):
            u_pt = rut.robot2unity_transform(pt, T_M)
            if u_pt is not None:
                unity_points.append(u_pt)
                valid_indices.append(i)
                
        if not unity_points: return
        
        unity_points = np.array(unity_points, dtype=np.float32)
        unity_colors = colors[valid_indices].astype(np.float32)
        
        # 3. 打包：每个点 6 个 float -> [x, y, z, r, g, b]
        # 使用 hstack 拼接然后展平为一维数组
        payload_data = np.hstack((unity_points, unity_colors)).flatten()
        
        # 4. 发送 TCP 数据包
        header = b'C'
        count_bytes = struct.pack('<i', len(unity_points))
        payload_bytes = payload_data.tobytes() # 自动转为小端序的 bytes
        
        # 新的安全代码
        try:
            safe_send_packet(conn, header+count_bytes+payload_bytes, timeout_sec=5.0)
        except ConnectionError as e:
            print(f"💥 [TCP] 点云发送超时或断开: {e}")
            # 🌟 致命防线：既然没发完，TCP 流已经错位了！
            # 绝对不能 pass！必须向上抛出异常，让主程序的 except 捕获并重启连接！
            raise e
        except Exception:
            pass

        
        print(f"☁️ [TCP] 已向 HoloLens 发送对齐点云，总计: {len(unity_points)} 点 ({len(payload_bytes)/1024:.1f} KB)")
        
    except BlockingIOError:
        pass
    except Exception as e:
        print(f"❌ [TCP] 发送点云失败: {e}")

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

  # 初始化 YOLO 检测器
    yolo_detector = YoloGraspDetector(weights_filename="yolo_weights.pt", conf=0.30)
    
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
        print(" [System] Ready to go without manual calibration.")
    else:
        print(" [System] No calibration cache found. Calibration required.")

    # 初始化读取Z轴偏置 与 阀门状态
    global_z_offset = load_z_offset()
    is_offset_edit_mode = False  # 阀门状态：False表示全自动放行，True表示扫描后暂停等待修改
    print(f"🔧 [Config] 初始点云 Z 轴偏置量已加载: {global_z_offset:.4f} 米")

    # ===  HRI 状态机初始化 ===
    is_HRI_Demo = False  # 是否进入 Demo 模式的总开关
    STATE_IDLE = -1          # 待机状态
    STATE_INIT = 0           # 0. 机械臂前往初始姿态
    STATE_TRACKING = 1       # 1. 跟踪人体，维持在视野中央
    STATE_CHECK_INTENT = 2   # 2. 识别意图 
    STATE_SCAN_OBJECTS = 3   # 3. 扫描桌面物品
    STATE_GAZE_INTERSECTION = 4 # 4. gaze selection
    STATE_GRAB_OBJECT = 5       # 5. 抓取物品
    STATE_GRAB_OBJECT_YOLO = 51 # 51. 使用 YOLO 模型抓取物品
    STATE_LOOKING_FOR_USER = 6     # 6. 寻找用户来拿
    STATE_TRACKING_AND_PASS = 7     # 7. 跟踪并递给用户
    
    has_completed_initial_scan = False

    current_hri_state = STATE_IDLE
    current_sub_state = 0  # 新增：统一的当前子状态变量
    hri_start_time = 0.0     # 记录状态切换的时间，用于非阻塞等待
    handover_dwell_start = 0.0

    last_synced_state = -999
    last_synced_sub_state = -999
    
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
    
    instant_select_flag = False

    current_point_cloud = None

    global_holo_hand_pos = None
    
    # ====== Multipointcloud test ======
    # 这里使用的是 [x, y, z, qx, qy, qz, qw] 或关节角，只要符合你的 robot.move_to 格式即可

    SCAN_START_POSE = [0.4600, 0.2700, 0.5, 0.91, -0.42, 0.04, -0.02]
    SCAN_END_POSE   = [0.4600, -0.1800, 0.5, -0.92, 0.38, -0.04, -0.01] #[INFO] [1777634168.026776]: sent #149 UPDATED d=0.000223 xyz=(0.5621, -0.1739, 0.5610) Euler[Deg]=(Rx:-179.5, Ry:-4.6, Rz:-45.0) q=(-0.92, 0.38, -0.04, -0.01)
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
                    print(f"[TCP] Connection Error: {e}")
                    break
                
                if header in ['d', 'r', 'b', 'm', 'p', 'v', 'f', 'e', 'R', 'h', 'O', 'P', 'Z', 'U', 'V']:
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
                            print(">> Connecting to robot...")
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
                        print("[TCP]  Header 'p' Detected, Receiving Pose Sequence...")
                        
                        # 1. 读取点数 (4字节 Int)
                        count_bytes = recv_exact(conn, 4)
                        if not count_bytes: 
                            print("   ❌ failed to receive point count")
                            break
                        
                        num_points = struct.unpack('<i', count_bytes)[0]
                        print(f"   -> Planned Points: {num_points}")

                        # 2. 读取数据包 (每个点 28 字节: 3 float pos + 4 float rot)
                        bytes_per_point = 28
                        total_bytes = num_points * bytes_per_point
                        data_bytes = recv_exact(conn, total_bytes)
                        if not data_bytes: 
                            print("   ❌ failed to receive pose data")
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
                                    print(f"       Position: Unity {np.round(u_pos, 2)} -> 机器人 {np.round(r_pos, 3)}")
                                    print(f"       Orientation: Original RPY(Unity): {np.round(res['raw_rpy'], 1)}°")
                                    print(f"       Orientation: Transformed RPY(Robot): {np.round(res['robot_rpy'], 1)}°")
                                    # 如果需要调试 XYZ 单位向量，可以打印 res['rhs_axes']

                            # 4. 生成带姿态插值的平滑路径
                            if len(path_with_orientations) >= 2:
                                print(f"\n   [Interpolation] Generating Smooth Path...")
                                final_smooth_path = pathInterpolation.generate_smooth_path_with_orientation(
                                    path_with_orientations, 
                                    resolution=3
                                )
                                final_smooth_path.append(path_with_orientations[-1])
                                
                                # 5. 执行机器人运动
                                if robot is None: 
                                    robot = RobotController()
                                
                                print(f"   🚀 Starting Execution, Total Interpolated Points: {len(final_smooth_path)}")
                                # 发送给机械臂执行
                                robot.execute_path(final_smooth_path, speed=0.02)

                                # 2. 🌟 关键：增加等待逻辑
                                # 假设目标点是路径的最后一个点
                                target_pos = final_smooth_path[-1]['pos'] 

                                print(" ⏳ Waiting for robot to reach target position...")
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
                                    # 新的安全代码
                                    try:
                                        safe_send_packet(conn, b'm')
                                    except ConnectionError as e:
                                        print(e)
                                        # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
                                        # 跳出内层 while 循环，重新等待 HoloLens 连接
                                    except Exception:
                                        pass
                                    print("   ✅ [TCP] Robot motion completed, 'm' unlock signal sent to HoloLens")
                                except Exception as e:
                                    print(f"   ❌ Failed to send completion signal: {e}")
                        else:
                            print("   ⚠️ T_M matrix is empty, please calibrate first by sending 'c'!")
                        print("="*50 + "\n")

                    # ===============================================
                    # CASE 'f': 接收力控点序列 (Position + Orientation + Force)
                    # ===============================================
                    elif header == 'f':
                        print("\n" + "="*50)
                        print("[TCP]  Header 'f' Detected, starting to parse 【Force Control】pose sequence...")
                        
                        # 1. 读取点数 (4字节 Int)
                        count_bytes = recv_exact(conn, 4)
                        if not count_bytes: 
                            print("   ❌ Failed to receive force control point count")
                            break
                        
                        num_points = struct.unpack('<i', count_bytes)[0]
                        print(f"   -> Plan to receive force control key points: {num_points}")

                        # 2. 读取数据包 (每个点 32 字节: 3 pos + 4 rot + 1 force)
                        bytes_per_point = 32
                        total_bytes = num_points * bytes_per_point
                        data_bytes = recv_exact(conn, total_bytes)
                        if not data_bytes: 
                            print("   ❌ Failed to receive complete force control data")
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

                                    print(f"   [{i}] Position:{np.round(r_pos, 3)} | Orientation RPY:{np.round(res['robot_rpy'], 1)}° | Target Force:{target_force:.1f}N")

                            # 4. 生成带姿态和【力】的平滑路径
                            if len(path_with_force) >= 2:
                                print(f"\n   [Interpolation] Generating Smooth Path with Force...")
                                
                                # 注意：你的插值函数需要更新，以支持 force 字段（见下文）
                                final_smooth_path = pathInterpolation.generate_smooth_path_with_orientation(
                                    path_with_force, 
                                    resolution=3
                                )
                                
                                final_smooth_path.append(path_with_force[-1])
                                # 5. 执行机器人运动
                                if robot is None: 
                                    robot = RobotController()
                                
                                print(f"   🚀 Starting Force Control Trajectory, Total Interpolated Points: {len(final_smooth_path)}")
                                # 统一使用 execute_path，不再需要传 mode 参数
                                robot.execute_path(final_smooth_path, speed=0.02)

                                # 2. 🌟 关键：增加等待逻辑
                                # 假设目标点是路径的最后一个点
                                target_pos = final_smooth_path[-1]['pos'] 

                                print(" ⏳ Waiting for robot to reach target position...")
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
                                    # 新的安全代码
                                    try:
                                        safe_send_packet(conn, b'm')
                                    except ConnectionError as e:
                                        print(e)
                                        # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
                                        # 跳出内层 while 循环，重新等待 HoloLens 连接
                                    except Exception:
                                        pass
                                    print("   ✅ [TCP] Robot force control motion completed, 'm' unlock signal sent to HoloLens")
                                except Exception as e:
                                    print(f"   ❌ Failed to send completion signal: {e}")
                        else:
                            print("   ⚠️ T_M matrix is empty, please calibrate first by sending 'c'!")
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
                            print(f" ⚠️ Image decoding failed, received length: {img_len}")

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
                            print("   ⚠️ T_M is None. Cannot transform gaze coordinates. Please calibrate first.")

                    # ===============================================
                    # CASE 'R': 接收原生手部射线坐标 (Hand Ray Pinch)
                    # ===============================================
                    elif header == 'R':
                        ray_data_bytes = recv_exact(conn, 24)
                        if not ray_data_bytes: break
                        
                        ray_data = struct.unpack('<6f', ray_data_bytes)
                        u_origin_pos = np.array(ray_data[0:3])
                        u_hit_pos = np.array(ray_data[3:6])
                        
                        if T_M is not None:
                            # 坐标转换
                            r_origin_pos = rut.unity2robot_transform(u_origin_pos, T_M)
                            r_hit_pos = rut.unity2robot_transform(u_hit_pos, T_M)
                            
                            # 🌟 覆盖给通用的射线全局变量 (和眼动共用一套碰撞逻辑)
                            global_ray_origin = r_origin_pos
                            global_ray_hit = r_hit_pos
                            
                            # 🌟 核心：点燃“立刻选择”标记！
                            instant_select_flag = True 
                            
                            print(f"🤏 [TCP] Pinch Click Detected! Hand Ray: {np.round(r_hit_pos, 3)}")
                        else:
                            print("   ⚠️ T_M is None. Cannot transform hand ray coordinates.")

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
                                print("  ⚠️ T_M is None. Cannot transform hand coordinates. Please calibrate first.")

                    elif header == 'O': 
                        is_HRI_Demo = True
                        print("Demo Mode Activated")
                    elif header == 'P': 
                        is_HRI_Demo = False
                        print("Demo Mode Deactivated")

                    # ===============================================
                    # 🌟 阀门与偏置控制：'U'(开阀门编辑), 'V'(关阀门放行), 'Z'(微调高度)
                    # ===============================================
                    elif header == 'U':
                        is_offset_edit_mode = True
                        print("\n🔒 [Valve] Offset Edit Mode Activated")
                        
                    elif header == 'V':
                        is_offset_edit_mode = False
                        scene_mapper.valve_open = True  # 立即放行当前的阻塞
                        print("\n🔓 [Valve] Offset Edit Mode Deactivated (Auto Release)！")
                        
                    elif header == 'Z':
                        # 接收 4 字节的 float (小端序)，代表要增减的数值 delta (如 +0.01 或 -0.01)
                        delta_bytes = recv_exact(conn, 4)
                        if delta_bytes:
                            delta_z = struct.unpack('<f', delta_bytes)[0]
                            global_z_offset += delta_z
                            save_z_offset(global_z_offset) # 立即持久化
                            
                            print(f"\n📐 [Offset] Offset adjusted by {delta_z*100:.2f} cm. Current total offset: {global_z_offset:.4f} m")
                            
                            # === 核心：对内存中已经建好的地图进行实时位移，并刷新 HoloLens ===
                            if hasattr(scene_mapper, 'global_pcd') and scene_mapper.global_pcd is not None and not scene_mapper.global_pcd.is_empty():
                                shift_vec = np.array([0.0, 0.0, delta_z])
                                scene_mapper.global_pcd.translate(shift_vec)
                                if hasattr(scene_mapper, 'display_pcd'):
                                    scene_mapper.display_pcd.translate(shift_vec)
                                if hasattr(scene_mapper, 'objects_pcd') and scene_mapper.objects_pcd is not None and not scene_mapper.objects_pcd.is_empty():
                                    scene_mapper.objects_pcd.translate(shift_vec)
                                    
                                scene_mapper.update_window()
                elif header == 'S': 
                    sender.is_streaming = True
                    print("▶️ Video Streaming Started")
                elif header == 'E': 
                    sender.is_streaming = False
                    print("⏹️ Video Streaming Stopped")

                elif header == 'K': 
                    is_skeleton_streaming = True
                    print("▶️ Skeleton Streaming Started")
                elif header == 'L': 
                    is_skeleton_streaming = False
                    print("⏹️ Skeleton Streaming Stopped")
                
                elif header == 'J': 
                    is_robot_state_streaming = True
                    print("▶️ Started streaming 【Full Joint Real-time Poses】(Converted on Python side)")
                elif header == 'H': 
                    is_robot_state_streaming = False
                    print("⏹️ Stopped streaming 【Full Joint Real-time Poses】")

                

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
                                if len(skeleton_coord_unity) >= 17:
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

                        # 新的安全代码
                        try:
                            safe_send_packet(conn, header_j+payload_j)
                        except ConnectionError as e:
                            print(e)
                            # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
                            # 跳出内层 while 循环，重新等待 HoloLens 连接
                        except Exception:
                            pass


                        
                    except Exception as e:
                        pass
            
            if is_HRI_Demo:
                # -1. 待机状态：等待 'O' 信号启动
                if current_hri_state == STATE_IDLE:
                    if robot is None:
                        robot = RobotController()
                    print("\n" + "="*50)
                    print("🤖 [HRI] Command received, starting HRI workflow!")
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
                        print("🔄 [HRI] State 0 (Initial Pose): Moving robot to initial observation pose...")
                        if robot is not None:
                            robot.move_to(INIT_POSE, speed=0.05)
                        hri_start_time = time.time() # 🌟 开始计时！
                        
                    # 每次循环检查时间是否够了，如果没够就直接跳过，继续发视频
                    else:
                        if time.time() - hri_start_time > 20.0:
                            print(" ✅ [HRI] State 0 (Initial Pose): 20 seconds passed. Starting visual servoing to track the user...")
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
                                print(f" 🎯 [HRI] State 1 (Tracking): Continuous confirmation of reaching intent (Confidence: {true_ratio*100:.1f}%)! Stopping tracking, preparing to look at the table...")
                                
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
                            print(f"👀 [HRI] State 1 (Tracking) Chest Center: X={cam_x:.4f}m, Y={cam_y:.4f}m, DepthZ={chest_3d[2]:.4f}m")
                            
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
                        print("👀 [HRI] State 2 (Checking Intent): Moving robot to look at the table...")
                        if robot is not None:
                            # 前往我们预设好的桌面观测点
                            # 请确保在 main() 开头定义了 LOOK_AT_TABLE_POSE = [x, y, z, qx, qy, qz, qw]
                            #pass
                            robot.move_to(LOOK_AT_TABLE_POSE, speed=0.05)
                            
                        hri_start_time = time.time()
                        
                    else:
                        # 非阻塞等待机械臂走到位 (这里给 4 秒时间，可根据实际距离调整)
                        if time.time() - hri_start_time > 10.0:
                            print(" ✅ [HRI] State 2 (Checking Intent): Line of sight locked on the table! Preparing to enter object recognition (State 3)...")
                            current_hri_state = STATE_SCAN_OBJECTS
                            hri_start_time = 0.0

                # -----------------------------------
                # 状态 3：全自动直线巡航建图 (一次扫描，多次复用版)
                # -----------------------------------
                elif current_hri_state == STATE_SCAN_OBJECTS:
                    
                    if hri_start_time == 0.0:
                        if not has_completed_initial_scan:
                            print(f"\n🚀 [HRI] State 3 (Scanning Objects): Starting first automatic physical scan. Planned points: {SCAN_STEPS}")
                            scan_current_step = 0 
                            if robot is not None:
                                robot.move_to(scan_waypoints[scan_current_step], speed=0.05)
                        else:
                            print(f"\n🚀 [HRI] State 3 (Scanning Objects): Skipping physical scan, directly reusing and refreshing existing environment memory...")
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
                                
                                # 🌟【核心魔改】基于当前步数，动态配置就位规则与消震时间
                                if scan_current_step == 0:
                                    # 🚨 仅针对起点：引入手腕旋转判定 + 0.8秒强力消震
                                    target_quat = scan_waypoints[scan_current_step][3:7]
                                    is_rotation_ok = robot.is_rotation_reached(target_quat, tolerance_deg=1.5) if robot is not None else True
                                    
                                    is_ready_to_stop = (dist < 0.02) and is_rotation_ok and is_robot_idle
                                    delay_threshold = 0.8
                                    arrival_log = "📍 [State 3 (Scanning Objects)] Position and wrist angle aligned 100%! Starting 0.8s anti-shake..."
                                else:
                                    # 🟢 针对中间点：放开限制，不检查旋转，恢复你原本的 0.5 秒轻度停顿
                                    is_ready_to_stop = (dist < 0.02) and is_robot_idle
                                    delay_threshold = 0.5
                                    arrival_log = f"📍 [State 3 (Scanning Objects)] Arrived at node {scan_current_step + 1}! Pausing 0.5s for anti-shake..."

                                # -----------------------------------------------------
                                # 第一阶段：就位拦截（起点用严格规则，中间点用宽松规则）
                                # -----------------------------------------------------
                                if last_capture_time == 0.0:
                                    if is_ready_to_stop: 
                                        print(arrival_log)
                                        last_capture_time = time.time()
                                        
                                # -----------------------------------------------------
                                # 第二阶段：满足各自的倒计时后，统一执行点云获取
                                # -----------------------------------------------------
                                else:
                                    if time.time() - last_capture_time > delay_threshold: 
                                        print(f"📸 [State 3 (Scanning Objects)] Acquiring point cloud from view {scan_current_step + 1}/{SCAN_STEPS}...")
                                        
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

                                            # Manual Offset
                                            # verts_robot_base[:,2]-=0.06


                                            scene_mapper.add_point_cloud(verts_robot_base, c_crop)
                                            print(f"   ✅ [State 3 (Scanning Objects)] Node {scan_current_step + 1}successfully captured and added to the global point cloud. Total points: {len(scene_mapper.global_pcd.points)}")
                                        else:
                                            print(f"   ⚠️ [State 3 (Scanning Objects)] Node {scan_current_step + 1} skipped: No point cloud data received from the camera!")
                                            
                                        # 迈向下一步
                                        scan_current_step += 1
                                        if scan_current_step < SCAN_STEPS:
                                            if robot is not None:
                                                robot.move_to(scan_waypoints[scan_current_step], speed=0.05)
                                        last_capture_time = 0.0

                                        
                            # =========================================================
                            # 【已走完或直接跳过时】：进入安全后处理与退回阶段
                            # =========================================================
                            else:
                                # ---------------------------------------------------------
                                # 阶段 0：滤波 -> 【全局偏置】 -> 聚类 -> 发送展示 -> 关阀门
                                # ---------------------------------------------------------
                                if getattr(scene_mapper, 'scan_post_process', 0) == 0:
                                    print("\n🎉 State 3 (Scanning Objects): Starting processing/refreshing point cloud map data...")
                                    
                                    final_pcd = scene_mapper.global_pcd
                                    
                                    if final_pcd is None or len(final_pcd.points) < 50:
                                        print("⚠️ State 3 (Scanning Objects): Table is empty! Automatically switching back to idle state...")
                                        current_hri_state = STATE_IDLE
                                        hri_start_time = 0.0
                                        scene_mapper.scan_post_process = 0
                                        continue

                                    print("   ✂️ State 3 (Scanning Objects): 0. Performing Z axis cropping...")
                                    points = np.asarray(final_pcd.points)
                                    valid_z_indices = np.where(points[:, 2] >= -0.25)[0] 
                                    final_pcd = final_pcd.select_by_index(valid_z_indices)

                                    print("   🧹 State 3 (Scanning Objects): 1. Performing statistical filtering...")
                                    cleaned_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)

                                    # 🌟【核心】：在这里进行初始的全局偏置
                                    print(f"   📐 State 3: Applying initial global Z offset: {global_z_offset:.4f} m")
                                    cleaned_pcd.translate(np.array([0.0, 0.0, global_z_offset]))
                                    
                                    scene_mapper.global_pcd = cleaned_pcd
                                    scene_mapper.display_pcd.points = cleaned_pcd.points
                                    scene_mapper.display_pcd.colors = cleaned_pcd.colors
                                    scene_mapper.update_window()

                                    try:
                                        o3d.io.write_point_cloud("scanned_scene.pcd", cleaned_pcd)
                                        if T_M is not None:
                                            send_point_cloud_to_hololens(conn, cleaned_pcd, T_M)
                                    except Exception as e:
                                        pass

                                    # 分割与聚类
                                    print("   🪚 State 3 (Scanning Objects): 2. Identifying table (RANSAC)...")
                                    plane_model, inliers = cleaned_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                                    objects_pcd = cleaned_pcd.select_by_index(inliers, invert=True)

                                    table_points = np.asarray(cleaned_pcd.select_by_index(inliers).points)
                                    scene_mapper.table_z = np.mean(table_points[:, 2]) 
                                    
                                    print("   📦 State 3 (Scanning Objects): 3. Clustering remaining objects...")
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
                                    print(f"   ✅ State 3 (Scanning Objects): Environment refreshed! {clean_labels.max() + 1 if len(clean_labels)>0 else 0} objects remaining.")
                                    
                                    ee_pos_snap, ee_quat_snap = robot_listener.get_current_pose()
                                    scene_mapper.locked_robot_t_c = camera2unity.get_camera_to_robot_matrix(ee_pos_snap, ee_quat_snap, EE_T_C)

                                    if robot is not None:
                                        robot.move_to(READY_FOR_PASSING_POSE, speed=0.05)
                                    
                                    # 🌟【状态阀门判定逻辑】
                                    if is_offset_edit_mode:
                                        print("\n   ⏳ [Valve Closed] 偏置编辑模式已开启！正在挂起，等待 HoloLens 'Z' 调整或 'V' 放行...")
                                        scene_mapper.valve_open = False
                                    else:
                                        print("\n   ⏩ [Valve Open] 自动模式！直接加载默认偏置，放行进入 AI 计算...")
                                        scene_mapper.valve_open = True
                                        
                                    scene_mapper.scan_post_process = 3
                                    hri_start_time = time.time()


                                # ---------------------------------------------------------
                                # 🌟 阶段 0.5：等待阀门开启，开启后提取当前内存点云计算抓取位姿
                                # ---------------------------------------------------------
                                elif getattr(scene_mapper, 'scan_post_process', 0) == 3:
                                    if getattr(scene_mapper, 'valve_open', False):
                                        print("\n   🧠 [AI Pipeline] 提取当前【已偏置点云】并送入神经网络运算...")

                                        if request_grasps_from_graspnet is not None and not has_completed_initial_scan:
                                            
                                            ROBOT_T_C = scene_mapper.locked_robot_t_c
                                            C_T_ROBOT = np.linalg.inv(ROBOT_T_C)

                                            # 这里提取的全局点云，可能已经在刚刚的等待期被 'Z' 调整过
                                            real_points_base = np.asarray(scene_mapper.global_pcd.points)
                                            ones = np.ones((real_points_base.shape[0], 1))
                                            points_hom = np.hstack((real_points_base, ones))
                                            
                                            # 逆变换回相机坐标系
                                            real_points_camera = (C_T_ROBOT @ points_hom.T).T[:, :3] 
                                            real_colors = np.asarray(scene_mapper.global_pcd.colors) if scene_mapper.global_pcd.has_colors() else np.zeros_like(real_points_base)

                                            def ai_task(points, colors, K, ip):
                                                return request_grasps_from_graspnet(points, colors, K, server_ip=ip)

                                            future = executor.submit(ai_task, real_points_camera, real_colors, K_1, "100.116.99.44")
                                            scene_mapper.current_robot_t_c = ROBOT_T_C 
                                            scene_mapper.ai_future = future
                                            
                                            has_completed_initial_scan = True
                                            scene_mapper.has_ai_request = True
                                        else:
                                            scene_mapper.has_ai_request = False

                                        # 🌟🌟🌟 这里是解决死循环的关键：一定要推动状态机往前走！
                                        scene_mapper.scan_post_process = 2
                                
                              # ---------------------------------------------------------
                                # 阶段 2：等待 AI 返回并处理结果
                                # ---------------------------------------------------------
                                elif getattr(scene_mapper, 'scan_post_process', 0) == 2:
                                    
                                    # 如果发起了 AI 请求，那就等它完成
                                    if getattr(scene_mapper, 'has_ai_request', True):
                                        if scene_mapper.ai_future is not None and scene_mapper.ai_future.done():
                                            ai_grasps_camera = scene_mapper.ai_future.result() 
                                            
                                            if ai_grasps_camera is not None:
                                                print(f"   🧠 State 3: AI predicted {ai_grasps_camera.shape[0]} poses, transforming to robot base frame...")
                                                
                                                ROBOT_T_C = scene_mapper.current_robot_t_c
                                                ai_grasps_base = np.zeros_like(ai_grasps_camera)
                                                
                                                for i in range(ai_grasps_camera.shape[0]):
                                                    ai_grasps_base[i] = ROBOT_T_C @ ai_grasps_camera[i]
                                                    
                                                scene_mapper.ai_grasps = ai_grasps_base
                                                np.save("test_output_grasps.npy", ai_grasps_base)
                                                print("   🎉 [State 3] Inference completed! File saved.")
                                                
                                            scene_mapper.has_ai_request = False
                                            scene_mapper.scan_post_process = 1
                                    else:
                                        # 没有发起请求（比如重复利用旧数据），直接跳入阶段 1
                                        scene_mapper.scan_post_process = 1
                                                                                    
                                elif getattr(scene_mapper, 'scan_post_process', 0) == 1:
                                    if is_robot_idle and (time.time() - hri_start_time > 1.0):
                                        print("\n👀 [State 3 (Scanning Objects)] Switching to state 4: Please use gaze to select the target object...")
                                        current_hri_state = STATE_GAZE_INTERSECTION
                                        hri_start_time = 0.0
                                        scene_mapper.scan_post_process = 0 # 重置标记


                # -----------------------------------
                # 状态 4：全景眼动求交 + 纯净物品吸附
                # -----------------------------------
                elif current_hri_state == STATE_GAZE_INTERSECTION:
                                        
                    if hri_start_time == 0.0:
                        print("👀 [State 4 (Gaze Intersection)]: Please gaze at the object you want to grasp (hold for 2 seconds)...")
                        hri_start_time = time.time()
                        debug_print_time = time.time() 
                        
                        fixation_point = None       
                        fixation_start_time = 0.0   
                        FIXATION_TOLERANCE = 0.09   
                        FIXATION_TIME_REQUIRED = 1.0 
                        
                    else:
                        scene_mapper.update_window() 

                        # =======================================================
                        # 🛑 阶段 999：路线预演阻断模式（展示路径 1.5 秒后清除并重置）
                        # =======================================================
                        if getattr(scene_mapper, 'gaze_step', 0) == 999:
                            if time.time() - hri_start_time > 1.5:
                                print("   🧹 [预演模式] 1.5秒观察结束，清空预测路线并重新开启自由选择！\n" + "="*50)
                                if T_M is not None:
                                    send_path_to_hololens(conn, [], T_M) # 发送空数组，清除丝带
                                
                                # 重置各类状态，回到阶段 0 允许用户再次选择
                                scene_mapper.selected_grasp = None
                                scene_mapper.gaze_step = 0
                                fixation_point = None
                                fixation_start_time = time.time()
                            continue # ⚠️ 非常重要：在预演期间拦截下方的射线求交逻辑

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
                            current_time = time.time()
                            # 利用 scene_mapper 存一下上次发送的时间，避免报错
                            last_send = getattr(scene_mapper, 'last_cursor_send_time', 0.0)
                            
                            # 每隔 0.033 秒 (30Hz) 才允许发送一次
                            if current_time - last_send > 0.033:
                                try:
                                    u_pt = rut.robot2unity_transform(current_pt, T_M)
                                    packet_p = b't' + struct.pack('<fff', u_pt[0], u_pt[1], u_pt[2])
                                    # 新的安全代码
                                    try:
                                        safe_send_packet(conn, packet_p)
                                    except ConnectionError as e:
                                        print(e)
                                        # 可以在这里做个标记，比如触发你在 main() 里的重连大循环
                                        # 跳出内层 while 循环，重新等待 HoloLens 连接
                                    except Exception:
                                        pass
                                    # 记录发送时间
                                    scene_mapper.last_cursor_send_time = current_time
                                except Exception as e:
                                    pass
                        # ==========================================

                        if time.time() - debug_print_time > 0.5:
                            if current_pt is not None:
                                print(f"🎯 [Debug] State 4 (Gaze Intersection): Gaze point : {np.round(current_pt, 3)}")
                            debug_print_time = time.time()

                        # 🎯 2. 凝视确认与“智能吸附”
                        if current_pt is not None:
                            if fixation_point is None:
                                fixation_point = current_pt
                                fixation_start_time = time.time()
                            else:
                                dist = np.linalg.norm(current_pt - fixation_point)

                                is_confirmed = False

                                if instant_select_flag:
                                    is_confirmed = True
                                    fixation_point = current_pt  # 使用捏合瞬间最精确的点
                                    instant_select_flag = False  # 用完即焚，防止无限连发
                                    print("\n⚡ [State 4] Pinch confirmation triggered! Bypassing dwell timer.")

                                if dist < FIXATION_TOLERANCE:
                                    dwell_time = time.time() - fixation_start_time
                                    
                                    # 盯住 2 秒了！开始吸附盒子！
                                    if dwell_time >= FIXATION_TIME_REQUIRED:
                                        is_confirmed = True
                                
                                else:
                                    fixation_point = current_pt
                                    fixation_start_time = time.time()

                                if is_confirmed:    
                                    if hasattr(scene_mapper, 'objects_pcd') and not scene_mapper.objects_pcd.is_empty():
                                        # 🧲 去【纯净盒子地图】里找离当前落点最近的点
                                        kdtree = o3d.geometry.KDTreeFlann(scene_mapper.objects_pcd)
                                        _, idx, sq_dist = kdtree.search_knn_vector_3d(fixation_point, 1)
                                        
                                        distance_to_box = np.sqrt(sq_dist[0])
                                        
                                        # 如果你看的地方方圆 20 厘米内根本没有盒子，说明你在看空桌子，放弃抓取
                                        if distance_to_box > 0.3:
                                            print(f"\n ⚠️ [State 4 (Gaze Intersection)]: No boxes found around gaze point, please look at a box! distance to box : {np.round(distance_to_box, 3)}")
                                            fixation_point = None
                                            fixation_start_time = time.time()
                                            continue
                                            
                                        # 成功吸附！获取盒子 ID（因为噪点被删了，这里必定是有效盒子）
                                        box_id = scene_mapper.object_labels[idx[0]]
                                        
                                        # =======================================================
                                        # 🔍【全面诊断补丁】打印所有盒子中心、视线落点及吸附过程
                                        # =======================================================
                                        print("\n" + "🔍" * 20)
                                        print(f"🎯 [State 4 (Gaze Intersection)]: Gaze point in world frame (robot base): {np.round(fixation_point, 3)}")
                                        print(f"🧲 [State 4 (Gaze Intersection)]: Distance to nearest point from KDTree: {np.round(distance_to_box * 100, 1)} cm")
                                        
                                        # 遍历当前场景里所有被分割出来的不同盒子，算出它们各自的中心点
                                        unique_labels = np.unique(scene_mapper.object_labels)
                                        print(f"📦 [State 4 (Gaze Intersection)]: Currently detected {len(unique_labels)} objects in the memory map:")
                                        
                                        for label in unique_labels:
                                            # 提取这个标签对应的所有点
                                            lbl_indices = np.where(scene_mapper.object_labels == label)[0]
                                            lbl_points = np.asarray(scene_mapper.objects_pcd.points)[lbl_indices]
                                            
                                            # 粗略算一下这个盒子的中心（这里用 mean 快速计算用于诊断）
                                            lbl_center = np.mean(lbl_points, axis=0)
                                            
                                            # 计算你的眼睛落点到这个盒子中心的距离
                                            dist_from_gaze = np.linalg.norm(fixation_point - lbl_center)
                                            
                                            # 如果这个标签刚好是被吸附的标签，加个醒目的五星标记
                                            flag = "⭐ [State 4 (Gaze Intersection)]: Target Object" if label == box_id else "  "
                                            print(f"   {flag} 物体ID {label} -> 几何中心: {np.round(lbl_center, 3)} | 离你视线落点距离: {np.round(dist_from_gaze * 100, 1)} cm")
                                        print("🔍" * 20 + "\n")
                                        # =======================================================

                                        # 📦 提取这整个盒子的所有点，计算几何中心
                                        box_indices = np.where(scene_mapper.object_labels == box_id)[0]
                                        box_points = np.asarray(scene_mapper.objects_pcd.points)[box_indices]

                                        # 1. 将 numpy 数组转为 Open3D 点云对象
                                        single_box_pcd = o3d.geometry.PointCloud()
                                        single_box_pcd.points = o3d.utility.Vector3dVector(box_points)

                                        # 2. 计算有向包围盒 (OBB)
                                        obb = single_box_pcd.get_oriented_bounding_box()

                                        # # =======================================================
                                        # # 🌟【修改 2】残影消除：把这个盒子从全局记忆地图中抹除！
                                        # # =======================================================
                                        # # 稍微放大一下 OBB (比如 1.2 倍)，确保边缘和底部的噪点也能被包裹进去
                                        # eraser_obb = o3d.geometry.OrientedBoundingBox(obb)
                                        # eraser_obb.scale(1.2, eraser_obb.center)
                                        
                                        # # 获取全局地图中，落在这个“橡皮擦盒子”里的所有点的索引
                                        # ghost_indices = eraser_obb.get_point_indices_within_bounding_box(scene_mapper.global_pcd.points)
                                        
                                        # # 使用 invert=True 进行反向提取，相当于把这些点“抠除”
                                        # scene_mapper.global_pcd = scene_mapper.global_pcd.select_by_index(ghost_indices, invert=True)
                                        
                                        # # 同步把这个物体从“磁铁”点云中剔除，防止接下来发生重复吸附
                                        # ghost_indices_obj = eraser_obb.get_point_indices_within_bounding_box(scene_mapper.objects_pcd.points)
                                        # scene_mapper.objects_pcd = scene_mapper.objects_pcd.select_by_index(ghost_indices_obj, invert=True)
                                        
                                        # print(f"🧹 [State 4 (Gaze Intersection)]: Successfully removed ghost points of box {box_id} from the global point cloud map!")
                                        # # =======================================================

                                        # 3. 获取真正的几何中心 (这比 np.mean 准得多！)
                                        box_center = obb.center

                                        # 4. 获取盒子的长宽高尺寸 (用于决定夹爪张开多大)
                                        box_size = obb.extent 
                                        print(f"📦 [State 4 (Gaze Intersection)]: Box dimensions: {np.round(box_size, 3)} meters")

                                        # 5. 获取盒子的旋转矩阵 (用于对齐夹爪姿态！)
                                        box_rotation = obb.R
                                        
                                        print(f"\n 🎉 [State 4 (Gaze Intersection)]: Target Box {box_id}")
                                        print(f" 🎯 Box Geometry Center: {np.round(box_center, 3)}")
                                        

                                    #    
                                    # =======================================================
                                    # 🪐【重构筛选】多目标优选 —— 提取前3名顺位黄金抓取姿态
                                    # =======================================================
                                    if hasattr(scene_mapper, 'ai_grasps') and scene_mapper.ai_grasps is not None and len(scene_mapper.ai_grasps) > 0:
                                        CANDIDATE_POOL_LIMIT = 200 
                                        candidates_scored = [] # 新增：用于存放所有合法姿态的临时列表
                                        
                                        # 1. 第一轮过滤与双标大评分
                                        for idx_g, grasp_matrix in enumerate(scene_mapper.ai_grasps[:CANDIDATE_POOL_LIMIT]):
                                            grasp_xyz = grasp_matrix[:3, 3]
                                            approach_vector = grasp_matrix[:3, 2] 
                                            
                                            # A. 计算探针距离
                                            PROJECTION_DEPTH = 0.15 
                                            projected_xyz = grasp_xyz + PROJECTION_DEPTH * approach_vector
                                            dist_to_center = np.linalg.norm(projected_xyz - box_center)
                                            if dist_to_center > 0.15: continue
                                                
                                            # B. 计算倾斜垂直度
                                            downward_vector = np.array([0.0, 0.0, -1.0])
                                            tilt_alignment = np.dot(approach_vector, downward_vector)
                                            tilt_alignment = np.clip(tilt_alignment, -1.0, 1.0)
                                            if tilt_alignment < 0.64: continue
                                                
                                            # C. 综合评分并塞入池中
                                            distance_score = 1.0 - (dist_to_center / 0.15)
                                            combined_score = 0.6 * distance_score + 0.4 * tilt_alignment
                                            
                                            candidates_scored.append((combined_score, grasp_matrix, dist_to_center, tilt_alignment, idx_g))

                                        # 2. 第二轮：按照得分从高到低强行排序，并斩取前 3 名
                                        candidates_scored.sort(key=lambda x: x[0], reverse=True)
                                        top_3_candidates = candidates_scored[:3]
                                        
                                        if len(top_3_candidates) > 0:

                                            eraser_obb = o3d.geometry.OrientedBoundingBox(obb)
                                            eraser_obb.scale(1.2, eraser_obb.center)
                                            
                                            # Disabled at Experiment 1------------------------------
                                            # 1. 删 Global Map
                                            ghost_indices = eraser_obb.get_point_indices_within_bounding_box(scene_mapper.global_pcd.points)
                                            scene_mapper.global_pcd = scene_mapper.global_pcd.select_by_index(ghost_indices, invert=True)
                                            
                                            # 2. 删 Object Map
                                            ghost_indices_obj = eraser_obb.get_point_indices_within_bounding_box(scene_mapper.objects_pcd.points)
                                            ghost_indices_np = np.asarray(ghost_indices_obj, dtype=np.int64)
                                            
                                            # 3. 🌟 绝对同步删 Labels (必须在更新 objects_pcd 之前做！)
                                            mask = np.ones(len(scene_mapper.object_labels), dtype=bool)
                                            mask[ghost_indices_np] = False
                                            scene_mapper.object_labels = scene_mapper.object_labels[mask]
                                            
                                            # 4. 最后更新 objects_pcd，保证 1:1 对齐
                                            scene_mapper.objects_pcd = scene_mapper.objects_pcd.select_by_index(ghost_indices_obj, invert=True)

                                            print(f"🧹 顺位筛选通过！已从记忆地图中抹除该物品的残影。")
                                            #------------------------------------------------------------

                                            scene_mapper.tuned_grasps_pool = [] # 🌟 新增：存放微调后的前3顺位矩阵池
                                            
                                            print(f"\n🎯 [多目标筛选] 针对当前目标成功锁定 {len(top_3_candidates)} 个顺位备份姿态:")
                                            
                                            # 3. 统一对前3名进行局部微调运算
                                            for rank, (score, selected_grasp, saved_dist, saved_tilt, selected_idx) in enumerate(top_3_candidates):
                                                # 注入 45 度自转
                                                Rz_45 = np.array([
                                                    [ 0.7071,  0.7071,  0.0,  0.0],
                                                    [-0.7071,  0.7071,  0.0,  0.0],
                                                    [ 0.0,     0.0,     1.0,  0.0],
                                                    [ 0.0,     0.0,     0.0,  1.0]
                                                ])
                                                tuned_grasp = selected_grasp @ Rz_45
                                                
                                                # 注入 7 厘米深插下探
                                                local_advance = np.eye(4)
                                                local_advance[2, 3] = 0.015 
                                                tuned_grasp = tuned_grasp @ local_advance
                                                
                                                # 180° 抄近道对称性优化
                                                ee_pos_temp, ee_quat_temp = robot_listener.get_current_pose()
                                                from scipy.spatial.transform import Rotation as Rot
                                                R_curr = Rot.from_quat(ee_quat_temp).as_matrix()
                                                R_target1 = tuned_grasp[:3, :3]
                                                Rz_180 = np.array([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]])
                                                R_target2 = R_target1 @ Rz_180
                                                if np.trace(R_curr.T @ R_target2) > np.trace(R_curr.T @ R_target1):
                                                    tuned_grasp[:3, :3] = R_target2

                                                # 存入顺位池
                                                scene_mapper.tuned_grasps_pool.append(tuned_grasp)
                                                
                                                tilt_angle_deg = np.degrees(np.arccos(saved_tilt))
                                                print(f"   👉 Rank #{rank+1} (Original Index: {selected_idx}) -> Perfect Score: {score*100:.1f} | Deviation: {saved_dist*100:.1f}cm | Tilt Angle: {tilt_angle_deg:.1f}°")
                                            
                                            # 4. 初始化默认策略：直接挂载顺位第1（索引0）的黄金姿态，重置重试计数器
                                            scene_mapper.selected_grasp = scene_mapper.tuned_grasps_pool[0]
                                            scene_mapper.grasp_retry_count = 0  # 🌟 新增：重试计数器归零
                                            
                                            scene_mapper.target_box_points = box_points
                                            
                                            # # =======================================================
                                            # # 🌟 新增：在确定坐标后，立刻计算悬停点并发送全息路线
                                            # # =======================================================
                                            # HOVER_BACK_DIST = 0.00
                                            # local_retreat = np.eye(4)
                                            # local_retreat[2, 3] = -HOVER_BACK_DIST
                                            
                                            # hover_matrix = scene_mapper.selected_grasp @ local_retreat
                                            # hover_x, hover_y, hover_z = hover_matrix[:3, 3]
                                            
                                            # curr_p, _ = robot_listener.get_current_pose()
                                            # if curr_p is not None and T_M is not None:
                                            #     visual_path = []
                                            #     num_visual_points = 2
                                            #     start_pt = np.array(curr_p)
                                            #     end_pt = np.array([hover_x, hover_y, hover_z])
                                                
                                            #     for i in range(num_visual_points + 1):
                                            #         ratio = i / float(num_visual_points)
                                            #         pt = start_pt + ratio * (end_pt - start_pt)
                                            #         # 若需抛物线可加上这一行：
                                            #         # pt[2] += 0.08 * 4.0 * ratio * (1.0 - ratio) 
                                            #         visual_path.append(pt)
                                                    
                                            #     send_path_to_hololens(conn, visual_path, T_M)
                                            #     print(f"✨ [HRI 可视化] 物品已确认！已向 HoloLens 发送飞往该物品的预测轨迹！")

                                            # # =======================================================
                                            # # 🛑 切入预演阶段，而不是直接去状态 5
                                            # # =======================================================
                                            # print("   ⏳ [预演模式] 保持全息路线 1.5 秒钟供用户观察...")
                                            # scene_mapper.gaze_step = 999    # 激活上方的阻断代码
                                            # hri_start_time = time.time()  # 重置计时器用于 1.5 秒倒数
                                            
                                            #实验2、3取消注释
                                            #current_hri_state = STATE_GRAB_OBJECT_YOLO 
                                            current_hri_state = STATE_GRAB_OBJECT
                                            hri_start_time = 0.0

                                            is_confirmed = False
                                            instant_select_flag = False
                                        
                                        else:
                                            # 🌟 如果没找到合适姿态，什么数组都不删，直接让他重试！
                                            print(f"⚠️ [Not Matched] 没有为该物体找到安全的抓取姿态，放弃抓取。请看向其他物体！")
                                            scene_mapper.selected_grasp = None
                                            fixation_point = None
                                            fixation_start_time = time.time()
                                        
                                    else:
                                        print(f"⚠️ [Not Matched] No valid grasp candidates found for the selected box {box_id}. Please try again.")
                                        scene_mapper.selected_grasp = None
                                
                        else:
                            fixation_point = None
                            fixation_start_time = 0.0

                # -----------------------------------
                # 状态 5：高精度 6-DoF AI 神经网络抓取 (Selected AI Grasp)
                # -----------------------------------
                elif current_hri_state == STATE_GRAB_OBJECT:

                    if hri_start_time == 0.0:
                        print("\n" + "="*50)
                        print("🦾 [HRI] State 5(Grab Object):Performing 【Along Local Z-Axis Oblique Insertion】High-Level Control Flow...")
                        
                        if hasattr(scene_mapper, 'selected_grasp') and scene_mapper.selected_grasp is not None:
                            # 拿到经过 45°自转、180°抄近道以及 4cm深插修正后的最终黄金抓取矩阵
                            grasp_matrix = scene_mapper.selected_grasp
                            
                            # 🌟 1. 【核心改变】利用矩阵右乘数学，在工具坐标系下反向后退 15 厘米，算出轴线悬停点
                            # 后退 15 厘米相当于在局部 Z 轴平移项注入 -0.15
                            HOVER_BACK_DIST = 0.15
                            local_retreat = np.eye(4)
                            local_retreat[2, 3] = -HOVER_BACK_DIST
                            
                            # 右乘得到悬停矩阵
                            hover_matrix = grasp_matrix @ local_retreat
                            
                            # 🌟 2. 提取悬停点的 3D 位置与四元数姿态
                            hover_x, hover_y, hover_z = hover_matrix[:3, 3]
                            from scipy.spatial.transform import Rotation as Rot
                            hover_quat = Rot.from_matrix(hover_matrix[:3, :3]).as_quat().tolist()
                            
                            # 挂载到全局变量
                            scene_mapper.hover_pose = [hover_x, hover_y, hover_z] + hover_quat
                            
                            # 记录我们要顺着轴线直插前进的总距离（悬停后退了多少，下探就要前进多少）
                            scene_mapper.local_stroke = HOVER_BACK_DIST
                            
                            print(f"   🚀 [State 5 (Grab Object)]: Claw has retreated {HOVER_BACK_DIST*100}cm along the oblique angle to establish a firing position")
                            print(f"      📍 Hover Point Position: X={hover_x:.3f}, Y={hover_y:.3f}, Z={hover_z:.3f}")
                        else:
                            print("🚨 [Error] No valid selected_grasp found!")
                            current_hri_state = STATE_GAZE_INTERSECTION
                            continue
                        
                        # 3. 立刻派发第一阶段：飞往斜上方悬停点
                        print("   -> 🛫 Step 1: Flying to the oblique axis hover point...")
                        if robot is not None:
                            robot.move_to(scene_mapper.hover_pose, speed=0.10) 
                            
                        scene_mapper.grasp_step = 1       
                        hri_start_time = time.time()
                        
                    else:
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True

                        # -----------------------------------
                        # 阶段 2：手腕完全转正对齐后，张开夹爪，沿局部 Z 轴斜向直插
                        # -----------------------------------
                        if getattr(scene_mapper, 'grasp_step', 0) == 1 and time.time() - hri_start_time > 1.0:
                            
                            # 二次硬对齐检查：手腕角度必须锁死到目标悬停姿态
                            target_quat = scene_mapper.hover_pose[3:7]
                            is_rotation_ok = robot.is_rotation_reached(target_quat, tolerance_deg=5.0)
                            
                            if is_robot_idle and is_rotation_ok:
                                if robot is not None:
                                    robot.open_gripper(width=0.08) # 张开夹爪
                                    
                                print(f"   -> 🛬 Step 2: [Pose Lock] Starting oblique insertion along the local Z-axis {scene_mapper.local_stroke*100:.1f}cm...")
                                if robot is not None:
                                    # 🌟 核心调用：顺着指尖方向向前开火推进！
                                    robot.move_along_local_z(stroke_distance=scene_mapper.local_stroke, speed=0.1) 
                                    
                                scene_mapper.grasp_step = 2
                                hri_start_time = time.time()
                                
                        # -----------------------------------
                        # 阶段 3：到达底部，闭合夹爪 
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 2 and time.time() - hri_start_time > 3.0 and is_robot_idle:
                            print("   -> ✊ Step 3: Contacting the target, closing the gripper...")
                            if robot is not None:
                                robot.close_gripper(force=25.0, speed=0.08) 
                            
                            scene_mapper.grasp_step = 3
                            hri_start_time = time.time()
                            

                        # -----------------------------------
                        # 阶段 4：抓稳后垂直向上提拉，退出障碍区 (你缺失的提拉部分)
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 3 and time.time() - hri_start_time > 0.5: 
                            print("   -> 🚀 Step 4: Lifting up object...")
                            
                            # 临时算一个世界坐标系正上方的提拉点
                            curr_p, curr_r = robot.get_current_pose()
                            if curr_p is not None:
                                scene_mapper.lift_pose = [curr_p[0], curr_p[1], curr_p[2] + 0.18] + curr_r.tolist()
                                if robot is not None:
                                    robot.move_to(scene_mapper.lift_pose, speed=0.08) 
                                    
                            scene_mapper.grasp_step = 4
                            hri_start_time = time.time()
                            
                        # -----------------------------------
                        # 阶段 5：提拉完成后，执行【成功率物理闭环检测】与顺位降级
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 4 and time.time() - hri_start_time > 1.0 and is_robot_idle:
                            
                            # 🌟 真实物理判定：读取夹爪宽度
                            current_width = 0.0
                            if robot is not None:
                                current_width = robot.get_gripper_width()
                            
                            # 如果夹爪间距大于 5 毫米，说明中间确实夹到了物体；
                            # 如果小于 5 毫米，说明抓空了（夹爪完全闭合碰拢了）
                            grasp_success = (current_width > 0.020) 
                            
                            if grasp_success:
                                print(f"\n🎉 [HRI] 物理闭环检测通过 (当前握持宽: {current_width*100:.1f}cm)，物品已成功抓稳！正在递送给用户...")
                                current_hri_state = STATE_LOOKING_FOR_USER
                                hri_start_time = 0.0
                            else:
                                # ==========================================
                                # 🚨 触发多级顺位降级重试防线
                                # ==========================================
                                current_retry = getattr(scene_mapper, 'grasp_retry_count', 0) + 1
                                scene_mapper.grasp_retry_count = current_retry
                                
                                if current_retry < len(scene_mapper.tuned_grasps_pool):
                                    print(f"\n⚠️ [HRI] 警报：顺位第 {current_retry} 次抓取判定失败（夹爪间距 {current_width*100:.1f}cm，疑似抓空或滑落）！")
                                    print(f"🔄 启动自动降级 -> 顺位延续至得分第 {current_retry + 1} 高的备份姿态...")
                                    
                                    # 1. 必须先松开夹爪，避免带着闭合的夹爪去对齐新姿态撞坏物品
                                    if robot is not None:
                                        robot.open_gripper(width=0.08)
                                        
                                    # 2. 从池子里捞出顺位延续的下一个黄金矩阵
                                    scene_mapper.selected_grasp = scene_mapper.tuned_grasps_pool[current_retry]
                                    
                                    # 3. 极其精妙的一步：重置内部步骤为 0，将计时器清零
                                    scene_mapper.grasp_step = 0
                                    hri_start_time = 0.0  # 🌟 强制状态机在下一帧原地复活
                                    
                                else:
                                    # 试满 3 个点（或把池子里搜到的点全试完）都失败了
                                    print(f"\n❌ [HRI] 终极警报：顺位前 {len(scene_mapper.tuned_grasps_pool)} 的黄金姿态全部尝试失败！触发一票否决。")
                                    print("🔄 放弃本次抓取任务。自动张开夹爪并退回眼动选择状态 (State 4)...")
                                    
                                    if robot is not None:
                                        robot.open_gripper(width=0.08)
                                        robot.move_to(READY_FOR_PASSING_POSE, speed=0.05) # 安全退回准备位
                                        
                                    # 清除策略残影，彻底退回
                                    scene_mapper.selected_grasp = None
                                    current_hri_state = STATE_GAZE_INTERSECTION
                                    hri_start_time = 0.0
# ... existing code ...
                        # -----------------------------------
                        # 任务完成
                        # -----------------------------------
                        elif getattr(scene_mapper, 'grasp_step', 0) == 4 and time.time() - hri_start_time > 1.0 and is_robot_idle:
                            print("\n🎉 [HRI] Grab success! Passing the object to user")
                            current_hri_state = STATE_LOOKING_FOR_USER
                            hri_start_time = 0.0
                
                # -----------------------------------
                # 状态 5：基于 YOLO 闭环视觉伺服的动态盲抓
                # -----------------------------------
                elif current_hri_state == STATE_GRAB_OBJECT_YOLO:

                    if hri_start_time == 0.0:
                        print("\n" + "="*50)
                        print("🦾 [HRI] State 5 (YOLO Vision Grasp): Switching to Absolute 3D Servoing!")
                        
                        # 1. 计算悬停安全点 (高度固定为桌面上方 30cm)
                        safe_hover_z = getattr(scene_mapper, 'table_z', 0.0) + 0.30 
                        
                        # 👇 填入你提供的夹爪垂直向下四元数
                        DOWNWARD_QUAT = [0.91, -0.42, 0.04, -0.02] 
                        
                        if fixation_point is None:
                            print("🚨 丢失目标方位，退回状态 4...")
                            current_hri_state = STATE_GAZE_INTERSECTION
                            continue
                            
                        # 合成前往物体正上方的粗略坐标
                        scene_mapper.yolo_hover_pose = [fixation_point[0], fixation_point[1], safe_hover_z] + DOWNWARD_QUAT
                        
                        # 飞往正上方安全点
                        if robot is not None:
                            robot.move_to(scene_mapper.yolo_hover_pose, speed=0.08)
                            
                        scene_mapper.yolo_step = 1
                        hri_start_time = time.time()

                    else:
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True
                        
                        # -----------------------------------------------------
                        # 阶段 1：等待到达上方观测点，启动 3D 平滑伺服
                        # -----------------------------------------------------
                        if getattr(scene_mapper, 'yolo_step', 0) == 1:
                            if is_robot_idle and (time.time() - hri_start_time > 1.0):
                                print("👀 [YOLO Grasp] Arrived at overhead safety point. Activating 3D Servoing...")
                                if robot is not None:
                                    robot.start_servoing() # 启动 3D 绝对伺服
                                scene_mapper.yolo_step = 2
                                hri_start_time = time.time()
                                
                        # -----------------------------------------------------
                        # 阶段 2：调用封装好的 YOLO 进行视觉对中与下探
                        # -----------------------------------------------------
                        elif getattr(scene_mapper, 'yolo_step', 0) == 2:
                            
                            # 🎯 1. 调用外部函数获取 YOLO 识别结果
                            current_color_image = BodyPointCloud_dual.global_raw_color_image 
                            detections = yolo_detector.detect(current_color_image) # 传入当前彩色帧
                            
                            bbox_center_x, bbox_center_y = None, None 
                            target_depth = None
                            
                            if len(detections) > 0:
                                best_target = detections[0]
                                bbox_center_x = best_target['x']
                                bbox_center_y = best_target['y']
                                target_angle_rad = best_target['angle'] # 🌟 拿到 2D 旋转角
                                
                                # 🌟 从 RealSense 硬件深度帧直接极速获取距离 (极其精准)
                                # 假设你在外面提取到了 depth_frame_1
                                if BodyPointCloud_dual.global_depth_frame is not None:
                                    target_depth = BodyPointCloud_dual.global_depth_frame.get_distance(bbox_center_x, bbox_center_y)
                                else:
                                    print("depthframe not found")
                            # 获取机械臂当前坐标
                            ee_pos, ee_quat = robot_listener.get_current_pose()
                            
                            if bbox_center_x is not None and bbox_center_y is not None and ee_pos is not None:
                                
                                # 2. 计算像素误差
                                cx, cy = 320, 240 # 请修改为你画面的真实中心点(如 w/2, h/2)
                                err_x_px = bbox_center_x - cx
                                err_y_px = bbox_center_y - cy
                                
                                # 3. 像素到物理位移的映射系数 (调参关键！)
                                M_PER_PIXEL = 0.0002 
                                
                                # 计算物理增量
                                delta_x = err_y_px * M_PER_PIXEL  
                                delta_y = err_x_px * M_PER_PIXEL  
                                delta_z = 0.0
                                
                                # 4. XY 对准后开始下探 (误差在 15 像素以内)
                                if abs(err_x_px) < 15 and abs(err_y_px) < 15:
                                    delta_z = -0.01  # 每次往下降 1cm
                                    
                                # 5. 合成绝对坐标目标点并发送给伺服
                                target_x = ee_pos[0] + delta_x
                                target_y = ee_pos[1] + delta_y
                                target_z = ee_pos[2] + delta_z
                                
                                from scipy.spatial.transform import Rotation as Rot
                                
                                # 1. 你原始的固定向下姿态
                                base_rot = Rot.from_quat([0.91, -0.42, 0.04, -0.02])
                                
                                # 2. 根据 YOLO 角度生成绕 Z 轴的自转矩阵
                                # 注意：这里的正负号可能需要根据你相机的物理安装方向做反转 (-target_angle_rad)
                                z_rotation = Rot.from_euler('z', target_angle_rad, degrees=False)
                                
                                # 3. 叠加旋转 (右乘，相当于在夹爪局部坐标系下自转)
                                final_rot = base_rot * z_rotation
                                final_quat = final_rot.as_quat().tolist()

                                # 4. 更新伺服目标 (如果你的 update_servo_target 支持传 7 维 Pose)
                                # 🚨 注意：你需要把底层的 update_servo_target 升级一下，让它不仅接收 XYZ，还能接收 四元数
                                target_pose_7d = [target_x, target_y, target_z] + final_quat

                                if robot is not None:
                                    robot.update_servo_target(target_pose_7d)
                                    
                                # 6. 触底检测：距离小于 12cm 时刹车！
                                if target_depth is not None and target_depth > 0.0 and target_depth < 0.12:
                                    print(f"⬇️ [YOLO Grasp] Depth trigger hit ({target_depth:.3f}m)! Stopping servoing...")
                                    if robot is not None:
                                        robot.stop_servoing()
                                        
                                    scene_mapper.yolo_step = 3
                                    hri_start_time = time.time()
                            else:
                                # 画面中丢失目标，悬停在原地
                                if robot is not None and ee_pos is not None:
                                    robot.update_servo_target(ee_pos)
                                    
                        # -----------------------------------------------------
                        # 阶段 3：闭合夹爪与提拉
                        # -----------------------------------------------------
                        elif getattr(scene_mapper, 'yolo_step', 0) == 3:
                            # 留 0.5 秒让机械臂把残余伺服动作停稳
                            if time.time() - hri_start_time > 0.5:
                                print("✊ [YOLO Grasp] Object reached! Closing gripper...")
                                if robot is not None:
                                    robot.close_gripper(force=15.0, speed=0.05)
                                    
                                    # 提拉动作
                                    curr_p, curr_r = robot_listener.get_current_pose()
                                    if curr_p is not None:
                                        lift_pose = [curr_p[0], curr_p[1], curr_p[2] + 0.20] + curr_r.tolist()
                                        robot.move_to(lift_pose, speed=0.10)
                                
                                print("🎉 [YOLO Grasp] Blind grasp complete! Moving to delivery state...")
                                current_hri_state = STATE_LOOKING_FOR_USER
                                hri_start_time = 0.0
                                scene_mapper.yolo_step = 0
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
                        print("🤖 [HRI] State 6(Waiting for User): Grabbed object successfully! Now looking for user's hand...")
                        
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
                                    print(f"🎯 [HRI] Detected hand entering workspace (Distance: {dist_to_base:.2f}m)! Initiating gripper flip to downward delivery pose...")
                                    
                                    # # 🌟 触发移动到夹爪向下的准备姿态
                                    # if robot is not None:
                                    #     robot.move_to(READY_FOR_PASSING_POSE, speed=0.08)
                                    
                                    scene_mapper.looking_step = 2 # 切换到阶段 2
                                    hri_start_time = time.time()  # 重置计时器
                                    
                            else:
                                # 如果一直没收到坐标，每 10 秒提醒一次
                                if time.time() - hri_start_time > 10.0:
                                    print("⚠️ [HRI] Please enable hand tracking in HoloLens and extend your hand towards the robot...")
                                    hri_start_time = time.time()
                                        
                        # -----------------------------------
                        # 阶段 2：等待机械臂翻转到递送准备姿势
                        # -----------------------------------
                        elif getattr(scene_mapper, 'looking_step', 1) == 2:
                            # 确保翻转动作已经走完，并且稍微给 1 秒冗余时间防抖
                            if is_robot_idle and (time.time() - hri_start_time > 1.0):
                                print("✅ [HRI] Has reached downward delivery pose, switching to dynamic tracking (State 7)!")
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
                            
                            # 🌟 新增：在开启伺服的瞬间，记录当前的空载(带物品)基准力
                            try:
                                scene_mapper.tracking_baseline_fz = robot.get_wrench()[2]
                                scene_mapper.tracking_baseline_fx = robot.get_wrench()[0]
                            except:
                                scene_mapper.tracking_baseline_fz = 0.0
                                scene_mapper.tracking_baseline_fx = 0.0

                        hri_start_time = time.time()
                        handover_dwell_start = 0.0
                        scene_mapper.pass_step = 0  
                        print("🚀 [HRI] State 7: Dynamically tracking user's hand...")

                    # =========================================================
                    # 阶段 0：3D 伺服追踪，直到在手掌上方稳定停留
                    # =========================================================
                    if getattr(scene_mapper, 'pass_step', 0) == 0:

                        # 🌟🌟🌟 在伺服追踪过程中，高频检测是否有用户“提前拿取/拉拽”
                        early_grab_triggered = False
                        if robot is not None and (hasattr(scene_mapper, 'tracking_baseline_fz') or hasattr(scene_mapper, 'tracking_baseline_fx')):
                            try:
                                current_fz = robot.get_wrench()[2]
                                delta_fz = abs(current_fz - scene_mapper.tracking_baseline_fz)
                                
                                current_fx = robot.get_wrench()[0]
                                delta_fx = abs(current_fz - scene_mapper.tracking_baseline_fx)

                                # 因为伺服运动伴随加减速惯性，阈值需稍大于匀速阶段 (比如 5.0N - 6.0N)
                                EARLY_FORCE_THRESHOLD_Z = 10 
                                EARLY_FORCE_THRESHOLD_X = 16
                                if delta_fz != 0:
                                    if delta_fz > EARLY_FORCE_THRESHOLD_Z or delta_fx > EARLY_FORCE_THRESHOLD_X :
                                        print(f"\n⚡ [HRI] Early interaction detected! (ΔFz={delta_fz:.2f}N)(ΔFx={delta_fx:.2f}N). User is pulling the object!")
                                        early_grab_triggered = True
                            except Exception:
                                pass
                                
                        # 🌟🌟🌟 如果用户提前抢夺，立刻执行“紧急松手”逻辑，并跳出当前帧
                        if early_grab_triggered:
                            if robot is not None:
                                robot.stop_servoing()          # 立刻刹车
                                robot.path_queue.queue.clear() # 清除后续路径
                                
                            print("🤝 [HRI] Successfully transferred (Early Release), opening gripper!")
                            time.sleep(0.2) 
                            
                            if robot is not None:
                                robot.open_gripper(width=0.08) 
                            
                            if T_M is not None:
                                send_path_to_hololens(conn, [], T_M)
                                
                            # 状态机重置，切回扫描状态
                            current_hri_state = STATE_SCAN_OBJECTS
                            hri_start_time = 0.0
                            handover_dwell_start = 0.0
                            global_holo_hand_pos = None
                            scene_mapper.pass_step = 0
                            if hasattr(scene_mapper, 'tracking_baseline_fz'): 
                                del scene_mapper.tracking_baseline_fz
                                
                            continue # 提前结束本帧循环，不再往下执行伺服运动指令！

                        if global_holo_hand_pos is not None:
                            
                            # 🌟 新增：工作空间边界安全保护 (安全电子围栏)
                            dist_to_base = np.linalg.norm(global_holo_hand_pos)
                            SAFE_RADIUS = 1.05 # 比 State 6 的触发半径(1.0m)稍大，形成迟滞区间，防止边缘反复横跳
                            
                            if dist_to_base > SAFE_RADIUS:
                                # 如果手超出了范围，且机械臂还在伺服，立刻刹停
                                if robot is not None and getattr(robot, 'is_servoing', False):
                                    print(f"⚠️ [HRI] Hand is outside the safe workspace ({dist_to_base:.2f}m > {SAFE_RADIUS}m)! Pausing tracking.")
                                    robot.stop_servoing()
                                    handover_dwell_start = 0.0 # 读秒清零
                                    if T_M is not None:
                                        send_path_to_hololens(conn, [], T_M) # 清空 HoloLens 连线
                                        
                            else:
                                # 如果之前被暂停了，现在手又乖乖回到安全范围内了，恢复追踪！
                                if robot is not None and not getattr(robot, 'is_servoing', False):
                                    print(f"✅ [HRI] Hand is back in the safe range ({dist_to_base:.2f}m), resuming servo tracking!")
                                    robot.start_servoing()
                                    
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
                                            print("⏳ [HRI] Robot is in position, please keep your hand stable for 1 second...")
                                    else:
                                        if dist_to_target > 0.10:
                                            print("⚠️ [HRI] Target moved significantly, interrupting stability timer, resuming tracking...")
                                            handover_dwell_start = 0.0
                                            
                                        elif time.time() - handover_dwell_start >= 0.5:
                                            print("⬇️ [HRI] Tracking stable! Stopping servoing and initiating gentle descent...")
                                            
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
                                    print(f"⚖️ [HRI] Motion is stable, recording baseline force: Fz = {current_fz:.2f}N")
                                
                                delta_fz = abs(current_fz - scene_mapper.baseline_fz)
                                
                                # 稍微提高一点点阈值增加鲁棒性 (3.5N)
                                FORCE_THRESHOLD = 3.5 
                                if delta_fz > FORCE_THRESHOLD:
                                    force_triggered = True
                                    print(f"🖐️ [HRI] Force detected (ΔFz={delta_fz:.2f}N)！User has stabilized!")
                                    
                            except Exception as e:
                                pass
                            
                        is_robot_idle = robot.path_queue.empty() if robot is not None else True
                        
                        # 情况 A：成功摸到了手
                        if force_triggered:
                            if robot is not None:
                                robot.path_queue.queue.clear() 
                                
                            print("🤝 [HRI] Successully transferred, releasing object!")
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
                                print("⚠️ [HRI] Delivery failed! Hand support not detected.")
                                print("🔄 [HRI] Cancelling release, moving back to safe height...")
                                
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

            # -------------------------------------------------
            # 【全场最后】统一收集本帧最终状态，并安全同步给 HoloLens
            # -------------------------------------------------
            if is_HRI_Demo:  # 只有开启 Demo 模式才收集发送
                if current_hri_state == STATE_SCAN_OBJECTS:
                    current_sub_state = getattr(scene_mapper, 'scan_post_process', 0)
                elif current_hri_state == STATE_GRAB_OBJECT:
                    current_sub_state = getattr(scene_mapper, 'grasp_step', 0)
                elif current_hri_state == STATE_LOOKING_FOR_USER:
                    current_sub_state = getattr(scene_mapper, 'looking_step', 0)
                elif current_hri_state == STATE_TRACKING_AND_PASS:
                    current_sub_state = getattr(scene_mapper, 'pass_step', 0)
                else:
                    current_sub_state = 0

                # 检查是否有任何改变，有则一并送出
                if (current_hri_state != last_synced_state) or (current_sub_state != last_synced_sub_state):
                    send_hri_status_packet(conn, current_hri_state, current_sub_state)
                    last_synced_state = current_hri_state
                    last_synced_sub_state = current_sub_state


    except Exception as e:
        print(f"[TCP] Server Error: {e}")
        traceback.print_exc()  # 🌟 加这一行！它会打印出到底是哪个文件的哪一行触发了 resize 报错

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