import socket
import struct
import numpy as np
import yaml
import os
import sys
import time
import cv2

# -------------------------------------------------
# 1. 基础配置与加载
# -------------------------------------------------
def load_config(path="config.yaml"):
    default_config = {
        'tcp': {'host': '0.0.0.0', 'port': 8848},
        'alignment': {'aruco_path': 'Realsense_Aruco.txt'},
        'robot': {'position_file': 'robotPosition.txt'},
        'recording': {'output_file': 'calibration_data_recorded.txt'} # 默认保存路径
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
RECORD_FILE = config['recording']['output_file'] # [新增] 保存路径

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
    import pathInterpolation
except ImportError:
    print("⚠️ Warning: pathInterpolation.py not found. No way to generate paths.")
    pathInterpolation = None

try:
    from videoSender import VideoSender
except ImportError:
    print("⚠️ Warning: videoSender.py not found. No way to send video.")
    VideoSender = None
# -------------------------------------------------
# 2. 核心与辅助函数
# -------------------------------------------------
def recv_exact(conn, size):
    """精确读取指定长度的字节"""
    buffer = b''
    while len(buffer) < size:
        try:
            chunk = conn.recv(size - len(buffer))
            if not chunk: return None
            buffer += chunk
        except Exception:
            return None
    return buffer

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
        robot_raw_pos[2] = -robot_raw_pos[2]
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

# -------------------------------------------------
# 3. 主循环
# -------------------------------------------------
def main():
    #Initialize the video sender
    sender = VideoSender(port=8849)

    #Initialize the robot controller
    robot = None

    #Initialize the robot listener
    robot_listener = robotPositionListener.RobotPositionListener(port=5006)

    sSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    T_M = None 

    # [新增] 获取缓存文件路径
    TM_CACHE_PATH = config['alignment'].get('tm_cache_file', 'tm_matrix.npy')
    
    # [新增] 启动时尝试自动加载
    T_M = load_tm_matrix(TM_CACHE_PATH) 
    if T_M is not None:
        print("🚀 [System] Ready to go without manual calibration.")
    else:
        print("ℹ️ [System] No calibration cache found. Calibration required.")

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
            
            if header in ['d', 'r', 'b', 'm', 'p', 'v', 'x']:
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
                        
                        # 读取当前机器人坐标
                        robot_pos = robot_listener.get_position()
                        
                        if robot_pos is not None:
                            # 保存到文件
                            save_recorded_point(RECORD_FILE, point_index, robot_pos)
                        else:
                            print("   ❌ Failed to read robot position for recording.")
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
                elif header == 'm':
                    data = recv_exact(conn, 12) # 再读 12 个字节 (3个float)
                    if len(data) == 12:
                        # '<fff' 表示：小端序 (Little Endian), 3个 float
                        ux, uy, uz = struct.unpack('<fff', data)
                        print(f"Unity target received: X={ux}, Y={uy}, Z={uz}")

                        if T_M is not None:
                            # === 调用转换函数 ===
                            target_robot_pos = rut.unity2robot_transform((ux, uy, uz), T_M)
                            
                            if target_robot_pos is not None:
                                print(f"target in robot coordinate frame{target_robot_pos}")
                                
                                robot.move_to(target_robot_pos, speed=0.02)
                                
                            else:
                                print("   ❌ 转换失败")
                        else:
                            print("   ⚠️ T_M 尚未计算，无法转换坐标。请先进行校准 (Header 'd')。")

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
                            r_pos = rut.unity2robot_transform(u_pos, T_M)
                            
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
                                resolution=30
                            )
                            
                            # 5. 执行机器人运动
                            if robot is None: 
                                robot = RobotController()
                            
                            print(f"   🚀 开始执行，总插值点数: {len(final_smooth_path)}")
                            # 发送给机械臂执行
                            robot.execute_path(final_smooth_path, speed=0.02)
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

            # ===============================================
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
    main()