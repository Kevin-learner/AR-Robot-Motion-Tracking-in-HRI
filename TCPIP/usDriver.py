import sys
import ctypes
from ctypes import cdll
import numpy as np
import time
import keyboard
import os
from PIL import Image
import cv2
import socket
import struct
# ================== 全局参数 ==================
W = 500
H = 500
SAVE_DIR = 'ultrasound_frames'  # 保存目录

TARGET_FPS = 15          # 每秒 2 张
SAVE_EVERY = 1          # 每隔多少帧保存一次；=1 表示每帧都保存
SAVE_FORMAT = 'jpg'     # 'bmp'（无压缩最快），可选 'jpg'/'png'
PRINT_EVERY = 10        # 每多少帧打印一次进度

# =============== TELEMED 初始化相关 ===============
def init_ultrasound(usgfw2):
    usgfw2.on_init()

    ERR = usgfw2.init_ultrasound_usgfw2()
    if ERR == 2:
        print('Main Usgfw2 library object not created')
        usgfw2.Close_and_release()
        sys.exit(1)

    ERR = usgfw2.find_connected_probe()
    if ERR != 101:
        print('Probe not detected')
        usgfw2.Close_and_release()
        sys.exit(1)

    ERR = usgfw2.data_view_function()
    if ERR < 0:
        print('Main ultrasound scanning object for selected probe not created')
        sys.exit(1)

    ERR = usgfw2.mixer_control_function(0, 0, W, H, 0, 0, 0)
    if ERR < 0:
        print('B mixer control not returned')
        sys.exit(1)

def get_resolution(usgfw2):
    res_X = ctypes.c_float(0.0)
    res_Y = ctypes.c_float(0.0)
    usgfw2.get_resolution(ctypes.pointer(res_X), ctypes.pointer(res_Y))
    return res_X.value, res_Y.value

# =============== 数据转换 & 保存 ===============
def normalize_and_reshape(p_array):
    """
    从 ctypes 缓冲区生成 numpy 视图（零拷贝），
    并重塑为 (H, W, 4)，截取前 3 通道 (RGB)。
    说明：沿用你原始的 uint32 * (W*H*4) 的内存布局假设。
    """
    arr = np.ctypeslib.as_array(p_array)  # 视图，不复制
    arr = arr.reshape(H, W, 4)[..., :3]
    # 设备若返回 32 位值，做一次截断到 0~255
    return (arr & 0xFF).astype(np.uint8)

def save_frame(frame, iteration):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if SAVE_FORMAT.lower() == 'png':
        Image.fromarray(frame, 'RGB').save(
            f"{SAVE_DIR}/frame_{iteration:04d}.png", optimize=False, compress_level=0
        )
    elif SAVE_FORMAT.lower() == 'jpg':
        Image.fromarray(frame, 'RGB').save(
            f"{SAVE_DIR}/frame_{iteration:04d}.jpg", quality=90
        )
    else:  # bmp
        Image.fromarray(frame, 'RGB').save(
            f"{SAVE_DIR}/frame_{iteration:04d}.bmp"
        )

# =============== 主循环（只保存，不显示） ===============
def main_loop(usgfw2):
    iteration = 0

    # 注意：沿用你原来的分配方式（uint32 * (W*H*4)）
    p_array = (ctypes.c_uint32 * (W * H * 4))()

    frame_period = 1.0 / max(1, TARGET_FPS)  # 0.5 s 对应 2 FPS
    next_t = time.monotonic()

    print(f"[INFO] 保存到: {os.path.abspath(SAVE_DIR)}")
    print(f"[INFO] 速率: ~{TARGET_FPS:.1f} FPS（每 {frame_period:.3f}s 一张）")
    print("[INFO] 按 'q' 退出。")

    while True:
        iteration += 1

        # 取帧
        usgfw2.return_pixel_values(ctypes.pointer(p_array))
        frame_rgb = normalize_and_reshape(p_array)

        frame_rgb = cv2.flip(frame_rgb, 0)  # 根据需要翻转

        # # 保存：按 SAVE_EVERY 控制频率
        # if iteration % SAVE_EVERY == 0:
        #     save_frame(frame_rgb, iteration)

        # # 进度打印
        # if iteration % PRINT_EVERY == 0:
        #     print(f"[INFO] 已保存帧: {iteration}")

        # 2. 将 RGB 转换为 BGR (因为 OpenCV 使用 BGR 格式)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 3. 实时显示图像
        cv2.imshow('Telemed Ultrasound Real-time', frame_bgr)

        # 4. 退出逻辑 (cv2.waitKey 是必须的，否则窗口会卡死)
        # 注意：这里不仅能检测 'q'，也是刷新窗口画面的核心
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 退出显示并清理资源...")
            break

        # 退出
        if keyboard.is_pressed('q'):
            print("[INFO] 接收到退出指令，正在清理资源…")
            break

        # 限速到 TARGET_FPS
        next_t += frame_period
        sleep_left = next_t - time.monotonic()
        if sleep_left > 0:
            time.sleep(sleep_left)
        else:
            next_t = time.monotonic()

    # 清理
    usgfw2.Freeze_ultrasound_scanning()
    usgfw2.Stop_ultrasound_scanning()
    usgfw2.Close_and_release()

# =============== 入口 ===============
def main():
    # 修改为你的 DLL 路径
    root = r'D:\MT\Python\PythonProject\TCPIP\TCPIP\\'
    usgfw2 = cdll.LoadLibrary(os.path.join(root, 'usgfw2wrapper.dll'))

    init_ultrasound(usgfw2)

    # 可选：如果想打印一次分辨率
    try:
        rx, ry = get_resolution(usgfw2)
        print(f"[INFO] 当前分辨率: dX={rx:.3f} mm, dY={ry:.3f} mm")
    except Exception:
        pass

    #main_loop(usgfw2)
    main_loop_pusher(ip='127.0.0.1', port=8849, usgfw2=usgfw2)
    del usgfw2

def main_loop_pusher(ip = '127.0.0.1' ,port = 8849, usgfw2 = None):
    # 这里可以实现一个简单的 TCP 服务器，等待连接并发送帧数据
    # ================== 1. 初始化 Socket ==================
    ROBOT_PC_IP = ip  # 请替换为实际的推流接收端 IP 地址 
    VIDEO_PORT = port
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_sock.connect((ROBOT_PC_IP, VIDEO_PORT))
        print(f"✅ 已连接到推流接收端 {ROBOT_PC_IP}")
    except Exception as e:
        print(f"❌ 无法连接推流服务器: {e}")
        return

    p_array = (ctypes.c_uint32 * (W * H * 4))()
    print("[INFO] 正在抓取超声画面并推送...")

    while True:
        # ================== 2. 获取 SDK 图像 ==================
        usgfw2.return_pixel_values(ctypes.pointer(p_array))
        frame_rgb = normalize_and_reshape(p_array) # 这里的 H, W 需要与你的 SDK 设置一致

        # ================== 3. 图像预处理 (为推流优化) ==================
        # 转换为 BGR (OpenCV 格式)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        frame_bgr = cv2.flip(frame_bgr, 0)  # 根据需要翻转
        # 实时显示（可选，方便调试）
        cv2.imshow('Local Preview', frame_bgr)
        
        send_frame = frame_bgr
        # 调整分辨率以降低传输延迟 (320x180 或保持原样)
        #send_frame = cv2.resize(frame_bgr) 
        
        # ================== 4. 编码与发送 ==================
        _, img_encode = cv2.imencode('.jpg', send_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        data = img_encode.tobytes()
        
        try:
            # 发送：[4字节长度] + [数据]
            client_sock.sendall(struct.pack(">I", len(data)) + data)
        except:
            print("❌ 发送失败，断开连接")
            break

        # ================== 5. 退出与频率控制 ==================
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 也可以根据需要保留你的 time.sleep(0.03)

    # 清理
    cv2.destroyAllWindows()
    client_sock.close()
    usgfw2.Freeze_ultrasound_scanning()
    usgfw2.Stop_ultrasound_scanning()
    usgfw2.Close_and_release()

if __name__ == "__main__":
    main()

