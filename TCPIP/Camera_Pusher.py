import cv2
import socket
import struct
import time

# B 电脑（机械臂主机）的 IP 和 专门接收视频的端口
ROBOT_PC_IP = '100.93.142.100' 
VIDEO_PORT = 8849

def start_pusher():
    cap = cv2.VideoCapture(0)
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_sock.connect((ROBOT_PC_IP, VIDEO_PORT))
        print(f"✅ 已连接到机械臂主机 {ROBOT_PC_IP}")

        while True:
            ret, frame = cap.read()
            if not ret: continue

            # 压缩图像 (建议 320x240 以降低中转延迟)
            frame = cv2.resize(frame, (480, 270)) 
            _, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            data = img_encode.tobytes()
            
            # 发送长度 + 数据
            client_sock.sendall(struct.pack(">I", len(data)) + data)
            
            # 保持 20-30 FPS
            time.sleep(0.04)
            
    except Exception as e:
        print(f"❌ 转发中断: {e}")
    finally:
        cap.release()
        client_sock.close()

if __name__ == "__main__":
    start_pusher()