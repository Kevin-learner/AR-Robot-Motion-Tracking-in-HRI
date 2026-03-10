import cv2
import socket
import struct
import time

# 配置 B 电脑（机械臂主机）的 IP
ROBOT_PC_IP = '127.0.0.1' #'100.93.142.100'  #请替换为实际的 B 电脑 IP 地址
VIDEO_PORT = 8849

def start_pusher():
    # 0 对应默认摄像头
    cap = cv2.VideoCapture(0)
    # 优化：设置摄像头缓存，减少延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_sock.connect((ROBOT_PC_IP, VIDEO_PORT))
        print(f"✅ 已连接到中转主机 {ROBOT_PC_IP}")

        while True:
            ret, frame = cap.read()
            if not ret: continue

            # 优化：降低分辨率和压缩质量（这是降延迟最快的方法）
            # frame = cv2.resize(frame, (320, 180)) 
            _, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            data = img_encode.tobytes()
            
            # 发送：[4b长度] + [数据]
            try:
                client_sock.sendall(struct.pack(">I", len(data)) + data)
            except:
                print("❌ 失去与主机的连接")
                break
            
            # 限制发送频率，避免填满缓冲区
            time.sleep(0.03) 
            
    finally:
        cap.release()
        client_sock.close()

if __name__ == "__main__":
    start_pusher()