import socket
import struct
import cv2
import time

class VideoSender:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.is_streaming = False

    def send_frame(self, conn, sensor_type='c'):
        """
        核心发送函数：抓取、压缩、打包、发送
        """
        if not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        try:
            # 1. 图像压缩 (640x480, 质量60)
            frame = cv2.resize(frame, (320, 240))
            result, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not result:
                return False
            
            img_data = img_encode.tobytes()
            img_len = len(img_data)

            # 2. 构造符合你 Unity 协议的报文
            # [1b 'v'] + [1b sensor_type] + [4b img_len] + [N bytes data]
            header = b'v'
            s_type = sensor_type.encode('utf-8')
            length_pack = struct.pack('>i', img_len)        # 4字节长度 (大端)

            # 3. 合并发送
            packet = header + s_type + length_pack + img_data
            conn.sendall(packet)
            print(f"Sent frame, size: {img_len}")
            return True
        except Exception as e:
            print(f"发送过程中出错: {e}")
            return False

    def release(self):
        self.cap.release()