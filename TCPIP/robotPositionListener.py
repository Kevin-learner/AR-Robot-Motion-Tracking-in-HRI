import threading
import socket
import struct
import numpy as np

class RobotPositionListener:
    def __init__(self, ip="0.0.0.0", port=5006):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        
        self.current_pos = None
        self.current_quat = None  # 🌟 新增：存储姿态
        
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                # 🌟 修改：接收 7 个 float (28 字节)
                data, addr = self.sock.recvfrom(28)
                if len(data) == 28:
                    x, y, z, qx, qy, qz, qw = struct.unpack("!fffffff", data)
                    self.current_pos = np.array([x, y, z])
                    self.current_quat = np.array([qx, qy, qz, qw]) # 🌟 保存四元数
            except Exception as e:
                print(f"UDP Error: {e}")

    # 保留原方法兼容老代码
    def get_position(self):
        return self.current_pos

    # 🌟 新增方法：同时获取位置和姿态
    def get_current_pose(self):
        return self.current_pos, self.current_quat