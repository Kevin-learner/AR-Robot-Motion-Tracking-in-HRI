import threading
import socket
import struct
import numpy as np
import copy # 引入深拷贝，用于安全传递数据

class RobotPositionListener:
    def __init__(self, ip="0.0.0.0", port=5006):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        
        # 原有变量完全保留
        self.current_pos = None
        self.current_quat = None
        
        # 新增变量：用于存储完整的 8 关节全量数据
        self.joint_angles = [0.0] * 8
        self.all_joint_positions = [np.zeros(3) for _ in range(8)]
        self.all_joint_quats = [np.array([0.0, 0.0, 0.0, 1.0]) for _ in range(8)]
        
        # 🌟 增加线程锁，防止数据撕裂
        self.lock = threading.Lock()
        
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                # 🌟 缓冲区给到 1024，防止意外截断
                data, addr = self.sock.recvfrom(1024)
                
                # 🌟 新版数据长度：8个关节 * 8个float = 64个float = 256字节
                if len(data) == 256:
                    unpacked = struct.unpack("!64f", data)
                    
                    # 🌟 获取锁，开始更新数据
                    with self.lock:
                        for i in range(8):
                            base = i * 8
                            self.joint_angles[i] = unpacked[base]
                            self.all_joint_positions[i] = np.array([unpacked[base+1], unpacked[base+2], unpacked[base+3]])
                            self.all_joint_quats[i] = np.array([unpacked[base+4], unpacked[base+5], unpacked[base+6], unpacked[base+7]])
                        
                        # 兼容老逻辑：将第 8 个关节(法兰盘 Link8) 赋给原有的变量
                        self.current_pos = self.all_joint_positions[7]
                        self.current_quat = self.all_joint_quats[7]

                # 兼容老版数据长度：7个float = 28字节
                elif len(data) == 28:
                    x, y, z, qx, qy, qz, qw = struct.unpack("!fffffff", data)
                    with self.lock:
                        self.current_pos = np.array([x, y, z])
                        self.current_quat = np.array([qx, qy, qz, qw])

            except Exception as e:
                print(f"UDP Error: {e}")

    # ==========================================
    # 🌟 对外暴露的方法：全部加锁并返回拷贝
    # ==========================================
    def get_position(self):
        with self.lock:
            return copy.deepcopy(self.current_pos)

    def get_current_pose(self):
        with self.lock:
            return copy.deepcopy(self.current_pos), copy.deepcopy(self.current_quat)

    def get_joints(self):
        with self.lock:
            return copy.deepcopy(self.joint_angles)

    def get_all_joint_positions(self):
        with self.lock:
            return copy.deepcopy(self.all_joint_positions)

    def get_all_joint_quats(self):
        with self.lock:
            return copy.deepcopy(self.all_joint_quats)