import threading
import socket
import struct
import numpy as np

class RobotPositionListener:
    def __init__(self, ip="0.0.0.0", port=5006):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        
        # 🌟 原有变量完全保留
        self.current_pos = None
        self.current_quat = None
        
        # 🌟 新增变量：用于存储完整的 7 关节全量数据
        self.joint_angles = [0.0] * 7
        self.all_joint_positions = [np.zeros(3) for _ in range(7)]
        self.all_joint_quats = [np.array([0.0, 0.0, 0.0, 1.0]) for _ in range(7)]
        
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                # 接收最大 224 字节
                data, addr = self.sock.recvfrom(224)
                
                # 🌟 新版数据长度：56个float = 224字节
                if len(data) == 224:
                    unpacked = struct.unpack("!56f", data)
                    
                    for i in range(7):
                        base = i * 8
                        self.joint_angles[i] = unpacked[base]
                        self.all_joint_positions[i] = np.array([unpacked[base+1], unpacked[base+2], unpacked[base+3]])
                        self.all_joint_quats[i] = np.array([unpacked[base+4], unpacked[base+5], unpacked[base+6], unpacked[base+7]])
                    
                    # 🌟 兼容老逻辑：将第 7 个关节赋给原有的变量，确保老代码完全兼容
                    self.current_pos = self.all_joint_positions[6]
                    self.current_quat = self.all_joint_quats[6]

                # 🌟 兼容你老版的代码数据长度：7个float = 28字节
                elif len(data) == 28:
                    x, y, z, qx, qy, qz, qw = struct.unpack("!fffffff", data)
                    self.current_pos = np.array([x, y, z])
                    self.current_quat = np.array([qx, qy, qz, qw])

            except Exception as e:
                print(f"UDP Error: {e}")

    # ==========================================
    # 🌟 原有方法：名字、参数、返回值完全不动！
    # ==========================================
    def get_position(self):
        return self.current_pos

    def get_current_pose(self):
        return self.current_pos, self.current_quat

    # ==========================================
    # 🌟 新增方法：供你需要拿全量数据时调用
    # ==========================================
    def get_joints(self):
        return self.joint_angles

    def get_all_joint_positions(self):
        return self.all_joint_positions

    def get_all_joint_quats(self):
        return self.all_joint_quats