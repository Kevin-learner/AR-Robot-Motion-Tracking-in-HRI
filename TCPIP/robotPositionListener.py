import threading
import socket
import struct
import numpy as np

class RobotPositionListener:
    def __init__(self, ip="0.0.0.0", port=5006):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.current_pos = None
        self.running = True
        # Start the background thread
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()

    def _listen(self):
        while self.running:
            try:
                # Program 1 sends 3 floats (12 bytes) using "!fff"
                data, addr = self.sock.recvfrom(12)
                x, y, z = struct.unpack("!fff", data)
                self.current_pos = np.array([x, y, z])
            except Exception as e:
                print(f"UDP Error: {e}")

    def get_position(self):
        return self.current_pos