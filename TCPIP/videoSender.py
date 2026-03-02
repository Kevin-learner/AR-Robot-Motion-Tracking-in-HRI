import socket
import struct
import threading

class VideoSender:
    def __init__(self, port=8849):
        self.latest_frame_data = None
        self.is_streaming = False
        # 开启后台线程，专门接收 A 电脑发来的图片数据
        threading.Thread(target=self._receive_loop, args=(port,), daemon=True).start()

    def _receive_loop(self, port):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('0.0.0.0', port))
        server_sock.listen(1)
        
        while True:
            conn_from_a, addr = server_sock.accept()
            conn_from_a.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while True:
                    # 读取 A 发来的 4 字节长度
                    len_bytes = self._recv_exact(conn_from_a, 4)
                    if not len_bytes: break
                    img_len = struct.unpack(">I", len_bytes)[0]
                    
                    # 读取图片二进制
                    img_data = self._recv_exact(conn_from_a, img_len)
                    if not img_data: break
                    
                    # 覆盖更新最新帧
                    self.latest_frame_data = img_data
            except:
                break
            finally:
                conn_from_a.close()

    def _recv_exact(self, conn, count):
        buf = b''
        while count:
            newbuf = conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def send_frame(self, conn, sensor_type='c'):
        """ 这个函数被你的主循环调用，将最新的图转发给 HoloLens """
        if not self.is_streaming or self.latest_frame_data is None:
            return False

        # 取出当前最新的图
        data_to_send = self.latest_frame_data
        # 重要：取完之后清空，防止网络慢时重复发送同一张旧图
        self.latest_frame_data = None 

        try:
            # 协议：[v] + [sensor_type] + [4b len] + [data]
            header = b'v' + sensor_type.encode('utf-8') + struct.pack(">i", len(data_to_send))
            conn.sendall(header + data_to_send)
            return True
        except:
            return False

    def release(self):
        pass