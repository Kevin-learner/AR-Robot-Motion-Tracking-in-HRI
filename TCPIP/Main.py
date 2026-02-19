import socket
import os
import sys
import time
import struct
import numpy as np
import threading
from BodyPointCloud_dual import Body3DSkeletonProcess_dual
from BodyPointCloud_dual import BodyPointCloudProcess_dual
from BodyPointCloud_dual import main_pointcloud_visualizer_loop
from compute_alignment import align_with_realsense
from BodyPointCloud_dual import main_3DSkeleton_visualizer_loop
import queue
import cv2
import csv
import os
import datetime
import yaml
from pynput import keyboard as kb
#Basic onfiguration
def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
config = load_config()
serverHost = config['tcp']['host']
serverPort = config['tcp']['port']
T_M = np.array(config['skeleton']['T_M'])
use_dual_camera = config['skeleton']['use_dual_camera']
pointcloud_max_points = config['pointcloud']['max_points']
pointcloud_ero_para = config['pointcloud']['ero_para']
frame_count_global = 0
sending_thread = None
sending_stop_flag = threading.Event()
save_skeleton_event = threading.Event()
system_start_time = time.time()
pointcloud_visualize_thread = None
csv_writer = None
csv_file = None
csv_lock = threading.Lock()
last_write_time = None
#Processing

# Initializes and starts / stopssaving skeleton data to a timestamped CSV file.
def start_saving():
    global save_skeleton_event, csv_writer, csv_file
    global next_write_time, frame_count_global
    frame_count_global = 1
    next_write_time = time.time() + 0.001
    with csv_lock:
        if save_skeleton_event.is_set():
            print("Already saving, ignoring duplicate 'b'")
            return

        save_dir = config['saving']['output_dir']
        os.makedirs(save_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_file_path = os.path.join(save_dir, f"skeleton_data_{timestamp_str}.csv")
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)

        header = ['frame'] + [f'joint{i}_{axis}' for i in range(17) for axis in ['x', 'y', 'z']]
        csv_writer.writerow(header)

        save_skeleton_event.set()
        print(f"[LOCAL] Saving started: {csv_file_path}")
def stop_saving():
    global save_skeleton_event, csv_writer, csv_file
    with csv_lock:
        if not save_skeleton_event.is_set():
            print("Not currently saving, ignoring 'e'")
            return

        save_skeleton_event.clear()
        if csv_file:
            csv_file.close()
            print("[LOCAL] Saving stopped and file closed.")
        csv_writer = None
        csv_file = None
# Continuously captures and sends 3D skeleton data over a connection. If saving is enabled, writes skeleton frames to a CSV file at fixed intervals.
def skeleton_sending_loop(conn):
    global T_M, frame_count_global, next_write_time

    frame_count = 0
    fps_timer = time.time()
    write_interval = 0.1

    while not sending_stop_flag.is_set():
        start_time = time.time()
        try:
            timestamp_capture = time.time() - system_start_time
            send_list, should_quit = Body3DSkeletonProcess_dual(T_M, use_dual_camera=use_dual_camera)
            timestamp_send = time.time() - system_start_time

            if send_list is None or len(send_list) != 59:
                print(f"Invalid skeleton (length = {len(send_list) if send_list else 'None'}), skipping.")
                continue

            # Write data at fixed intervals
            current_time = time.time()
            if save_skeleton_event.is_set() and csv_writer and current_time >= next_write_time:
                with csv_lock:
                    row = [frame_count_global] + [coord for joint in send_list[:17] for coord in joint]
                    csv_writer.writerow(row)
                    interval = current_time - (next_write_time - write_interval)
                    print(f"Fixed Write at {current_time:.3f} s (interval = {interval:.3f} s)")
                next_write_time += write_interval

            # Sending
            send_array = np.array(send_list, dtype=np.float32).flatten()
            packet = b's' + struct.pack('<' + 'f' * len(send_array), *send_array)
            packet += struct.pack('<f', np.float32(timestamp_capture))
            packet += struct.pack('<f', np.float32(timestamp_send))
            conn.sendall(packet)

            frame_count_global += 1
            if time.time() - fps_timer >= 1.0:
                frame_count = 0
                fps_timer = time.time()

        except Exception as e:
            print("Error during skeleton sending:", e)
            break

        elapsed = time.time() - start_time
        time.sleep(max(0.001, 0.033 - elapsed))
# Continuously processes and sends 3D point cloud data over a socket connection.
def pointcloud_sending_loop(conn):
    global T_M
    frame_count = 0
    fps_timer = time.time()
    # max_points = 8000
    # ero_para = 1
    while not sending_stop_flag.is_set():
        start_time = time.time()
        try:
            timestamp_capture = time.time() - system_start_time
            #send_coords, should_quit = BodyPointCloudProcess_dual(T_M, max_points=max_points,use_dual_camera=False,ero_para=ero_para)
            send_coords, should_quit = BodyPointCloudProcess_dual(
                T_M,
                max_points=pointcloud_max_points,
                use_dual_camera=use_dual_camera,
                ero_para=pointcloud_ero_para
            )
            timestamp_send = time.time() - system_start_time
            print(f"Total Delay:       {(timestamp_send - timestamp_capture) * 1000:.3f} ms")
            if should_quit:
                print("Quit signal received. Sending empty pointcloud instead.")
                send_coords = []

            coords_mm = (np.array(send_coords) * 1000).astype(np.int16)
            num_points = len(coords_mm)
            # print(f"Preview Points (first 5): {send_coords[:5]}")
            packet = b'p' + struct.pack('<I', num_points) + coords_mm.tobytes()
            conn.sendall(packet)

            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                #print(f"PointCloud FPS: {frame_count}")
                frame_count = 0
                fps_timer = time.time()
        except Exception as e:
            print("Error sending pointcloud:", e)
            break
        elapsed = time.time() - start_time
        time.sleep(max(0.033 - elapsed, 0))
#  TCP server that listens for control commands to start/stop sending skeleton or point cloud data.
def tcp_server_trigger_push(queue_for_main_visualizer):
    global T_M, sending_thread, sending_stop_flag, visualize_thread, pointcloud_visualize_thread

    # serverHost = '145.126.94.17'#'145.126.90.167'#'145.126.95.164'#'145.126.92.191'
    # serverPort = 8888

    sSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sSock.bind((serverHost, serverPort))
        print(f'[TCP] Listening on port {serverPort}...')
    except socket.error as msg:
        print(f'Bind failed. Error Code : {msg[0]} Message {msg[1]}')
        return

    sSock.listen(1)
    conn, addr = sSock.accept()
    print(f'[TCP] Connected with {addr[0]}:{addr[1]}')

    try:
        while True:
            print("[TCP] Waiting for message...")
            data = conn.recv(4096)
            if not data:
                print("[TCP] Empty data received, breaking.")
                break

            try:
                header = data[0:1].decode('utf-8')
            except Exception as e:
                print(f"Header decoding failed: {e}")
                continue

            print(f"[TCP] Header: {header}")

            if header == 'x':
                print("[TCP] Stop signal received.")
                sending_stop_flag.set()
                if sending_thread and sending_thread.is_alive():
                    sending_thread.join()
                sending_stop_flag.clear()
                continue

            if header == 'd':
                num_points = struct.unpack("<i", data[1:5])[0]
                float_data = np.frombuffer(data[5:5 + num_points * 12], dtype='<f4')
                points3d = float_data.reshape((num_points, 3))
                if np.all(points3d == 0.0):
                    continue
                if num_points == 12:
                    aruco_path = config['alignment']['aruco_path']
                    T_M = align_with_realsense(points3d, aruco_path)
                    print("Updated transformation matrix:\n", T_M)
                continue

            if header == 's':
                print("[TCP] Trigger received. Start sending skeleton frames...")
                queue_for_main_visualizer.put("skeleton")
                sending_stop_flag.set()
                if sending_thread and sending_thread.is_alive():
                    sending_thread.join()
                sending_stop_flag.clear()
                sending_thread = threading.Thread(target=skeleton_sending_loop, args=(conn,))
                sending_thread.start()
                continue

            if header == 'p':
                print("[TCP] Trigger received. Start sending POINTCLOUD frames...")
                queue_for_main_visualizer.put("pointcloud")
                sending_stop_flag.set()
                if sending_thread and sending_thread.is_alive():
                    sending_thread.join()
                sending_stop_flag.clear()
                sending_thread = threading.Thread(target=pointcloud_sending_loop, args=(conn,))
                sending_thread.start()

    except Exception as e:
        print("[TCP] Connection error:", e)

    finally:
        print("[TCP] Closing socket...")
        if sending_thread and sending_thread.is_alive():
            sending_stop_flag.set()
            sending_thread.join()
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except:
            pass
        try:
            conn.close()
            sSock.close()
        except:
            pass
        os._exit(0)
# Starts a keyboard listener that triggers save/start functions on specific key presses.
def on_press(key):
    try:
        if key.char == 'a':
            print("⌨️ Pressed 'b' — Start saving")
            start_saving()
        elif key.char == 's':
            print("⌨️ Pressed 'e' — Stop saving")
            stop_saving()
    except AttributeError:
        pass
def keyboard_listener():
    with kb.Listener(on_press=on_press) as listener:
        listener.join()

#Main
if __name__ == "__main__":
    import queue
    queue_for_main_visualizer = queue.Queue()
    server_thread = threading.Thread(target=tcp_server_trigger_push, args=(queue_for_main_visualizer,))
    server_thread.start()
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    visualizer_thread = None
    while True:
        try:
            trigger = queue_for_main_visualizer.get(timeout=1.0)
            if trigger == "pointcloud":
                print("Launching pointcloud visualizer in new thread...")

                if visualizer_thread and visualizer_thread.is_alive():
                    print("Stopping existing visualizer thread...")


                visualizer_thread = threading.Thread(target=main_pointcloud_visualizer_loop)
                visualizer_thread.start()

            elif trigger == "skeleton":
                print("Launching skeleton visualizer in new thread...")

                if visualizer_thread and visualizer_thread.is_alive():
                    print("Stopping existing visualizer thread...")

                visualizer_thread = threading.Thread(target=main_3DSkeleton_visualizer_loop)
                visualizer_thread.start()

        except queue.Empty:
            continue