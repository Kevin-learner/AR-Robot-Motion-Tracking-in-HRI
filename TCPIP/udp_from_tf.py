#!/usr/bin/env python3
import rospy
import socket
import struct
import tf2_ros
import math

from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import JointState # 🌟 新增：订阅关节角度需要用到

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def ema(prev, cur, alpha):
    # alpha in [0,1], higher -> smoother
    return (
        (1-alpha)*prev[0] + alpha*cur[0],
        (1-alpha)*prev[1] + alpha*cur[1],
        (1-alpha)*prev[2] + alpha*cur[2],
    )

# 🌟 新增：全局变量存储最新的关节角
latest_joint_angles = {
    "panda_joint1": 0.0, "panda_joint2": 0.0, "panda_joint3": 0.0,
    "panda_joint4": 0.0, "panda_joint5": 0.0, "panda_joint6": 0.0,
    "panda_joint7": 0.0
}

def joint_callback(msg):
    global latest_joint_angles
    for i, name in enumerate(msg.name):
        if name in latest_joint_angles:
            latest_joint_angles[name] = msg.position[i]

def main():
    rospy.init_node("udp_from_tf")

    target_ip = rospy.get_param("~ip", "100.93.142.100")
    port      = int(rospy.get_param("~port", 5006))
    hz        = float(rospy.get_param("~hz", 1.0))

    parent = rospy.get_param("~parent_frame", "panda_link0")
    child  = rospy.get_param("~child_frame",  "panda_link8")

    # --- params ---
    deadband  = float(rospy.get_param("~deadband", 0.0005))  
    send_mode = rospy.get_param("~send_mode", "hold")        
    alpha     = float(rospy.get_param("~alpha", 0.0))        

    if send_mode not in ("hold", "skip"):
        rospy.logwarn("~send_mode should be 'hold' or 'skip', got '%s', using 'hold'", send_mode)
        send_mode = "hold"
    alpha = max(0.0, min(1.0, alpha))

    # 🌟 新增：订阅关节状态话题
    rospy.Subscriber("/joint_states", JointState, joint_callback)

    # 🌟 新增：我们要获取的 7 个关节和 Link (最后一个使用传进来的 child_frame)
    joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
        "dummy_joint"
    ]
    child_frames = [
        "panda_link1", "panda_link2", "panda_link3", 
        "panda_link4", "panda_link5", "panda_link6", "panda_link7",
        child 
    ]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    _ = tf2_ros.TransformListener(buf)

    rate = rospy.Rate(hz if hz > 0 else 1.0)
    rospy.loginfo("UDP -> %s:%d | TF: %s -> %s | %.2f Hz | deadband=%.6f m | mode=%s | alpha=%.2f",
                  target_ip, port, parent, child, hz, deadband, send_mode, alpha)

    seq = 0
    last_sent_pos = None          
    last_raw_pos  = None          

    while not rospy.is_shutdown():
        try:
            # 获取末端坐标用于原有的判断逻辑
            t = buf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(0.2))
            
            # 1. 获取平移 (XYZ)
            raw_pos = (t.transform.translation.x,
                       t.transform.translation.y,
                       t.transform.translation.z)
            
            # 2. 获取姿态 (Quaternion)
            raw_quat = (t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w)

            # initialize
            if last_sent_pos is None:
                last_sent_pos = raw_pos
                last_raw_pos = raw_pos

            # optional EMA smoothing applied to raw position
            cur_pos = raw_pos
            if alpha > 0.0:
                cur_pos = ema(last_sent_pos, raw_pos, alpha)

            # decide update based on position
            d = dist(cur_pos, last_sent_pos)
            updated = False
            if d >= deadband:
                last_sent_pos = cur_pos
                updated = True

            # send policy
            if updated or send_mode == "hold":
                x, y, z = last_sent_pos
                qx, qy, qz, qw = raw_quat 

                rx_rad, ry_rad, rz_rad = euler_from_quaternion([qx, qy, qz, qw])
                rx_deg = math.degrees(rx_rad)
                ry_deg = math.degrees(ry_rad)
                rz_deg = math.degrees(rz_rad)
                
                # 🌟 新增：打包 7 个关节全量数据 (56 个 float)
                payload = []
                for i in range(8):
                    j_name = joint_names[i]
                    link_name = child_frames[i]
                    
                    angle = latest_joint_angles.get(j_name, 0.0)
                    
                    if i == 7: 
                        # 第 8 个就是末端，直接使用上面经过平滑和死区判断的最终值
                        link_x, link_y, link_z = x, y, z
                        link_qx, link_qy, link_qz, link_qw = qx, qy, qz, qw
                    else:
                        # 获取前 7 个关节的 TF
                        t_link = buf.lookup_transform(parent, link_name, rospy.Time(0), rospy.Duration(0.05))
                        link_x = t_link.transform.translation.x
                        link_y = t_link.transform.translation.y
                        link_z = t_link.transform.translation.z
                        link_qx = t_link.transform.rotation.x
                        link_qy = t_link.transform.rotation.y
                        link_qz = t_link.transform.rotation.z
                        link_qw = t_link.transform.rotation.w
                        
                    payload.extend([angle, link_x, link_y, link_z, link_qx, link_qy, link_qz, link_qw])

                # 🌟 打包 56 个 float (224 字节) 发送
                pkt = struct.pack("!64f", *payload)
                sock.sendto(pkt, (target_ip, port))
                seq += 1
                
                # 完全保留你原有的打印日志逻辑
                
                rospy.loginfo("sent #%d UPDATED d=%.6f xyz=(%.4f, %.4f, %.4f) Euler[Deg]=(Rx:%.1f, Ry:%.1f, Rz:%.1f) q=(%.2f, %.2f, %.2f, %.2f)", 
                                  seq, d, x, y, z, rx_deg, ry_deg, rz_deg, qx, qy, qz, qw)
                
            else:
                pass

            last_raw_pos = raw_pos

        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF/UDP failed: %s", str(e))

        rate.sleep()

if __name__ == "__main__":
    main()