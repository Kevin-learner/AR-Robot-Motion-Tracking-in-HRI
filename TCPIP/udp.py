#!/usr/bin/env python3
import rospy
import socket
import struct
import tf2_ros
import math

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def ema(prev, cur, alpha):
    # alpha in [0,1], higher -> smoother
    return (
        (1-alpha)*prev[0] + alpha*cur[0],
        (1-alpha)*prev[1] + alpha*cur[1],
        (1-alpha)*prev[2] + alpha*cur[2],
    )

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
            t = buf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(0.2))
            
            # 1. 获取平移 (XYZ)
            raw_pos = (t.transform.translation.x,
                       t.transform.translation.y,
                       t.transform.translation.z)
            
            # 🌟 2. 获取姿态 (Quaternion)
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
                qx, qy, qz, qw = raw_quat # 姿态直接使用最新值，不加 deadband 限制
                
                # 🌟 3. 打包 7 个 float (28 字节)
                pkt = struct.pack("!fffffff", float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw))
                sock.sendto(pkt, (target_ip, port))
                seq += 1
                
                if updated:
                    rospy.loginfo("sent #%d UPDATED d=%.6f xyz=(%.4f, %.4f, %.4f) q=(%.2f, %.2f, %.2f, %.2f)", 
                                  seq, d, x, y, z, qx, qy, qz, qw)
                else:
                    # 避免 HOLD 模式疯狂刷屏，仅在调试时取消注释
                    rospy.loginfo("sent #%d HOLD    d=%.6f xyz=(%.4f, %.4f, %.4f) q=(%.3f, %.3f, %.3f, %.3f)", 
                                  seq, d, x, y, z, qx, qy, qz, qw)
                    pass 
            else:
                pass

            last_raw_pos = raw_pos

        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF/UDP failed: %s", str(e))

        rate.sleep()

if __name__ == "__main__":
    main()