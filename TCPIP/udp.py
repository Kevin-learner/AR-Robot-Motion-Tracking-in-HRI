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
    child  = rospy.get_param("~child_frame",  "panda_link8") #

    # --- new params ---
    deadband  = float(rospy.get_param("~deadband", 0.0005))  # meters, default 0.5mm
    send_mode = rospy.get_param("~send_mode", "hold")         # "hold" or "skip"
    alpha     = float(rospy.get_param("~alpha", 0.0))         # EMA smoothing, 0=off, typical 0.2~0.5

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
    last_sent = None          # last "stable" xyz we send
    last_raw  = None          # last raw xyz (for comparing)

    while not rospy.is_shutdown():
        try:
            t = buf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(0.2))
            raw = (t.transform.translation.x,
                   t.transform.translation.y,
                   t.transform.translation.z)

            # initialize
            if last_sent is None:
                last_sent = raw
                last_raw = raw

            # optional EMA smoothing applied to raw before deadband compare
            cur = raw
            if alpha > 0.0:
                cur = ema(last_sent, raw, alpha)

            # decide update
            d = dist(cur, last_sent)
            updated = False
            if d >= deadband:
                last_sent = cur
                updated = True

            # send policy
            if updated or send_mode == "hold":
                x, y, z = last_sent
                pkt = struct.pack("!fff", float(x), float(y), float(z))
                sock.sendto(pkt, (target_ip, port))
                seq += 1
                if updated:
                    rospy.loginfo("sent #%d UPDATED d=%.6f xyz=(%.6f, %.6f, %.6f)", seq, d, x, y, z)
                else:
                    rospy.loginfo("sent #%d HOLD    d=%.6f xyz=(%.6f, %.6f, %.6f)", seq, d, x, y, z)
            else:
                rospy.loginfo("skip (d=%.6f < deadband)", d)

            last_raw = raw

        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF/UDP failed: %s", str(e))

        rate.sleep()

if __name__ == "__main__":
    main()
