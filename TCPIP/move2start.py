#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import math
import geometry_msgs.msg
from tf.transformations import quaternion_slerp

# =========================================================
# 🌟 在这里填入你机械臂的“初始绝对安全坐标” (法兰盘坐标)
# =========================================================
HOME_POS = np.array([0.306, 0.0, 0.487])       # 替换为你的真实起始 X, Y, Z
HOME_ROT = np.array([1.0, 0.0, 0.0, 0.0])      # 替换为你的真实起始 qx, qy, qz, qw (通常向下是 1,0,0,0)
# =========================================================

def move_to_home():
    rospy.init_node('standalone_home_mover', anonymous=True)
    
    controller_name = "cartesian_impedance_example_controller"
    target_topic = f"/{controller_name}/equilibrium_pose"
    
    pub = rospy.Publisher(target_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
    tf_listener = tf.TransformListener()
    
    base_frame = "panda_link0"
    ee_frame = "panda_link8"
    
    print("⏳ 等待 TF 变换...")
    try:
        tf_listener.waitForTransform(base_frame, ee_frame, rospy.Time(0), rospy.Duration(3.0))
        (curr_trans, curr_rot) = tf_listener.lookupTransform(base_frame, ee_frame, rospy.Time(0))
        start_pos = np.array(curr_trans)
        start_rot = np.array(curr_rot)
        print(f"✅ 获取到当前位置: {np.round(start_pos, 3)}")
    except Exception as e:
        print(f"❌ 无法获取当前位置，请检查 ROS 或 TF: {e}")
        return

    # 计算移动距离来决定耗时 (速度设为 0.05 m/s)
    dist = np.linalg.norm(HOME_POS - start_pos)
    speed = 0.05 
    duration = max(dist / speed, 1.0) # 至少花 1 秒钟，保证安全
    
    rate_hz = 200
    total_steps = int(duration * rate_hz)
    rate = rospy.Rate(rate_hz)
    
    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = base_frame
    
    print(f"🚀 开始平滑回零... 预计耗时: {duration:.1f} 秒")
    
    for i in range(1, total_steps + 1):
        if rospy.is_shutdown():
            break
            
        t = i / float(total_steps)
        # S型速度插值，极其平滑
        smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0 
        
        curr_p = start_pos + (HOME_POS - start_pos) * smooth_t
        curr_r = quaternion_slerp(start_rot, HOME_ROT, smooth_t)
        
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = curr_p[0]
        msg.pose.position.y = curr_p[1]
        msg.pose.position.z = curr_p[2]
        msg.pose.orientation.x = curr_r[0]
        msg.pose.orientation.y = curr_r[1]
        msg.pose.orientation.z = curr_r[2]
        msg.pose.orientation.w = curr_r[3]
        
        pub.publish(msg)
        rate.sleep()

    print("🏁 回零完毕，机械臂已到达安全位置！")

    # 🌟 加上这个死循环！让它永远不退出，持续保持当前姿态！
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_to_home()
    except rospy.ROSInterruptException:
        pass