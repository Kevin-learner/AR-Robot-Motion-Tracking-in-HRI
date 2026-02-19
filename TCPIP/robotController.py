#!/usr/bin/env python3
import rospy
import threading
import numpy as np
import geometry_msgs.msg
import tf
import math
from queue import Queue
# 核心：引入四元数转换库
from tf.transformations import quaternion_slerp 

class RobotController:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('hololens_path_driver', anonymous=True)

        self.target_topic = "/my_cartesian_motion_controller/target_frame"
        self.base_frame = "panda_link0"
        self.ee_frame = "panda_link8" 

        self.pub = rospy.Publisher(self.target_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()
        
        self.path_queue = Queue()      
        self.is_running = True         
        self.target_speed = 0.02       
        
        # 存储当前和目标姿态用于插值 (包含 pos 和 rot)
        self.last_target_pose = None 

        self.worker_thread = threading.Thread(target=self._path_executor)
        self.worker_thread.setDaemon(True) 
        self.worker_thread.start()

        print(f"⏳ [Robot] 等待 TF 变换...")
        try:
            self.tf_listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(5.0))
            # 初始化：将当前位姿存为第一个“上一次目标”
            p, r = self.get_current_pose()
            self.last_target_pose = (p, r)
            print(f"✅ [Robot] 笛卡尔平滑模式已解锁姿态锁定！")
        except Exception as e:
            print(f"❌ [Robot] TF 失败: {e}")

    def get_current_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
            return np.array(trans), rot 
        except:
            return None, None

    def _path_executor(self):
        rate_hz = 200 # 高频发布
        rate = rospy.Rate(rate_hz)
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame

        while self.is_running and not rospy.is_shutdown():
            if not self.path_queue.empty():
                # --- 核心修改 1: 解析字典数据 ---
                step_data = self.path_queue.get()
                
                # 从字典中提取目标位置和目标旋转
                target_pos = np.array(step_data['pos'])
                target_rot = step_data['rot'] # 已经从 Unity 传过来的 [qx, qy, qz, qw]
                
                # 获取起点（即上一个路段的终点）
                start_pos, start_rot = self.last_target_pose
                
                # 计算距离（只基于位置）
                dist = np.linalg.norm(target_pos - start_pos)

                if dist > 0.0001:
                    duration = dist / max(self.target_speed, 0.001)
                    total_steps = int(duration * rate_hz)

                    for i in range(1, total_steps + 1):
                        if not self.is_running or rospy.is_shutdown(): break
                        
                        # 进度 t (0->1)
                        t = i / float(total_steps)
                        
                        # --- 核心改进 1: S型平滑进度 ---
                        smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0

                        # --- 核心改进 2: 位置插值 ---
                        curr_p = start_pos + (target_pos - start_pos) * smooth_t

                        # --- 核心改进 3: 姿态解锁 (使用 SLERP 插值到目标旋转) ---
                        # 不再使用 current_real_rot，而是使用从 Unity 传过来的 target_rot
                        curr_r = quaternion_slerp(start_rot, target_rot, smooth_t)

                        # 发布指令
                        msg.header.stamp = rospy.Time.now()
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = curr_p
                        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = curr_r

                        self.pub.publish(msg)
                        rate.sleep()
                
                # 更新最后一次目标，供下一段路径使用
                self.last_target_pose = (target_pos, target_rot)
                self.path_queue.task_done()
            else:
                rate.sleep()

    def execute_path(self, path_list, speed=None):
        """
        接收完整的字典列表: [{'pos': [x,y,z], 'rot': [x,y,z,w]}, ...]
        """
        if not path_list: return
        if speed is not None: self.target_speed = speed

        # 启动新路径前，获取当前最新的真实位姿作为起点
        p, r = self.get_current_pose()
        if p is not None:
            self.last_target_pose = (p, r)

        with self.path_queue.mutex:
            self.path_queue.queue.clear()
        
        for step in path_list:
            # 存入完整的字典（包含位置和姿态）
            self.path_queue.put(step)
        
        print(f"🚀 [Path] 正在执行带 LShape 姿态的平滑路径...")

# ==========================================
# 测试入口 (兼容新格式)
# ==========================================
if __name__ == "__main__":
    try:
        controller = RobotController()
        rospy.sleep(1.0)
        
        start_p, start_r = controller.get_current_pose()
        if start_p is not None:
            # 模拟新数据格式进行测试
            test_path = [
                {'pos': start_p + np.array([0, 0.05, 0]), 'rot': start_r},
                {'pos': start_p + np.array([0, 0.10, 0]), 'rot': start_r}
            ]
            
            input("👉 按回车开始执行测试路径...")
            controller.execute_path(test_path, speed=0.02)
            
            rospy.spin()
    except rospy.ROSInterruptException:
        pass