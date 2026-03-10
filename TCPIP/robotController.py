#!/usr/bin/env python3
import rospy
import threading
import numpy as np
import geometry_msgs.msg
import tf
import math
from queue import Queue
from tf.transformations import quaternion_slerp 

class RobotController:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('hololens_path_driver', anonymous=True)

        # ================= 核心配置：全程使用阻抗控制器 =================
        self.controller_name = "cartesian_impedance_example_controller"
        self.target_topic = f"/{self.controller_name}/equilibrium_pose"
        
        self.base_frame = "panda_link0"
        self.ee_frame = "panda_link8" # 暴力锚定 link8 (或者 panda_EE)

        # 发布平衡点 (Equilibrium Pose)
        self.pub = rospy.Publisher(self.target_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
        
        # 力控核心参数
        self.K_z = 400.0 # Z轴刚度 N/m 

        self.tf_listener = tf.TransformListener()
        self.path_queue = Queue()      
        self.is_running = True         
        self.target_speed = 0.02       
        
        # 存储当前状态用于插值: (pos, rot, force)
        self.last_target_state = None 

        self.worker_thread = threading.Thread(target=self._path_executor)
        self.worker_thread.setDaemon(True) 
        self.worker_thread.start()

        print(f"⏳ [Robot] 等待 TF 变换...")
        try:
            self.tf_listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(5.0))
            # 初始化起点：当前法兰位置，当前姿态，初始力0N
            p, r = self.get_current_pose()
            self.last_target_state = (p, r, 0.0)
            print(f"✅ [Robot] 暴力 link8 阻抗控制模式已就绪！")
        except Exception as e:
            print(f"❌ [Robot] TF 失败: {e}")

    def get_current_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
            return np.array(trans), rot 
        except:
            return None, None

    def _path_executor(self):
        rate_hz = 200 
        rate = rospy.Rate(rate_hz)
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame

        while self.is_running and not rospy.is_shutdown():
            if not self.path_queue.empty():
                step_data = self.path_queue.get()
                
                target_pos = np.array(step_data['pos'])
                target_rot = step_data['rot']
                target_force = step_data.get('force', 0.0) 
                
                start_pos, start_rot, start_force = self.last_target_state
                
                dist = np.linalg.norm(target_pos - start_pos)
                force_diff = abs(target_force - start_force)

                if dist > 0.0001 or force_diff > 0.01:
                    pos_duration = dist / max(self.target_speed, 0.001)
                    force_duration = force_diff / 1.0 
                    duration = max(pos_duration, force_duration)
                    
                    # ==========================================================
                    # 🌟 1. 智能判断：是稀疏关键点，还是密集的 B 样条点？
                    # 假设两点之间距离小于 1 厘米 (0.01m)，我们就认为是密集的连续曲线
                    is_dense_curve = dist < 0.01 
                    
                    # 只有稀疏点，才强制加 0.1s 的缓冲时间；密集点不需要！
                    if not is_dense_curve:
                        if duration < 0.1: duration = 0.1 
                    # ==========================================================

                    # 确保哪怕时间极短，至少执行 1 个 step
                    total_steps = max(1, int(duration * rate_hz))

                    for i in range(1, total_steps + 1):
                        if not self.is_running or rospy.is_shutdown(): break
                        
                        t = i / float(total_steps)
                        
                        # ==========================================================
                        # 🌟 2. 动态切换插值算法 (解决卡顿的核心)
                        if is_dense_curve:
                            # 密集点：使用【匀速直线插值】。点够密了，直接匀速穿过，不刹车！
                            smooth_t = t 
                        else:
                            # 稀疏点：使用【S型起停插值】。保护机械臂在长距离移动时平滑。
                            smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0
                        # ==========================================================

                        curr_p = start_pos + (target_pos - start_pos) * smooth_t
                        curr_r = quaternion_slerp(start_rot, target_rot, smooth_t)
                        curr_f = start_force + (target_force - start_force) * smooth_t 


                        ee_pos = curr_p
                        ee_rot = curr_r
                        # --- (如果你还在用笔尖补偿，保留下面这行，否则删掉) ---
                        #ee_pos, ee_rot = tool_tip_to_ee(curr_p, curr_r, self.tool_length)
                        # ----------------------------------------------------

                        # 阻抗控制 Z 轴偏移补偿
                        z_offset = curr_f / self.K_z
                        actual_target_z = ee_pos[2] - z_offset 

                        msg.header.stamp = rospy.Time.now()
                        msg.pose.position.x = ee_pos[0]
                        msg.pose.position.y = ee_pos[1]
                        msg.pose.position.z = actual_target_z
                        
                        msg.pose.orientation.x = ee_rot[0]
                        msg.pose.orientation.y = ee_rot[1]
                        msg.pose.orientation.z = ee_rot[2]
                        msg.pose.orientation.w = ee_rot[3]

                        self.pub.publish(msg)
                        rate.sleep()
                
                self.last_target_state = (target_pos, target_rot, target_force)
                self.path_queue.task_done()
            else:
                rate.sleep()

    def execute_path(self, path_list, speed=None):
        if not path_list: return
        if speed is not None: self.target_speed = speed

        # 暴力获取 link8 当前位置当做起点
        p, r = self.get_current_pose()
        if p is not None:
            current_f = self.last_target_state[2] if self.last_target_state else 0.0
            self.last_target_state = (p, r, current_f)

        with self.path_queue.mutex:
            self.path_queue.queue.clear()
        
        for step in path_list:
            self.path_queue.put(step)
        
        print(f"🚀 [Path] 正在执行暴力 link8 轨迹，点数: {len(path_list)}")