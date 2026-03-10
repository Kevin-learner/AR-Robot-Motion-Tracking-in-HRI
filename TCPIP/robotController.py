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
        self.ee_frame = "panda_link8" 

        # 唯一发布者：发布平衡点 (Equilibrium Pose)
        self.pub = rospy.Publisher(self.target_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
        
        # 力控核心参数
        self.K_z = 1000.0 # Z轴刚度 N/m (必须与 rqt_reconfigure 里面的 translational_stiffness 保持一致)

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
            # 初始化起点：当前位置，当前姿态，初始力设为 0.0N
            p, r = self.get_current_pose()
            self.last_target_state = (p, r, 0.0)
            print(f"✅ [Robot] 统一阻抗控制模式已就绪！")
        except Exception as e:
            print(f"❌ [Robot] TF 失败: {e}")

    def get_current_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
            return np.array(trans), rot 
        except:
            return None, None

    def _path_executor(self):
        rate_hz = 200 # 高频发布，保证弹簧系统的稳定性
        rate = rospy.Rate(rate_hz)
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame

        while self.is_running and not rospy.is_shutdown():
            if not self.path_queue.empty():
                step_data = self.path_queue.get()
                
                # --- 1. 提取目标数据 ---
                target_pos = np.array(step_data['pos'])
                target_rot = step_data['rot']
                
                # 【神来之笔】：如果字典里没有 'force' 键（说明是纯位置模式发来的），默认设为 0N
                target_force = step_data.get('force', 0.0) 
                
                # --- 2. 提取起点数据 ---
                start_pos, start_rot, start_force = self.last_target_state
                
                # --- 3. 计算运动时间 ---
                dist = np.linalg.norm(target_pos - start_pos)
                force_diff = abs(target_force - start_force)

                # 只要位置有移动，或者力有变化（原地施压），就触发插值运动
                if dist > 0.0001 or force_diff > 0.1:
                    # 确保即使距离很短，力的改变也有足够的缓冲时间 (至少0.5秒)
                    duration = max(dist / max(self.target_speed, 0.001), 0.5 if force_diff > 0.1 else 0.0) 
                    total_steps = int(duration * rate_hz)

                    for i in range(1, total_steps + 1):
                        if not self.is_running or rospy.is_shutdown(): break
                        
                        t = i / float(total_steps)
                        smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0 # S型速度曲线

                        # --- 4. 核心插值计算 ---
                        curr_p = start_pos + (target_pos - start_pos) * smooth_t
                        curr_r = quaternion_slerp(start_rot, target_rot, smooth_t)
                        curr_f = start_force + (target_force - start_force) * smooth_t # 力的平滑渐变

                        # --- 5. 阻抗控制 Z 轴偏移补偿 (胡克定律) ---
                        # 无论此时 curr_f 是多少 (哪怕是0)，这套公式都通用
                        z_offset = curr_f / self.K_z
                        
                        # 真实发送给控制器的 Z = Unity轨迹Z - 弹簧下压量
                        actual_target_z = curr_p[2] - z_offset 

                        # --- 6. 构建并发布 ROS 消息 ---
                        msg.header.stamp = rospy.Time.now()
                        msg.pose.position.x = curr_p[0]
                        msg.pose.position.y = curr_p[1]
                        msg.pose.position.z = actual_target_z
                        
                        msg.pose.orientation.x = curr_r[0]
                        msg.pose.orientation.y = curr_r[1]
                        msg.pose.orientation.z = curr_r[2]
                        msg.pose.orientation.w = curr_r[3]

                        self.pub.publish(msg)
                        rate.sleep()
                
                # --- 7. 更新最后一次状态 ---
                self.last_target_state = (target_pos, target_rot, target_force)
                self.path_queue.task_done()
            else:
                rate.sleep()

    def execute_path(self, path_list, speed=None):
        """
        接收完整的字典列表并执行: [{'pos': [x,y,z], 'rot': [x,y,z,w], 'force': 5.0}, ...]
        注意：如果没有 'force' 键，将自动作为 0N 的柔顺位置控制处理。
        """
        if not path_list: return
        if speed is not None: self.target_speed = speed

        # 获取当前实际位姿作为新轨迹起点
        p, r = self.get_current_pose()
        if p is not None:
            # 继承上一次的力，防止两段路径拼接时力瞬间归零
            current_f = self.last_target_state[2] if self.last_target_state else 0.0
            self.last_target_state = (p, r, current_f)

        # 清空旧队列，压入新任务
        with self.path_queue.mutex:
            self.path_queue.queue.clear()
        
        for step in path_list:
            self.path_queue.put(step)
        
        print(f"🚀 [Path] 正在执行统一阻抗控制路径，点数: {len(path_list)}")

        
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