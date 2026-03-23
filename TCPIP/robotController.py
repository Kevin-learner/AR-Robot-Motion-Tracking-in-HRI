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

        self.controller_name = "cartesian_impedance_example_controller"
        self.target_topic = f"/{self.controller_name}/equilibrium_pose"
        self.base_frame = "panda_link0"
        self.ee_frame = "panda_link8" 

        # 你的绝对安全待机点
        self.home_pos = np.array([0.306, 0.0, 0.487])
        self.home_rot = np.array([1.0, 0.0, 0.0, 0.0])

        self.pub = rospy.Publisher(self.target_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()
        
        self.path_queue = Queue()
        self.is_running = True
        self.target_speed = 0.05  # 目标速度: 2厘米/秒
        self.rate_hz = 200.0      # 极其关键：底层播放频率

        self.worker_thread = threading.Thread(target=self._tape_player_executor)
        self.worker_thread.setDaemon(True)
        self.worker_thread.start()

        print(f"⏳ [Robot] 等待 TF 变换...")
        try:
            self.tf_listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(5.0))
            print(f"✅ [Robot] 阻抗控制就绪！(🌟 纯位置追踪 + 全局流媒体重采样已激活)")
        except Exception as e:
            print(f"❌ [Robot] TF 失败: {e}")

    def get_current_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
            return np.array(trans), np.array(rot)
        except:
            return None, None

    def move_to_start(self, speed=0.03):
        print("🏠 [Robot] 正在平滑返回初始起点 (Home)...")
        self.execute_path([{'pos': self.home_pos, 'rot': self.home_rot}], speed=speed)

    def execute_path(self, path_list, speed=None):
        # ==========================================================
        # 🛡️ 霸体防御：如果队列里还有没播完的帧，拒绝新指令防止半路回头！
        if not self.path_queue.empty():
            print("⚠️ [Robot] 轨迹正在执行中，已拒绝本次重叠指令！请等待走完。")
            return
        # ==========================================================

        if not path_list: return
        if speed is not None: self.target_speed = speed

        # 暴力获取真实物理位置作为绝对起点，杜绝瞬移！
        start_p, start_r = self.get_current_pose()
        if start_p is None: return

        with self.path_queue.mutex:
            self.path_queue.queue.clear()

        print(f"⚙️ [Path] 开始全局轨迹重采样，原始密集点数: {len(path_list)}")

        # ==========================================================
        # 🌟 阶段 1: 完美衔接 (从悬停位置 S型 飞向起笔点)
        first_target = path_list[0]
        dist_to_start = np.linalg.norm(np.array(first_target['pos']) - start_p)

        if dist_to_start > 0.002:
            move_time = max(dist_to_start / self.target_speed, 0.5)
            steps = int(move_time * self.rate_hz)
            for i in range(1, steps + 1):
                t = i / float(steps)
                smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0
                curr_p = start_p + (np.array(first_target['pos']) - start_p) * smooth_t
                curr_r = quaternion_slerp(start_r, first_target['rot'], smooth_t)
                self.path_queue.put((curr_p, curr_r))

        # ==========================================================
        # 🌟 阶段 2: 轨迹全局空间重采样 (带软着陆与强制终点)
        if len(path_list) > 1:
            pts = [np.array(pt['pos']) for pt in path_list]
            rots = [np.array(pt['rot']) for pt in path_list]
            
            # 计算每一步的累计距离
            cum_dist = [0.0]
            for i in range(1, len(pts)):
                cum_dist.append(cum_dist[-1] + np.linalg.norm(pts[i] - pts[i-1]))
            
            total_dist = cum_dist[-1]
            if total_dist > 0.0001:
                total_time = total_dist / self.target_speed
                total_steps = int(total_time * self.rate_hz)
                
                for i in range(1, total_steps + 1):
                    # 🌟 修复 1：全局 S 型速度规划 (两头慢，中间快，防急刹抖动)
                    global_t = i / float(total_steps)
                    smooth_t = (1.0 - math.cos(global_t * math.pi)) / 2.0 
                    target_d = smooth_t * total_dist
                    
                    idx = np.searchsorted(cum_dist, target_d)
                    if idx == 0: idx = 1
                    if idx >= len(cum_dist): idx = len(cum_dist) - 1
                    
                    segment_length = cum_dist[idx] - cum_dist[idx-1]
                    if segment_length > 1e-6:
                        ratio = (target_d - cum_dist[idx-1]) / segment_length
                    else:
                        ratio = 1.0
                        
                    curr_p = pts[idx-1] + (pts[idx] - pts[idx-1]) * ratio
                    curr_r = quaternion_slerp(rots[idx-1], rots[idx], ratio)
                    
                    self.path_queue.put((curr_p, curr_r))

            # 🌟 修复 2：强行锁死绝对终点！弥补所有浮点误差！
            exact_final_p = np.array(path_list[-1]['pos'])
            exact_final_r = np.array(path_list[-1]['rot'])
            self.path_queue.put((exact_final_p, exact_final_r))

        # 注意看这里的打印信息变了！
        print(f"🚀 [Path] 重采样完成！生成 S 型全局帧流，帧数: {self.path_queue.qsize()}")

    def _tape_player_executor(self):
        rate = rospy.Rate(self.rate_hz)
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame
        
        # --- 牵引与下压配置 ---
        desired_tow_force = 5.0    # 横向牵引力 5N
        desired_press_force = 8.0  # 垂直下压力 8N (让你下得去终点)
        stiffness = 1000.0         # 确保你的 rqt_reconfigure 也是这个值
        
        tow_offset = desired_tow_force / stiffness    # 横向拉开 5mm
        press_offset = desired_press_force / stiffness # 向下压入 8mm
        # --------------------

        current_goal = None
        ema_p = None
        alpha = 0.1 # 提高响应速度

        while self.is_running and not rospy.is_shutdown():
            # 1. 获取物理反馈
            curr_p, curr_r = self.get_current_pose()
            
            if curr_p is not None:
                # 2. 如果当前没有目标，或者已经走到了上一个目标，就取新点
                if current_goal is None and not self.path_queue.empty():
                    current_goal = self.path_queue.get()
                
                if current_goal is not None:
                    goal_p = np.array(current_goal[0])
                    goal_r = current_goal[1]
                    
                    # 3. 计算【牵引矢量】
                    vec = goal_p - curr_p
                    dist = np.linalg.norm(vec)
                    
                    # 🌟 核心：只有距离目标点还有一段距离时，才进行牵引
                    if dist > 0.002: # 2mm 容差
                        unit_vec = vec / dist
                        # 平衡点 = 当前物理位置 + 指向目标的牵引偏移
                        equi_p = curr_p + unit_vec * tow_offset
                        # 叠加 Z 轴下压力
                        equi_p[2] -= press_offset 
                    else:
                        # 已经走到了，清除当前目标，让下一循环取新点
                        equi_p = goal_p
                        equi_p[2] -= press_offset
                        self.path_queue.task_done()
                        current_goal = None 
                    
                    # 4. 发布带滤波的指令
                    if ema_p is None: ema_p = equi_p
                    ema_p = alpha * equi_p + (1.0 - alpha) * ema_p

                    msg.header.stamp = rospy.Time.now()
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = ema_p
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = goal_r
                    self.pub.publish(msg)
                    
                    # 调试雷达
                    if rospy.get_time() % 1.0 < 0.005:
                        print(f"📡 [牵引] 距离目标: {dist:.4f}m, 指令Z: {ema_p[2]:.4f}")
            
            rate.sleep()