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
            rospy.init_node('cartesian_path_driver', anonymous=True)

        # --- 关键修改：切换控制器话题 ---
        # 确保这个话题与你的 controller_manager 配置一致
        self.target_topic = "/my_cartesian_motion_controller/target_frame"
        self.base_frame = "panda_link0" # 根据你的机器人修改，通常是 panda_link0
        self.ee_frame = "panda_link8"       # 根据你的机器人修改，通常是 panda_link8

        # 绝对安全待机点
        self.home_pos = np.array([0.306, 0.0, 0.487])
        self.home_rot = np.array([1.0, 0.0, 0.0, 0.0])

        self.pub = rospy.Publisher(self.target_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()
        
        self.path_queue = Queue()
        self.is_running = True
        self.target_speed = 0.05  # 5厘米/秒
        self.rate_hz = 100.0      # Motion Controller 通常 100Hz 足够，太高可能抖动

        self.worker_thread = threading.Thread(target=self._tape_player_executor)
        self.worker_thread.setDaemon(True)
        self.worker_thread.start()

        # ====================================================
        # [新增] 视觉伺服 PID 专用变量 (原有功能不受影响)
        # ====================================================
        self.is_tracking = False
        self.track_target_p = None
        self.track_target_r = None
        self.current_error = np.array([0.0, 0.0, 0.0])
        
        # PID 参数矩阵 [X, Y, Z]
        self.Kp = np.array([0.0005, 0.0005, 0.0005]) 
        self.Ki = np.array([0.0, 0.0, 0.0]) 
        self.Kd = np.array([0.0001, 0.0001, 0.0001])
        
        self.integral = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0, 0.0])
        self.max_tracking_vel = 0.1 # 安全限速 0.1m/s

        print(f"⏳ [Robot] 等待 TF 变换...")
        try:
            self.tf_listener.waitForTransform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(5.0))
            print(f"✅ [Robot] 笛卡尔运动控制器模式已就绪！")
        except Exception as e:
            print(f"❌ [Robot] TF 失败: {e}")

    def get_current_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.ee_frame, rospy.Time(0))
            return np.array(trans), np.array(rot)
        except:
            return None, None

    def execute_path(self, path_list, speed=None):
        if not self.path_queue.empty():
            print("⚠️ [Robot] 轨迹执行中，忽略新指令")
            return

        if not path_list: return
        if speed is not None: self.target_speed = speed

        # 获取当前物理位置作为平滑起点的依据
        start_p, start_r = self.get_current_pose()
        if start_p is None: return

        with self.path_queue.mutex:
            self.path_queue.queue.clear()

        # --- 轨迹预处理 (S型平滑重采样) ---
        # 逻辑保持不变，确保发给控制器的点是连续且平滑的
        self._resample_path(path_list, start_p, start_r)

    def _resample_path(self, path_list, start_p, start_r):
        """将离散点转换为时间连续的帧流"""
        first_target = path_list[0]
        pts = [start_p] + [np.array(pt['pos']) for pt in path_list]
        rots = [start_r] + [np.array(pt['rot']) for pt in path_list]

        # 计算总里程
        cum_dist = [0.0]
        for i in range(1, len(pts)):
            cum_dist.append(cum_dist[-1] + np.linalg.norm(pts[i] - pts[i-1]))
        
        total_dist = cum_dist[-1]
        if total_dist < 0.001: return

        total_time = total_dist / self.target_speed
        total_steps = int(total_time * self.rate_hz)

        for i in range(1, total_steps + 1):
            t = i / float(total_steps)
            # S-Curve 缓动插值
            smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0
            target_d = smooth_t * total_dist
            
            idx = np.searchsorted(cum_dist, target_d)
            if idx >= len(cum_dist): idx = len(cum_dist) - 1
            
            ratio = (target_d - cum_dist[idx-1]) / (cum_dist[idx] - cum_dist[idx-1] + 1e-9)
            curr_p = pts[idx-1] + (pts[idx] - pts[idx-1]) * ratio
            curr_r = quaternion_slerp(rots[idx-1], rots[idx], ratio)
            
            self.path_queue.put((curr_p, curr_r))
        
        print(f"🚀 [Path] 采样完成，共 {self.path_queue.qsize()} 帧")

    def _tape_player_executor(self):
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz # 新增时间步长用于积分
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame
        
        while self.is_running and not rospy.is_shutdown():
            msg.header.stamp = rospy.Time.now()
            should_publish = False
            
            # --- 优先级 1：原有的离散轨迹执行逻辑 (完全保留) ---
            if not self.path_queue.empty():
                goal_p, goal_r = self.path_queue.get()
                
                msg.pose.position.x = goal_p[0]
                msg.pose.position.y = goal_p[1]
                msg.pose.position.z = goal_p[2]
                msg.pose.orientation.x = goal_r[0]
                msg.pose.orientation.y = goal_r[1]
                msg.pose.orientation.z = goal_r[2]
                msg.pose.orientation.w = goal_r[3]
                
                self.path_queue.task_done()
                should_publish = True
            
            # --- 优先级 2：新增的视觉伺服跟踪逻辑 ---
            elif self.is_tracking and self.track_target_p is not None:
                err = self.current_error
                
                # PID 计算
                P_out = self.Kp * err
                self.integral += err * dt
                self.integral = np.clip(self.integral, -1000, 1000) 
                I_out = self.Ki * self.integral
                derivative = (err - self.prev_error) / dt
                D_out = self.Kd * derivative
                self.prev_error = err
                
                vel = P_out + I_out + D_out
                vel = np.clip(vel, -self.max_tracking_vel, self.max_tracking_vel)
                
                # 积分计算出目标位移
                self.track_target_p += vel * dt
                
                msg.pose.position.x = self.track_target_p[0]
                msg.pose.position.y = self.track_target_p[1]
                msg.pose.position.z = self.track_target_p[2]
                
                # 保持启动时的姿态不变
                msg.pose.orientation.x = self.track_target_r[0]
                msg.pose.orientation.y = self.track_target_r[1]
                msg.pose.orientation.z = self.track_target_r[2]
                msg.pose.orientation.w = self.track_target_r[3]
                
                should_publish = True

            # 如果有数据就发布
            if should_publish:
                self.pub.publish(msg)
                
            rate.sleep()

    # ====================================================
    # [新增] 视觉伺服专用接口
    # ====================================================
    def start_tracking(self):
        """开启跟踪模式"""
        p, r = self.get_current_pose()
        if p is not None:
            self.track_target_p = p.copy()
            self.track_target_r = r.copy() # 锁定启动时的姿态
            self.integral.fill(0.0)
            self.prev_error.fill(0.0)
            self.current_error.fill(0.0)
            self.is_tracking = True
            print("👁️ [Robot] 视觉 PID 跟踪模式已启动 (队列空闲时接管)")

    def stop_tracking(self):
        """停止跟踪模式"""
        self.is_tracking = False
        print("🛑 [Robot] 视觉 PID 跟踪已停止")

    def update_tracking_error(self, err_x, err_y, err_z=0.0):
        """更新图像误差"""
        if self.is_tracking:
            self.current_error = np.array([err_x, err_y, err_z])
    # ====================================================