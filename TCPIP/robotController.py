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
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame
        
        while self.is_running and not rospy.is_shutdown():
            if not self.path_queue.empty():
                goal_p, goal_r = self.path_queue.get()
                
                # --- 运动控制器核心：直接发布目标位姿 ---
                msg.header.stamp = rospy.Time.now()
                msg.pose.position.x = goal_p[0]
                msg.pose.position.y = goal_p[1]
                msg.pose.position.z = goal_p[2]
                
                msg.pose.orientation.x = goal_r[0]
                msg.pose.orientation.y = goal_r[1]
                msg.pose.orientation.z = goal_r[2]
                msg.pose.orientation.w = goal_r[3]
                
                self.pub.publish(msg)
                self.path_queue.task_done()
            
            rate.sleep()