#!/usr/bin/env python3
import rospy
import threading
import numpy as np
import geometry_msgs.msg
import tf
import math
from queue import Queue
from tf.transformations import quaternion_slerp, quaternion_about_axis, quaternion_multiply
try:
    from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
except ImportError:
    print("⚠️ 警告: 未找到 franka_gripper 环境，请确保已 source ROS 工作空间！")
import time
import actionlib
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

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

        # ====================================================
        # 🚀【全新炮台模式】视觉伺服 PID 专用变量 
        # ====================================================
        self.is_tracking = False
        self.track_start_p = None # 记录追踪起步时的绝对位置
        self.track_start_r = None # 记录追踪起步时的绝对姿态
        
        self.base_yaw = 0.0       # 记录基座需要旋转的总角度 (弧度)
        self.current_error_y = 0.0 # 我们现在只关心左右水平误差！
        
        # 旋转 PID 参数 (角速度控制)
        self.Kp_yaw = 0.5         # 比例系数 (误差转角速度，需调试)
        self.max_yaw_vel = 0.2   # 最大旋转角速度 0.3 rad/s (约 17度/秒，非常稳)
        self.current_yaw_vel = 0.0 # 当前平滑角速度
        self.max_yaw_accel = 0.4  # 最大角加速度 0.4 rad/s^2 (极致平滑起步)

        # ====================================================
        # 🛸【新增】3D 全向视觉伺服专用变量 (递送物品用)
        # ====================================================
        self.is_servoing = False
        self.servo_target_p = None    # 3D 目标坐标
        self.servo_target_r = None    # 锁定的姿态 (递送过程中姿态不能变)
        
        self.current_cmd_p = None     # 内部平滑指令缓存
        
        self.Kp_servo = 2.0           # 3D 追踪的比例系数 (Kp)
        self.max_servo_vel = 0.1      # 最大直线追击速度: 0.1 m/s (10cm/s，非常安全)

        # ====================================================
        # 🦾【新增】力觉传感 (Wrench) 专用变量
        # ====================================================
        # 初始值为 0: [Fx, Fy, Fz, Tx, Ty, Tz]
        self.current_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        
        # 订阅 Franka 的外部估计力话题 (如果不叫这个名字，请用 rostopic list 检查)
        self.wrench_topic = "/franka_state_controller/F_ext"
        self.wrench_sub = rospy.Subscriber(self.wrench_topic, WrenchStamped, self._wrench_callback)
        print(f">> 正在监听末端受力话题: {self.wrench_topic}")

        # ====================================================
        # 🖐️【新增】夹爪状态 (Gripper Width) 专用变量
        # ====================================================
        self.current_gripper_width = 0.0
        self.gripper_joint_topic = "/franka_gripper/joint_states"
        self.gripper_sub = rospy.Subscriber(self.gripper_joint_topic, JointState, self._gripper_joint_callback)
        print(f">> 正在监听夹爪状态话题: {self.gripper_joint_topic}")

        self.worker_thread = threading.Thread(target=self._tape_player_executor)
        self.worker_thread.setDaemon(True)
        self.worker_thread.start()

        # ==========================================
        # 🌟 初始化 Franka 夹爪 Action 客户端
        # ==========================================
        print(">> 正在连接 Franka 夹爪服务...")
        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        
        # 等待服务上线 (设 2 秒超时，防止如果没有开真机导致程序卡死)
        if self.move_client.wait_for_server(rospy.Duration(2.0)):
            print(">> ✅ Franka 夹爪连接成功！")
        else:
            print(">> ⚠️ 未检测到夹爪服务，将以模拟模式运行。")


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
        """将离散点转换为时间连续的帧流 (修复了原地旋转导致的超速Bug)"""
        pts = [start_p] + [np.array(pt['pos']) for pt in path_list]
        rots = [start_r] + [np.array(pt['rot']) for pt in path_list]

        # 1. 计算 XYZ 总直线里程
        cum_dist = [0.0]
        for i in range(1, len(pts)):
            cum_dist.append(cum_dist[-1] + np.linalg.norm(pts[i] - pts[i-1]))
        total_dist = cum_dist[-1]

        # 2. 🌟 核心修复：计算四元数的总旋转角度 (弧度)
        # 取终点和起点的四元数计算夹角
        end_r = rots[-1]
        dot_product = np.clip(np.dot(start_r, end_r), -1.0, 1.0)
        # 两个四元数的夹角公式：theta = 2 * acos(|q1·q2|)
        total_angle = 2.0 * math.acos(abs(dot_product))

        # 3. 分别计算平移和旋转需要的时间
        # 设定的平移速度 self.target_speed (如 0.05 m/s)
        linear_time = total_dist / (self.target_speed + 1e-6) 
        
        # 设定一个安全的最大角速度，比如 0.2 rad/s (约 11度/秒)
        max_angular_speed = 0.5 
        angular_time = total_angle / max_angular_speed

        # 🌟 最终时间取两者的最大值！(如果距离很短但要大转身，就多给点时间)
        total_time = max(linear_time, angular_time)
        
        # 兜底：哪怕完全不动，也给 0.1 秒的缓冲，防止除以 0
        total_time = max(total_time, 0.1) 

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
            
            # 这里对姿态也应用 S 曲线平滑 (原代码 ratio 直接用在线性上没问题，但用在四元数上也要 S 曲线更稳)
            curr_r = quaternion_slerp(rots[idx-1], rots[idx], ratio)
            
            self.path_queue.put((curr_p, curr_r))
        
        print(f"🚀 [Path] 采样完成，平移: {total_dist*100:.1f}cm, 旋转: {math.degrees(total_angle):.1f}° -> 耗时 {total_time:.2f} 秒")

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

            # --- 🌟 优先级 2：全新 3D 伺服递送模式 (追击手部) ---
            elif self.is_servoing and self.servo_target_p is not None and self.current_cmd_p is not None:
                # 1. 计算误差向量 (目标点 - 当前指令点)
                err_vec = self.servo_target_p - self.current_cmd_p
                
                # 2. 计算目标速度 (P 控制)
                target_vel = self.Kp_servo * err_vec
                
                # 3. 速度限幅 (防止手动得太快，机械臂猛冲)
                speed = np.linalg.norm(target_vel)
                if speed > self.max_servo_vel:
                    target_vel = target_vel * (self.max_servo_vel / speed)
                
                # 4. 积分得到下一帧的平滑位置
                self.current_cmd_p += target_vel * dt
                
                # 5. 填装发布信息 (注意：姿态死死锁定为 start_servoing 时的姿态！)
                msg.pose.position.x = self.current_cmd_p[0]
                msg.pose.position.y = self.current_cmd_p[1]
                msg.pose.position.z = self.current_cmd_p[2]
                
                msg.pose.orientation.x = self.servo_target_r[0]
                msg.pose.orientation.y = self.servo_target_r[1]
                msg.pose.orientation.z = self.servo_target_r[2]
                msg.pose.orientation.w = self.servo_target_r[3]
                
                should_publish = True
            
                # --- 优先级 2：炮台视觉伺服模式 (仅绕基座 Z 轴旋转) ---
            elif self.is_tracking and self.track_start_p is not None:
                # 1. 计算目标角速度
                target_yaw_vel = self.Kp_yaw * self.current_error_y
                target_yaw_vel = np.clip(target_yaw_vel, -self.max_yaw_vel, self.max_yaw_vel)
                
                # 2. 角加速度限幅 (物理减震器)
                max_dv = self.max_yaw_accel * dt 
                if target_yaw_vel > self.current_yaw_vel + max_dv:
                    self.current_yaw_vel += max_dv
                elif target_yaw_vel < self.current_yaw_vel - max_dv:
                    self.current_yaw_vel -= max_dv
                else:
                    self.current_yaw_vel = target_yaw_vel
                
                # 3. 积分得到当前基座需要旋转的总角度
                self.base_yaw += self.current_yaw_vel * dt
                
                # 🌟 4. 核心数学魔法：将起始位姿绕 Base 的 Z 轴旋转 base_yaw 角度 🌟
                cos_a = math.cos(self.base_yaw)
                sin_a = math.sin(self.base_yaw)
                
                # 旋转 X 和 Y 坐标 (Z 坐标保持绝对不变！)
                new_px = cos_a * self.track_start_p[0] - sin_a * self.track_start_p[1]
                new_py = sin_a * self.track_start_p[0] + cos_a * self.track_start_p[1]
                new_pz = self.track_start_p[2] 
                
                # 旋转四元数姿态
                # 生成一个纯绕 Z 轴旋转的四元数
                q_z_rotation = quaternion_about_axis(self.base_yaw, (0, 0, 1))
                # 将旋转四元数应用到初始姿态上
                new_r = quaternion_multiply(q_z_rotation, self.track_start_r)
                
                msg.pose.position.x = new_px
                msg.pose.position.y = new_py
                msg.pose.position.z = new_pz
                msg.pose.orientation.x = new_r[0]
                msg.pose.orientation.y = new_r[1]
                msg.pose.orientation.z = new_r[2]
                msg.pose.orientation.w = new_r[3]
                
                should_publish = True
                
                # 打印当前角速度 (调试用)
                # if abs(self.current_yaw_vel) > 0.001:
                #     print(f"🎯 [炮台模式] 角速度: {self.current_yaw_vel:.4f} rad/s | 总旋转角度: {math.degrees(self.base_yaw):.2f}°")
            # 如果有数据就发布
            if should_publish:
                self.pub.publish(msg)
                
            rate.sleep()

    def start_tracking(self):
        """开启炮台跟踪模式"""
        print("⏳ [Robot] 收到 start_tracking 调用，正在获取起始位姿...")
        p, r = self.get_current_pose()
        if p is not None:
            # 锁定起步坐标！在接下来的追踪中，机械臂其实是以这个位姿为刚体转动的
            self.track_start_p = p.copy()
            self.track_start_r = r.copy() 
            
            self.base_yaw = 0.0       # 角度清零
            self.current_yaw_vel = 0.0 # 速度清零
            self.current_error_y = 0.0 # 误差清零
            
            with self.path_queue.mutex:
                self.path_queue.queue.clear()
            self.is_tracking = True
            print("👁️ [Robot] 成功！炮台跟踪模式已启动！(仅旋转 Joint 1)")
        else:
            print("❌ [Robot] 获取起始位姿失败！")

    def stop_tracking(self):
        """停止跟踪模式"""
        self.is_tracking = False
        print("🛑 [Robot] 视觉 PID 跟踪已停止")

    def update_tracking_error(self, err_x, err_y, err_z=0.0):
        """更新图像误差"""
        if self.is_tracking:
            # 炮台模式只关心水平误差 (左右平移)，我们把 err_y 喂给它
            self.current_error_y = err_y

    def move_to(self, pose, speed=None):
        if pose is None or len(pose) != 7: return
        target_pos = pose[0:3]
        target_rot = pose[3:7]
        path_list = [{'pos': target_pos, 'rot': target_rot}]
        self.execute_path(path_list, speed)

    def is_rotation_reached(self, target_quat, tolerance_deg=1.5):
        """
        严格检查手腕实际姿态是否已经旋转就位（防止平移到了、旋转没到就抢跑）
        :param target_quat: 目标四元数 [qx, qy, qz, qw]
        :param tolerance_deg: 允许的旋转残余角度（度），默认 1.5 度以内算就位
        """
        _, curr_r = self.get_current_pose()
        if curr_r is None: return False
        
        # 计算两个四元数的点积
        dot_product = np.clip(np.dot(curr_r, target_quat), -1.0, 1.0)
        # 计算残余夹角 (弧度)
        residual_angle = 2.0 * math.acos(abs(dot_product))
        residual_deg = math.degrees(residual_angle)
        
        # 如果残余角度小于设定容差，返回 True 放行
        if residual_deg <= tolerance_deg:
            return True
        else:
            # 调试打印，方便你在控制台盯着它扭动
            # print(f"⏳ [手腕对齐中] 旋转残余角度: {residual_deg:.1f}° / 容许值: {tolerance_deg}°")
            return False
        
    # =================================================================
    # 🦾【新增工业级直插接口】强行锁死姿态的纯笛卡尔 Z 轴直下运动
    # =================================================================
    def move_straight_down(self, target_pose, speed=0.02):
        """
        以绝对刚性的姿态锁死模式，垂直直线切入目标点（解决下探晃动碰壁的特效药）
        :param target_pose: 包含 7 个元素的终点列表 [X, Y, Z, qx, qy, qz, qw]
        """
        if target_pose is None or len(target_pose) != 7: return
        
        # 1. 拦截当前一瞬间机械臂最真实的物理姿态，作为接下来的刚体姿态基准
        start_p, start_r = self.get_current_pose()
        if start_p is None: return
        
        # 目标绝对平移点
        target_pos = np.array(target_pose[0:3])
        
        # 2. 清空之前的队列
        with self.path_queue.mutex:
            self.path_queue.queue.clear()
            
        # 3. 强行重采样：位置按 S 曲线走，但姿态全场死死咬住 start_r，拒绝任何 SLERP 扭动！
        total_dist = np.linalg.norm(target_pos - start_p)
        total_time = total_dist / (speed + 1e-6)
        total_time = max(total_time, 0.1) # 兜底
        
        total_steps = int(total_time * self.rate_hz)
        
        for i in range(1, total_steps + 1):
            t = i / float(total_steps)
            smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0
            
            # 位置沿直线连续插入
            curr_p = start_p + (target_pos - start_p) * smooth_t
            
            # 🌟 核心魔法：姿态不做任何插值运算，直接强行克隆起步姿态！
            curr_r = start_r.copy()
            
            self.path_queue.put((curr_p, curr_r))
            
        print(f"🔒 [Pose Locked] 已生成刚性直线物理轨迹：下探 {total_dist*100:.1f}cm，姿态 100% 强行锁死，耗时 {total_time:.2f} 秒")

    # =================================================================
    # 🚀【战术升级】绝对姿态锁死 且 沿夹爪局部 Z 轴斜向直插接口
    # =================================================================
    def move_along_local_z(self, stroke_distance=0.15, speed=0.02):
        """
        顺着当前夹爪指尖指向的局部 Z 轴方向，笔直向前推进（斜向直插，彻底解决倾斜物体碰壁）
        :param stroke_distance: 下探推进的绝对物理距离（米），比如从悬停点到抓取点是 15 厘米，就传 0.15
        :param speed: 推进速度 (m/s)
        """
        # 1. 强行拦截当前悬停点最真实的物理位姿
        start_p, start_r = self.get_current_pose()
        if start_p is None: return
        
        # 2. 数学魔法：利用当前四元数换算出局部 +Z 轴在世界系下的方向矢量
        from scipy.spatial.transform import Rotation as Rot
        R_curr = Rot.from_quat(start_r).as_matrix()
        # 在 Contact-GraspNet/Franka 规范中，旋转矩阵的第三列 [:3, 2] 就是局部 +Z 轴（进近方向）
        approach_vector = R_curr[:3, 2]
        
        # 3. 计算终点的绝对 3D 坐标 (起点位置 + 推进长度 * 方向矢量)
        target_pos = start_p + stroke_distance * approach_vector
        
        # 4. 清空旧队列
        with self.path_queue.mutex:
            self.path_queue.queue.clear()
            
        # 5. S 曲线重采样：位置沿着这条倾斜的“局部射线”连续高精插值，姿态全场刚性锁死！
        total_time = stroke_distance / (speed + 1e-6)
        total_time = max(total_time, 0.1)
        total_steps = int(total_time * self.rate_hz)
        
        for i in range(1, total_steps + 1):
            t = i / float(total_steps)
            smooth_t = (1.0 - math.cos(t * math.pi)) / 2.0
            
            # 沿着局部 Z 轴射线斜向推进
            curr_p = start_p + (target_pos - start_p) * smooth_t
            # 姿态绝对锁死，拒绝任何扭动
            curr_r = start_r.copy()
            
            self.path_queue.put((curr_p, curr_r))
            
        print(f"🔒 [Local Z Injection] 刚性斜插轨迹已生成：沿局部 Z 轴推进 {stroke_distance*100:.1f}cm，姿态死锁，耗时 {total_time:.2f} 秒")

    # ==========================================
    # 🌟 Franka 夹爪控制接口
    # ==========================================
    def open_gripper(self, width=0.08, speed=0.1):
        """
        张开 Franka 夹爪 (最大 0.08 米)
        """
        print(f"🫱 [Franka] 正在张开夹爪 (宽度: {width*100}cm)...")
        
        if hasattr(self, 'move_client') and self.move_client.wait_for_server(rospy.Duration(0.1)):
            goal = MoveGoal()
            goal.width = width
            goal.speed = speed
            self.move_client.send_goal(goal)
            self.move_client.wait_for_result()
            print("   ✅ 夹爪已完全张开。")
        else:
            # 脱机调试时的假动作
            time.sleep(1.0) 
            print("   ✅ [模拟] 夹爪已张开。")


    def close_gripper(self, force=30.0, speed=0.1, expected_width=0.03):
        """
        闭合 Franka 夹爪抓取物品
        :param force: 抓取力 (建议 20N ~ 50N，最大70N)
        :param expected_width: 预计物体的宽度(米)。Franka 会在这个宽度附近启用力控。
        """
        print(f"✊ [Franka] 正在闭合抓取 (目标力: {force}N, 预期宽度: {expected_width*100}cm)...")
        
        if hasattr(self, 'grasp_client') and self.grasp_client.wait_for_server(rospy.Duration(0.1)):
            goal = GraspGoal()
            goal.width = expected_width # 告诉夹爪盒子大概多宽
            goal.speed = speed
            goal.force = force
            
            # Epsilon 是容差。意思是：在 expected_width 的内外各 4cm 范围内感受到阻力，都算抓取成功
            goal.epsilon.inner = 0.04 
            goal.epsilon.outer = 0.04 
            
            self.grasp_client.send_goal(goal)
            self.grasp_client.wait_for_result()
            print("   ✅ 夹爪已施加力控抓紧。")
        else:
            # 脱机调试时的假动作
            time.sleep(1.0) 
            print("   ✅ [模拟] 夹爪已抓紧。")

    def start_servoing(self):
        """启动 3D 平滑递送伺服模式"""
        print("⏳ [Robot] 正在启动 3D 递送伺服模式...")
        p, r = self.get_current_pose()
        if p is not None:
            # 锁定当前坐标和姿态作为起点
            self.servo_target_p = p.copy()
            self.servo_target_r = r.copy() 
            self.current_cmd_p = p.copy()  # 初始化内部平滑指令
            
            with self.path_queue.mutex:
                self.path_queue.queue.clear() # 清空所有旧路径
            
            self.is_servoing = True
            print("🛸 [Robot] 3D 伺服递送模式已启动！(XYZ全向平移)")
        else:
            print("❌ [Robot] 获取起始位姿失败，无法启动伺服！")

    def stop_servoing(self):
        """停止 3D 伺服模式"""
        self.is_servoing = False
        print("🛑 [Robot] 3D 伺服递送已停止")

    def update_servo_target(self, target_pos):
        """
        更新 3D 伺服的目标点 (绝对物理坐标)
        :param target_pos: 机器人坐标系下的目标 [X, Y, Z]
        """
        if self.is_servoing and target_pos is not None:
            self.servo_target_p = np.array(target_pos)
    
    def _wrench_callback(self, msg):
        """实时更新当前末端受力"""
        self.current_wrench = [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ]

    def get_wrench(self):
        """
        获取当前末端受力
        :return: [Fx, Fy, Fz, Tx, Ty, Tz] (单位：牛顿 / 牛·米)
        """
        return self.current_wrench
    
    def _gripper_joint_callback(self, msg):
        """实时更新当前夹爪的真实物理宽度"""
        try:
            # 确保消息中包含手指关节数据
            if 'panda_finger_joint1' in msg.name and 'panda_finger_joint2' in msg.name:
                idx1 = msg.name.index('panda_finger_joint1')
                idx2 = msg.name.index('panda_finger_joint2')
                
                # 两个手指的线性位移相加，即为夹爪的总物理张开宽度 (单位: 米)
                self.current_gripper_width = msg.position[idx1] + msg.position[idx2]
        except ValueError:
            pass

    def get_gripper_width(self):
        """
        获取夹爪当前真实的物理宽度
        :return: 宽度值 (单位：米)。全开约为 0.08m，完全闭合为 0.0m。
        """
        return self.current_gripper_width