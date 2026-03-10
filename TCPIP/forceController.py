import rospy
from geometry_msgs.msg import PoseStamped
import time

class ForceController:
    def __init__(self):
        # 初始化 ROS 节点 (如果你的主程序还没初始化的话)
        # rospy.init_node('unity_force_controller', anonymous=True)
        
        # 1. 创建发布者，连接到阻抗控制器的平衡点话题
        self.pose_pub = rospy.Publisher(
            '/cartesian_impedance_example_controller/equilibrium_pose', 
            PoseStamped, 
            queue_size=10
        )
        
        # 2. 设定弹簧刚度 (必须与你在 rqt_reconfigure 中设置的值一致！)
        self.K_trans = 1000.0 # N/m (牛顿每米)

    def execute_force_path(self, path_with_force):
        """
        执行带有力设定的路径
        :param path_with_force: 包含 'pos', 'rot', 'force' 的字典列表
        """
        rate = rospy.Rate(100) # 控制频率 100Hz
        
        print(f"🚀 开始执行力控轨迹，总点数: {len(path_with_force)}")
        
        for point in path_with_force:
            if rospy.is_shutdown():
                break
                
            u_pos = point['pos']     # [x, y, z]
            u_rot = point['rot']     # [qx, qy, qz, qw]
            target_force = point['force'] # 例如 5.0 或 2.0
            
            # --- 核心：计算 Z 轴下压偏移量 ---
            # 偏移量 = 目标力 / 刚度
            z_offset = target_force / self.K_trans 
            
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "panda_link0" # Franka 的基座坐标系
            
            # X, Y 保持精确跟随 Unity
            msg.pose.position.x = u_pos[0]
            msg.pose.position.y = u_pos[1]
            
            # Z 轴施加虚拟弹簧偏移 (向下压)
            msg.pose.position.z = u_pos[2] - z_offset 
            
            # 姿态保持精确跟随 Unity
            msg.pose.orientation.x = u_rot[0]
            msg.pose.orientation.y = u_rot[1]
            msg.pose.orientation.z = u_rot[2]
            msg.pose.orientation.w = u_rot[3]
            
            # 发布目标位姿
            self.pose_pub.publish(msg)
            
            rate.sleep()
            
        print("✅ 轨迹执行完毕！")

# ==========================================
# 在你的 TCP 接收代码中的调用方式：
# force_robot = ForceController()
# force_robot.execute_force_path(path_with_force)
# ==========================================