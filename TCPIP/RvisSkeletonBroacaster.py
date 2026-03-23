import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class RVizSkeletonBroadcaster:
    def __init__(self, frame_id="base_link", topic_name="/human_skeleton"):
        """
        初始化 RViz 骨架广播器 (已内置 17 关键点连线拓扑)
        :param frame_id: 机械臂基座坐标系名称 (默认 base_link)
        :param topic_name: ROS 话题名称
        """
        # 🌟 直接在类内部定义骨架的连线规则 (COCO 17点格式)
        self.skeleton_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (0, 5), (0, 6),
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        self.frame_id = frame_id
        self.publisher = None

        # 优雅地初始化 ROS 节点和 Publisher
        try:
            rospy.init_node('skeleton_vision_node', anonymous=True, disable_signals=True)
            self.publisher = rospy.Publisher(topic_name, Marker, queue_size=10)
            print(f"✅ RViz 骨架广播类已启动 (话题: {topic_name}, 坐标系: {frame_id})")
        except rospy.ROSException:
            print("⚠️ 未检测到 ROS master，RViz 发布功能已静默关闭。")

    def publish(self, skeleton_coords_robot):
        """
        处理点云并一键发布到 RViz
        :param skeleton_coords_robot: 转换到基座坐标系下的 Nx3 骨架点
        """
        if self.publisher is None or len(skeleton_coords_robot) == 0:
            return

        body_coords = skeleton_coords_robot[:17]
        if hasattr(body_coords, 'tolist'):
            body_coords = body_coords.tolist()

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "skeleton"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02  
        
        # 绿色画笔
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0

        for pair in self.skeleton_pairs:
            idx1, idx2 = pair
            if idx1 < len(body_coords) and idx2 < len(body_coords):
                pt1, pt2 = body_coords[idx1], body_coords[idx2]
                
                if tuple(pt1) == (0.0, 0.0, 0.0) or tuple(pt2) == (0.0, 0.0, 0.0): 
                    continue
                
                p1 = Point(x=pt1[0], y=pt1[1], z=pt1[2])
                p2 = Point(x=pt2[0], y=pt2[1], z=pt2[2])
                marker.points.extend([p1, p2])

        self.publisher.publish(marker)