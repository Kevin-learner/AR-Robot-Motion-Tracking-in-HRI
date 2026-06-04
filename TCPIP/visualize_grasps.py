import open3d as o3d
import numpy as np
import os

# ==========================================
# 🌟 核心修改区：创建自定义大号且旋转的夹爪
# ==========================================
def create_custom_gripper_marker(color=[0, 1, 0], scale=1.0, rotate_90_z=False):
    """
    基于 Franka Gripper 标准进行缩放和旋转。
    
    参数:
    - color: 线框颜色
    - scale: 缩放比例 (例如 2.0 表示变大一倍)
    - rotate_90_z: 是否绕局部 Z 轴旋转 90 度 (默认开启)
    """
    # 1. Franka Emika 标准尺寸 (单位: 米)
    std_depth = 0.08  # 指尖深度
    std_width = 0.08  # 最大开合宽度
    
    # 2. 应用缩放 (变大一些)
    # 🌟 你可以手动把默认 scale 改成 1.5 或 2.0
    half_w = (std_width * scale) / 2.0
    depth = std_depth * scale
    
    # 3. 原始 Z 轴正向生长点位 (对齐 AI 标准：掌心在 [0,0,0])
    pts = np.array([
        [0, 0, -0.02],                     # 0: 尾部连接法兰
        [0, 0, 0],                         # 1: 🌟 掌心中心(原点)
        [-half_w, 0, 0],                   # 2: 左关节
        [half_w, 0, 0],                    # 3: 右关节
        [-half_w, 0, depth],               # 4: 左指尖
        [half_w, 0, depth]                 # 5: 右指尖
    ])
    
    # 连线关系
    lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]
    
    final_pts = pts
    
    # 🌟 4. 数学魔法：绕 Z 轴旋转 90 度
    if rotate_90_z:
        # 旋转矩阵 Rz = [[cos,-sin,0],[sin,cos,0],[0,0,1]]
        # 这里直接对所有点进行坐标交换操作：NewX = -OldY, NewY = +OldX, NewZ = OldZ
        # 结果表现为原开合方向在 X 轴，现变为在 Y 轴开合
        rotated_pts = np.zeros_like(pts)
        for i in range(pts.shape[0]):
            rotated_pts[i, 0] = -pts[i, 1] # NewX = -OldY
            rotated_pts[i, 1] = pts[i, 0]  # NewY = +OldX
            rotated_pts[i, 2] = pts[i, 2]  # NewZ
        final_pts = rotated_pts
        
    # 5. 生成 Open3D 对象
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(final_pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    # 可选：添加局部坐标系观察 (红:X, 绿:Y, 蓝:Z)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03*scale, origin=[0, 0, 0])
    
    return [line_set, mesh_frame]

# ==========================================
# 主可视化逻辑 (保持干净，不做任何矩阵乘法)
# ==========================================
def visualize_scene(pcd_path, grasps_path, top_n=15, gripper_scale=1.5):
    """
    gripper_scale: 此处可以修改夹爪缩放倍数
    """
    geometries = []
    
    # 1. 加载点云 (确保文件名为 scanned_scene.pcd)
    if not os.path.exists(pcd_path):
        print(f"❌ 找不到点云文件: {pcd_path}")
        return
    pcd = o3d.io.read_point_cloud(pcd_path)
    geometries.append(pcd)
    
    # 2. 加载基座系抓取矩阵 (ensure fileName is test_output_grasps.npy)
    if not os.path.exists(grasps_path):
        print(f"❌ 找不到抓取结果文件: {grasps_path}")
        return
    grasps = np.load(grasps_path)
    
    display_count = min(top_n, grasps.shape[0])
    print(f"👉 准备渲染前 {display_count} 个抓取点 (应用自定义大号旋转夹爪)")
    
    for i in range(display_count):
        pose_matrix = grasps[i] # 干净纯粹，直接 transform
        
        color = [1, 0, 0] if i == 0 else [0, 1, 0] # 最高分红色，其余绿色
        
        # 🌟 调用我们自定义的模型，scale 设为 1.5 倍
        gripper_geoms = create_custom_gripper_marker(color=color, scale=gripper_scale, rotate_90_z=False)
        
        for geom in gripper_geoms:
            geom.transform(pose_matrix)
            geometries.append(geom)

    # 3. 绝对世界坐标系
    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(global_frame)

    o3d.visualization.draw_geometries(geometries, window_name="4060 brain -> Rotated Gripper可视化")

if __name__ == "__main__":
    # 你可以修改 gripper_scale 来改变大小
    visualize_scene("scanned_scene.pcd", "test_output_grasps.npy", top_n=15, gripper_scale=1.5)