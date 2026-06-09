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
    
    # 4. 数学魔法：绕 Z 轴旋转 90 度
    if rotate_90_z:
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
# 🌟 可视化逻辑 (新增 highlight_idx 参数)
# ==========================================
def visualize_scene(pcd_path, grasps_path, top_n=15, gripper_scale=1.0, highlight_idx=0):
    """
    highlight_idx: 想要高亮的抓取索引。被选中的会变红，其他的变灰作陪衬。
    """
    geometries = []
    
    # 1. 加载点云
    if not os.path.exists(pcd_path):
        print(f"❌ 找不到点云文件: {pcd_path}")
        return
    pcd = o3d.io.read_point_cloud(pcd_path)
    geometries.append(pcd)
    
    # 2. 加载基座系抓取矩阵
    if not os.path.exists(grasps_path):
        print(f"❌ 找不到抓取结果文件: {grasps_path}")
        return
    grasps = np.load(grasps_path)
    
    display_count = min(top_n, grasps.shape[0])
    print(f"👉 准备渲染前 {display_count} 个抓取点 (高亮索引: {highlight_idx})")
    
    for i in range(display_count):
        pose_matrix = grasps[i]
        
        # 🌟 核心高亮逻辑
        if i == highlight_idx:
            color = [1, 0, 0]  # 高亮为红色
            current_scale = gripper_scale * 1.2 # 高亮的夹爪可以稍微再放大一点，更醒目
            print(f"   🎯 正在高亮渲染抓取 [{i}]")
        else:
            color = [0.6, 0.6, 0.6]  # 未选中的全部变成暗灰色作陪衬
            current_scale = gripper_scale
            
        gripper_geoms = create_custom_gripper_marker(color=color, scale=current_scale, rotate_90_z=False)
        
        for geom in gripper_geoms:
            geom.transform(pose_matrix)
            geometries.append(geom)

    # 3. 绝对世界坐标系
    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(global_frame)

    o3d.visualization.draw_geometries(geometries, window_name=f"抓取可视化 - 高亮索引: {highlight_idx}")

if __name__ == "__main__":
    pcd_file = "scanned_scene.pcd"
    grasps_file = "test_output_grasps.npy"
    
    # 检查文件是否存在，给出友好的提示
    if not os.path.exists(grasps_file):
        print(f"⚠️ 找不到文件 {grasps_file}，请确认路径。")
    else:
        # 获取一下总数，方便提示用户
        total_grasps = np.load(grasps_file).shape[0]
        
        # 🌟 在终端请求用户输入想要高亮的索引
        user_input = input(f"🤔 发现了 {total_grasps} 个抓取点。请输入你想高亮的索引 (0~{total_grasps-1})，直接回车默认看第 0 个: ")
        
        try:
            target_idx = int(user_input.strip()) if user_input.strip() != "" else 0
            # 越界保护
            if target_idx < 0 or target_idx >= total_grasps:
                print(f"⚠️ 索引 {target_idx} 越界！已自动修正为 0。")
                target_idx = 0
        except ValueError:
            print("⚠️ 输入无效！已自动使用默认索引 0。")
            target_idx = 0

        # 执行可视化
        visualize_scene(pcd_file, grasps_file, top_n=15, gripper_scale=1.5, highlight_idx=target_idx)