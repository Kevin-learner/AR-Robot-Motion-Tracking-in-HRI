import open3d as o3d

# 替换为你想要查看的文件名
filename = "object_id_00.pcd"

print(f"正在读取点云文件: {filename}")
pcd = o3d.io.read_point_cloud(filename)

# 打印点云的基础信息（点数等）
print(pcd)

# 打开可视化窗口
# 鼠标左键拖拽旋转，鼠标右键拖拽平移，滚轮缩放
o3d.visualization.draw_geometries([pcd], 
                                  window_name=f"点云查看器 - {filename}",
                                  width=1024, 
                                  height=768,
                                  point_show_normal=False)