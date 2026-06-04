import requests
import io
import numpy as np

def request_grasps_from_graspnet(pcd_points, pcd_colors, cam_K_matrix, server_ip="100.116.99.44", port=5000):
    """
    将真实点云、颜色和内参，包装成 Contact-GraspNet 官方字典格式，发给 4060
    """
    print(f"\n🌐 [网络] 准备打包字典发送给 4060 (IP: {server_ip}) ...")

    # 如果没传 K 矩阵，用一套默认的 D435i 近似内参防报错
    if cam_K_matrix is None:
        cam_K_matrix = np.array([
            [615.0, 0.0, 320.0], 
            [0.0, 615.0, 240.0], 
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    # 2. 构造官方所需字典
    mock_data = {
        'xyz': pcd_points,          
        'rgb': pcd_colors, 
        'depth': np.zeros((10, 10)), 
        'segmap': np.zeros((10, 10)),
        'K': cam_K_matrix
    }

    # 3. 序列化为内存文件 (非常重要：加上 allow_pickle=True 并包成 object 数组)
    memfile = io.BytesIO()
    np.save(memfile, np.array([mock_data], dtype=object)) 
    memfile.seek(0)
    
    # 4. 发送 HTTP POST 请求
    url = f"http://{server_ip}:{port}/predict_grasp"
    try:
        print(f"🚀 正在上传数据 (点云数: {len(pcd_points)})...")
        # 超时时间设长一点，网络推理需要时间
        response = requests.post(url, data=memfile.read(), timeout=20.0) 
        
        if response.status_code == 200:
            result_file = io.BytesIO(response.content)
            grasps = np.load(result_file)

            save_path = "test_output_grasps.npy"
            np.save(save_path, grasps)
            
            print(f"🎉 成功拿到 {grasps.shape[0]} 个 AI 抓取姿态！")
            return grasps
        else:
            print(f"❌ 4060 服务端报错，状态码: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时！4060 算得太慢或网络卡顿。")
    except Exception as e:
        print(f"❌ 连接 4060 失败: {e}")
        
    return None