
import pyrealsense2 as rs

def get_d435i_intrinsics():
    # 初始化 pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    # 启用彩色流 (Color Stream)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 启动
    profile = pipeline.start(config)
    
    # 获取彩色相机的视频流配置文件
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    print("\n" + "="*40)
    print(f"你的 RealSense D435i 精确内参 (640x480):")
    print(f"分辨率: {intrinsics.width} x {intrinsics.height}")
    print(f"焦距 fx: {intrinsics.fx}")
    print(f"焦距 fy: {intrinsics.fy}")
    print(f"光心 ppx: {intrinsics.ppx}")
    print(f"光心 ppy: {intrinsics.ppy}")
    print(f"畸变模型: {intrinsics.model}")
    print(f"畸变系数: {list(intrinsics.coeffs)}")
    print("="*40)

    pipeline.stop()
    return intrinsics

if __name__ == "__main__":
    get_d435i_intrinsics()