import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from scipy.spatial.transform import Rotation as R
import robotPositionListener  # 确保你的项目路径中有这个文件

# --- 标定板配置 ---
CHESSBOARD_SIZE = (7, 4)
SQUARE_SIZE = 0.023  # 米 (25mm)

def main():
    # 1. 初始化机器人监听器
    listener = robotPositionListener.RobotPositionListener(port=5006)
    
    # 2. 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    samples = []
    print(">>> [采集模式] 已启动")
    print(">>> 操作: 移动机械臂，确保棋盘格在画面内且清晰")
    print(">>> 提示: 角度越乱越好，高度有远有近。按 [空格] 采集，按 [ESC] 完成。")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测棋盘格
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            
            show_img = img.copy()
            if ret:

                
                # 亚像素级角点优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 

                cv2.drawChessboardCorners(show_img, CHESSBOARD_SIZE, corners, ret)
            
            cv2.imshow("Calibration Samples Collector", show_img)
            key = cv2.waitKey(1)

            if key == ord('r'): # r采集
                if ret:
                    # 获取机器人实时位姿
                    ee_pos, ee_quat = listener.get_current_pose()
                    if ee_pos is not None:
                        # 转换成 4x4 矩阵 Base_T_EE
                        base_T_ee = np.eye(4)
                        base_T_ee[:3, :3] = R.from_quat(ee_quat).as_matrix()
                        base_T_ee[:3, 3] = ee_pos
                        
                        samples.append({
                            "base_T_ee": base_T_ee.tolist(),
                            "corners": corners.tolist()
                        })
                        print(f"✅ 已保存第 {len(samples)} 组样本! Robot Z: {ee_pos[2]:.3f}")
                    else:
                        print("❌ 错误：无法连接到机器人监听器 (port 5006)")
                else:
                    print("❌ 错误：画面中没找到棋盘格")

            elif key == 27: # ESC 退出
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(samples) > 0:
        with open("calib_samples.json", "w") as f:
            json.dump(samples, f)
        print(f"\n📂 采集完成！共 {len(samples)} 组数据已存入 calib_samples.json")

if __name__ == "__main__":
    main()