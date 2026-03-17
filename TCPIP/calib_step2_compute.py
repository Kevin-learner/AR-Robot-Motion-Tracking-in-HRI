import numpy as np
import cv2
import json

# --- 填入你刚才测得的 D435i 精确内参 ---
K = np.array([
    [606.0288696289062, 0.0,               319.1144714355469],
    [0.0,               605.4163208007812, 245.2572784423828],
    [0.0,               0.0,               1.0              ]
], dtype=np.float64)
D = np.zeros(5) # 你的畸变系数是全0

CHESSBOARD_SIZE = (7, 5)
SQUARE_SIZE = 0.025

def solve_calibration():
    # 1. 加载样本
    try:
        with open("calib_samples.json", "r") as f:
            samples = json.load(f)
    except FileNotFoundError:
        print("❌ 错误：找不到 calib_samples.json，请先运行采集脚本。")
        return

    print(f"⚙️ 正在处理 {len(samples)} 组样本...")

    # 棋盘格局部坐标系中的 3D 点 (物体坐标)
    obj_p = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    R_base_ee = []
    t_base_ee = []
    R_target_cam = []
    t_target_cam = []

    for s in samples:
        # 机器人部分
        mat_ee = np.array(s["base_T_ee"])
        R_base_ee.append(mat_ee[:3, :3])
        t_base_ee.append(mat_ee[:3, 3])

        # 视觉部分 (SolvePnP)
        corners = np.array(s["corners"])
        # 使用你精确的 K 和 D 计算标定板在相机里的位置
        _, rvec, tvec = cv2.solvePnP(obj_p, corners, K, D)
        R_mat, _ = cv2.Rodrigues(rvec)
        R_target_cam.append(R_mat)
        t_target_cam.append(tvec)

    # 2. 核心：手眼标定解算
    # 我们使用的是 Eye-in-Hand (相机在手上)
    # calibrateHandEye 的输入是 Gripper2Base (EE在基座里的位姿)
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_base_ee, t_base_ee,
        R_target_cam, t_target_cam,
        method=cv2.CALIB_HAND_EYE_TSAI # TSAI 算法最稳定
    )

    # 3. 组合成 4x4 矩阵 EE_T_C
    EE_T_C = np.eye(4)
    EE_T_C[:3, :3] = R_cam2ee
    EE_T_C[:3, 3] = t_cam2ee.flatten()

    print("\n" + "★"*40)
    print("🌟 恭喜！手眼标定完成 🌟")
    print("你的专属 EE_T_C 变换矩阵 (请复制到 config.yaml):")
    print("-" * 40)
    
    # 打印成 YAML 喜欢的格式
    formatted_matrix = EE_T_C.tolist()
    for row in formatted_matrix:
        print(f"  {row},")
    
    print("-" * 40)
    print("★"*40)

if __name__ == "__main__":
    solve_calibration()