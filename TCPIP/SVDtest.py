import numpy as np

def svd_alignment(source_points, target_points):
    """
    使用 SVD 计算最优刚体变换 (target = R * source + t)
    """
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    A = source_points - centroid_source
    B = target_points - centroid_target

    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # 处理手性翻转
    if np.linalg.det(R) < 0:
        print("🚩 Detecting reflection, correcting R matrix...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_target - R @ centroid_source

    T_M = np.identity(4)
    T_M[:3, :3] = R
    T_M[:3, 3] = t
    return T_M

def transform_points(points, T):
    points = np.atleast_2d(points)
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    transformed = (T @ points_hom.T).T
    return transformed[:, :3]

if __name__ == "__main__":
    # HoloLens Ground Truth (Target)
    holo_points = np.array([
        [0.069897, -0.496610, 0.872946],
        [0.168868, -0.503242, 0.885625],
        [0.154676, -0.560000, 0.966724],
        [0.152858, -0.477936, 1.023839],
        [0.167051, -0.421178, 0.942740]
    ])

    # RealSense Original (Source)
    rs_points = np.array([
        [0.37546667, 0.00649437, 0.23309649],
        [0.46547597, 0.05377951, 0.22937151],
        [0.42647126, 0.12445354, 0.17257010],
        [0.39864504, 0.17565104, 0.25501394],
        [0.44767522, 0.10290975, 0.31627357]
    ])

    # --- 关键步骤：拆分数据 ---
    train_ids = [0, 1, 2, 3]
    test_id = 4

    source_train = rs_points[train_ids]
    target_train = holo_points[train_ids]

    source_test = rs_points[test_id]
    target_test = holo_points[test_id]

    # 1. 使用前 4 个点计算 T_M
    print(f"📊 Using points {train_ids} for calibration...")
    T_matrix = svd_alignment(source_train, target_train)

    # 2. 验证训练集本身的误差 (Fitting Error)
    train_preds = transform_points(source_train, T_matrix)
    train_errors = np.linalg.norm(train_preds - target_train, axis=1)
    
    # 3. 验证第 5 个点 (Validation Error)
    test_pred = transform_points(source_test, T_matrix)[0]
    test_error = np.linalg.norm(test_pred - target_test)

    # --- 打印结果 ---
    print("\n" + "="*45)
    print("📌 Calculated Transformation Matrix (T_M):")
    print(np.array2string(T_matrix, separator=', '))
    print("="*45)

    print(f"\n📈 Training Set (Points 0-3) Mean Error: {np.mean(train_errors)*100:.2f} cm")
    print(f"🎯 Validation Set (Point 4) Error: {test_error*100:.2f} cm")
    
    print("\n🔍 Comparison for Point 4:")
    print(f"   Original RS   : {source_test}")
    print(f"   Predicted Holo: {test_pred}")
    print(f"   Ground Truth  : {target_test}")
    print(f"   XYZ Difference: {np.abs(test_pred - target_test) * 100} (cm)")