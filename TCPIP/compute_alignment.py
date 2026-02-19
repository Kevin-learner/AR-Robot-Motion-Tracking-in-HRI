import numpy as np
import vtk
import re


def vtkmatrix_to_numpy(matrix):
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m


def numpyArr2vtkPoints(npArray):
    vtkpoints = vtk.vtkPoints()
    for i in range(npArray.shape[0]):
        vtkpoints.InsertNextPoint(npArray[i])
    return vtkpoints


def initialAlignment(source, target):
    len_source = source.shape[0]
    len_target = target.shape[0]

    if len_source != len_target or len_source < 3:
        raise ValueError("At least three pairs of matching points are required, and the number of sources and targets must be the same.")

    tmpSource = numpyArr2vtkPoints(source)
    tmpTarget = numpyArr2vtkPoints(target)

    landmarkTransform = vtk.vtkLandmarkTransform()
    landmarkTransform.SetSourceLandmarks(tmpSource)
    landmarkTransform.SetTargetLandmarks(tmpTarget)
    landmarkTransform.SetModeToRigidBody()
    landmarkTransform.Update()

    matrixnp = vtkmatrix_to_numpy(landmarkTransform.GetMatrix())
    print("Rigid transformation matrix（4x4）:")
    print(matrixnp)
    return matrixnp


def parse_realsense_corners(file_path):
    corners = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'XYZ=' in line:
                parts = line.strip().split('XYZ=')[1]
                x, y, z = map(float, parts.strip('()').split(','))
                z = -z
                corners.append([x, y, z])
    return np.array(corners)

def parse_ROS_points(file_path):
    """
    修改后：解析录制的 calibration_data_recorded.txt
    格式示例：
    1: [0.45, 0.12, 0.33]
    2: [0.55, 0.12, 0.33]
    """
    point_robot = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 1. 既然录制格式是 "1: [x,y,z]"，这就是标准的 YAML 格式
            # 使用 safe_load 直接转为 Python 字典 {1: [...], 2: [...]}
            data_map = yaml.safe_load(f)
            
        if not data_map:
            print(f"⚠️ 文件为空: {file_path}")
            return []

        # 2. [关键步骤] 按 Key (序号) 进行排序
        # 确保 point_robot[0] 对应的是 Index 1，而不是录制时的随机顺序
        sorted_keys = sorted(data_map.keys())
        
        print(f"   -> Detected points indices: {sorted_keys}")

        for key in sorted_keys:
            coords = data_map[key]
            # 确保数据是 3 个 float
            if len(coords) == 3:
                point_robot.append([float(coords[0]), float(coords[1]), -float(coords[2])])
            else:
                print(f"   ⚠️ Point {key} has invalid format: {coords}")

    except Exception as e:
        print(f"❌ 解析录制文件失败: {e}")
        # 如果 YAML 解析失败，作为备用方案，尝试简单的正则解析 (防止文件格式有一点点偏差)
        # 备用逻辑：匹配方括号内的数字
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            point_robot = []
            for line in lines:
                # 寻找 [num, num, num]
                match = re.search(r"\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]", line)
                if match:
                    point_robot.append([float(match.group(1)), float(match.group(2)), -float(match.group(3))])
        except:
            pass

    return point_robot


def transform_points(points, T):

    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    transformed = (T @ points_hom.T).T
    return transformed[:, :3]


def align_with_realsense(holo_target_points, realsense_txt_path):
    #source_points = parse_realsense_corners(realsense_txt_path)
    source_points = np.array(parse_ROS_points(realsense_txt_path))

    if source_points.shape[0] != holo_target_points.shape[0]:
        raise ValueError(f"Inconsistent number of points：source {source_points.shape[0]} vs target {holo_target_points.shape[0]}")

    train_ids = [0, 1, 2, 3]
    test_ids = [4]
    source_train = source_points[train_ids]
    target_train = holo_target_points[train_ids]

    source_test = source_points[test_ids]
    target_test = holo_target_points[test_ids]

    T = initialAlignment(source_train, target_train)

    pred_test = transform_points(source_test, T)

    print("\n📌 Verification point comparison (Realsense original - transformed result vs HoloLens Ground Truth）:")
    for i, idx in enumerate(test_ids):
        print(f"\n🔹  index {idx}")
        print(f"  Realsense original  : {source_test[i]}")
        print(f"  Prediction point after transformation  : {pred_test[i]}")
        print(f"  HoloLens Ground Truth: {target_test[i]}")

    return T

if __name__ == "__main__":
    # Example usage
    holo_points = np.array([
        [0.069897, -0.496610, -0.872946],
        [0.168868, -0.503242, -0.885625],
        [0.154676, -0.560000, -0.966724],
        [0.152858, -0.477936, -1.023839],
        [0.167051, -0.421178, -0.942740]
    ])
    source_file = "calibration_data_recorded.txt"
    T_matrix = align_with_realsense(holo_points, source_file)
    print("Final Transformation Matrix:\n", T_matrix)