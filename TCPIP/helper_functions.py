import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import re
import numpy as np
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped, Quaternion, WrenchStamped, Pose
from franka_msgs.msg import FrankaState
import copy
import time
import math
from std_msgs.msg import Header, Bool
import matplotlib.pyplot as plt
import open3d as o3d
import apriltag
from enum import IntEnum

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def save_and_plot_object(object_points):
    """Save the object points as an XYZ file and create a 3D plot."""
    # Convert object points to a NumPy array
    object_points = np.array(object_points)

    # Save points to an XYZ file
    save_path = os.path.abspath(os.path.dirname(__file__)) + '/object_points.xyz'
    np.savetxt(save_path, object_points, delimiter=',', fmt='%.6f')

    # Plot the 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def process_frame(depth_frame, depth_intrinsics, bounding_box):
    """Process the depth frame to segment the object of interest."""
    point_cloud = create_point_cloud(depth_frame, depth_intrinsics)
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.008)

    # # Perform statistical outlier removal
    # cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)
    # point_cloud = point_cloud.select_by_index(ind)

    if len(point_cloud.points) >= 3:
        # Segment the plane (background)
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=500)
        a, b, c, d = plane_model  # The coefficients of the plane equation

        # Filter out points that are part of the plane
        outlier_indices = set(range(len(point_cloud.points))) - set(inliers)
        outlier_points = np.asarray(point_cloud.points)[list(outlier_indices)]

        # Apply distance threshold and filter out points below the plane
        points = []
        for p in outlier_points:
            # Calculate the point's position relative to the plane
            value = a * p[0] + b * p[1] + c * p[2] + d
            if value < -0.01:
                points.append(p)
        
        # Project points to image
        mask, points = project_points_to_image(points, depth_intrinsics, (480, 640), bounding_box)

        # Transform points to base frame
        O_T_EE = get_O_T_EE()  # Get the transformation from base to EE
        EE_T_C = get_EE_T_C()  # Get the transformation from EE to camera
        
        if len(points) > 0:
            points_in_base_frame = transform_points_to_base_frame(points, O_T_EE, EE_T_C)
        else:
            points_in_base_frame = []
    else:
        mask = np.zeros((480, 640))
        points_in_base_frame = []

    return mask, points_in_base_frame

def create_point_cloud(depth_frame, intrinsics):
    """Convert a depth frame to a point cloud."""
    depth_image = np.asanyarray(depth_frame.get_data())
    height, width = depth_image.shape

    # Convert depth image to Open3D format
    depth_image_o3d = o3d.geometry.Image(depth_image)
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, 
        intrinsics.fx, intrinsics.fy, 
        intrinsics.ppx, intrinsics.ppy
    )
    
    # Generate the point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, camera_intrinsics)
    
    return point_cloud

def project_points_to_image(points, intrinsics, image_shape, bounding_box):
    """Project 3D points onto a 2D image plane, optionally constrained by a bounding box.
       Returns both the mask and the filtered points within the bounding box."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    filtered_points = []

    min_x, max_x, min_y, max_y = bounding_box

    for point in points:
        # Project 3D point to 2D using camera intrinsics
        if point[2] != 0:  # Avoid division by zero
            x = int((point[0] * intrinsics.fx / point[2]) + intrinsics.ppx)
            y = int((point[1] * intrinsics.fy / point[2]) + intrinsics.ppy)

            # Check if the projected point is within the bounding box
            if min_x <= x < max_x and min_y <= y < max_y:
                mask[y, x] = 255
                filtered_points.append(point)

    return mask, filtered_points

def transform_points_to_base_frame(points, O_T_EE, EE_T_C):
    """Transform points from the camera frame to the base frame."""
    # Combine transformations: O_T_C = O_T_EE * EE_T_C
    O_T_C = np.dot(O_T_EE, EE_T_C)
    
    # Convert points from 3D to 4D (homogeneous coordinates)
    points = np.array(points)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply the transformation to each point
    points_in_base_frame = np.dot(O_T_C, points_homogeneous.T).T
    
    # Convert back to 3D by removing the homogeneous coordinate
    points_in_base_frame = points_in_base_frame[:, :3]
    
    return points_in_base_frame

def detect_apriltags(gray_image):
    # Ensure the image is in the correct format
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)

    # Initialize AprilTag detector
    detector = apriltag.Detector()  # Use the Detector class
    detector.add_tag_family("tag16h5")

    # Detect AprilTags in the grayscale image
    results = detector.detect(gray_image)

    detected_tags = []
    for tag in results:
        if tag.tag_id in [1, 2, 3, 4]:  # Use Tag IDs 1, 2, 3, and 4
            detected_tags.append(tag.corners)  # Append the corners of each detected tag

    return detected_tags

# Return bounding box using the extreme coordinates of the detected tags
def get_bounding_box(detected_tags):
    if len(detected_tags) == 4:
        # Collect all corners from detected tags
        corners = np.array([tag_corner[0] for tag_corner in detected_tags], dtype=int)

        # Get minimum and maximum coordinates using NumPy
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)

        return [min_x, max_x, min_y, max_y]
    return [0, 640, 0, 480]

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def move_to_pose(end_pose, duration=3):
    """
    Moves the robot's end-effector smoothly from a start pose to an end pose with a specified and duration.

    Args:
        end_pose (Pose): The target pose of the robot's end-effector in the base frame.
        duration (float, optional): The maximum duration of the movement in seconds.

    Returns:
        Pose: The final pose of the robot's end-effector after the motion. 
    """
    
    # Initialize publishers
    pose_pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)

    # Get the current pose of the end-effector
    start_pose = get_current_pose().pose
    
    # Calculate the Euclidean distance between the start and end poses
    distance = np.sqrt(
        (end_pose.position.x - start_pose.position.x) ** 2 +
        (end_pose.position.y - start_pose.position.y) ** 2 +
        (end_pose.position.z - start_pose.position.z) ** 2
    )

    max_speed = 0.02  # meters per second
    min_duration = distance / max_speed
    duration = max(min_duration, duration)

    step_length = 1e-4
    steps = max(int(distance / step_length), 1)

    # Calculate the control loop frequency based on the duration
    frequency = steps / duration
    rate = rospy.Rate(frequency)

    # Interpolation and movement loop
    desired_pose = PoseStamped()
    desired_pose.header.frame_id = "panda_link0"
    desired_pose.pose = copy.deepcopy(start_pose)

    for i in range(steps):
        alpha = float(i) / float(steps)

        # Position interpolation
        desired_pose.pose.position.x = (1 - alpha) * start_pose.position.x + alpha * end_pose.position.x
        desired_pose.pose.position.y = (1 - alpha) * start_pose.position.y + alpha * end_pose.position.y
        desired_pose.pose.position.z = (1 - alpha) * start_pose.position.z + alpha * end_pose.position.z

        # Quaternion (Orientation) interpolation
        start_quat = np.array([start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w])
        end_quat = np.array([end_pose.orientation.x, end_pose.orientation.y, end_pose.orientation.z, end_pose.orientation.w])
        desired_quat = slerp(start_quat, end_quat, alpha)
        desired_quat /= np.linalg.norm(desired_quat)

        desired_pose.pose.orientation.x = desired_quat[0]
        desired_pose.pose.orientation.y = desired_quat[1]
        desired_pose.pose.orientation.z = desired_quat[2]
        desired_pose.pose.orientation.w = desired_quat[3]

        # Update timestamp and publish
        desired_pose.header.stamp = rospy.Time.now()
        pose_pub.publish(desired_pose)

        rate.sleep()

    return end_pose

def get_current_pose():
    """
    Retrieves the current pose of the robot's end-effector, including position and orientation, and returns the corresponding rotation matrix.

    Returns:
        tuple:
            - PoseStamped: The current pose of the robot's end-effector in the base frame.
            - numpy.ndarray: The 3x3 rotation matrix representing the orientation of the end-effector.

    Notes:
        - The function waits for the latest message from the `/franka_state_controller/franka_states` topic to retrieve the current state.
        - The position is extracted from the transformation matrix `O_T_EE`.
        - The rotation matrix is converted to a quaternion, which is then used to populate the orientation of the `PoseStamped` message.
    """

    msg = rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
    current_pose = PoseStamped()
    current_pose.header = msg.header
    current_pose.pose.position.x = msg.O_T_EE[12]
    current_pose.pose.position.y = msg.O_T_EE[13]
    current_pose.pose.position.z = msg.O_T_EE[14]

    # Extract rotation matrix from O_T_EE    
    R = np.array([
        [msg.O_T_EE[0], msg.O_T_EE[4], msg.O_T_EE[8]],
        [msg.O_T_EE[1], msg.O_T_EE[5], msg.O_T_EE[9]],
        [msg.O_T_EE[2], msg.O_T_EE[6], msg.O_T_EE[10]]
    ])

    # Convert rotation matrix to quaternion
    q = quaternion_from_matrix(R)

    current_pose.pose.orientation.x = q[0]
    current_pose.pose.orientation.y = q[1]
    current_pose.pose.orientation.z = q[2]
    current_pose.pose.orientation.w = q[3]

    # determine_rotation(R)
    # return copy.deepcopy(current_pose), R
    return copy.deepcopy(current_pose)

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def quaternion_from_matrix(matrix):
    """Convert a 3x3 rotation matrix to a quaternion."""

    m = np.eye(4)
    m[:3, :3] = matrix[:3, :3]
    q = tf_trans.quaternion_from_matrix(m)
    q = q / np.linalg.norm(q)
    return q

def plot_rotation_matrix(R):
    # Plotting the rotation matrix as coordinate axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin
    origin = np.array([0, 0, 0])

    # Define the axes using the rotation matrix
    x_axis = R[:, 0]  # New x-axis
    y_axis = R[:, 1]  # New y-axis
    z_axis = R[:, 2]  # New z-axis

    # Plot the original coordinate frame (RGB: XYZ)
    ax.quiver(*origin, 1, 0, 0, color='r', length=0.5, normalize=True, label='Original X-axis')
    ax.quiver(*origin, 0, 1, 0, color='g', length=0.5, normalize=True, label='Original Y-axis')
    ax.quiver(*origin, 0, 0, 1, color='b', length=0.5, normalize=True, label='Original Z-axis')

    # Plot the new rotated coordinate frame
    ax.quiver(*origin, *x_axis, color='c', length=0.5, normalize=True, label='Rotated X-axis')
    ax.quiver(*origin, *y_axis, color='m', length=0.5, normalize=True, label='Rotated Y-axis')
    ax.quiver(*origin, *z_axis, color='y', length=0.5, normalize=True, label='Rotated Z-axis')

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotation Matrix Visualization')

    # Show legend
    ax.legend()

    # Show the plot
    plt.show()

def get_current_pose():
    """
    Retrieves the current pose of the robot's end-effector, including position and orientation, and returns the corresponding rotation matrix.

    Returns:
        tuple:
            - PoseStamped: The current pose of the robot's end-effector in the base frame.
            - numpy.ndarray: The 3x3 rotation matrix representing the orientation of the end-effector.

    Notes:
        - The function waits for the latest message from the `/franka_state_controller/franka_states` topic to retrieve the current state.
        - The position is extracted from the transformation matrix `O_T_EE`.
        - The rotation matrix is converted to a quaternion, which is then used to populate the orientation of the `PoseStamped` message.
    """

    msg = rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
    current_pose = PoseStamped()
    current_pose.header = msg.header
    current_pose.pose.position.x = msg.O_T_EE[12]
    current_pose.pose.position.y = msg.O_T_EE[13]
    current_pose.pose.position.z = msg.O_T_EE[14]

    # Extract rotation matrix from O_T_EE    
    R = np.array([
        [msg.O_T_EE[0], msg.O_T_EE[4], msg.O_T_EE[8]],
        [msg.O_T_EE[1], msg.O_T_EE[5], msg.O_T_EE[9]],
        [msg.O_T_EE[2], msg.O_T_EE[6], msg.O_T_EE[10]]
    ])

    # Convert rotation matrix to quaternion
    q = quaternion_from_matrix(R)

    current_pose.pose.orientation.x = q[0]
    current_pose.pose.orientation.y = q[1]
    current_pose.pose.orientation.z = q[2]
    current_pose.pose.orientation.w = q[3]

    # determine_rotation(R)
    # return copy.deepcopy(current_pose), R
    return copy.deepcopy(current_pose)

def pose_to_matrix(pose):
    """Converts a Pose to a 4x4 transformation matrix."""
    q = pose.orientation
    t = pose.position

    q_array = np.array([q.x, q.y, q.z, q.w])

    # Check if the quaternion is valid (not near zero)
    if np.dot(q_array, q_array) < np.finfo(q_array.dtype).eps:
        return np.eye(4)

    # Quaternion to rotation matrix
    R = np.array([
        [1 - 2*q.y**2 - 2*q.z**2, 2*q.x*q.y - 2*q.z*q.w, 2*q.x*q.z + 2*q.y*q.w],
        [2*q.x*q.y + 2*q.z*q.w, 1 - 2*q.x**2 - 2*q.z**2, 2*q.y*q.z - 2*q.x*q.w],
        [2*q.x*q.z - 2*q.y*q.w, 2*q.y*q.z + 2*q.x*q.w, 1 - 2*q.x**2 - 2*q.y**2]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T

def is_valid_rotation_matrix(R):
    return np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0)

def correct_rotation_matrix(R):
    U, _, Vt = np.linalg.svd(R)
    return np.dot(U, Vt)

def matrix_to_pose(matrix):
    pose = Pose()
    R = matrix[:3, :3]

    # Ensure R is a valid rotation matrix
    if not is_valid_rotation_matrix(R):
        R = correct_rotation_matrix(R)
    
    # Calculate quaternion
    q = Quaternion()
    trace = np.trace(R)
    if trace > 0:
        q.w = np.sqrt(1 + trace) / 2
        q.x = (R[2, 1] - R[1, 2]) / (4 * q.w)
        q.y = (R[0, 2] - R[2, 0]) / (4 * q.w)
        q.z = (R[1, 0] - R[0, 1]) / (4 * q.w)
    else:
        # Handle negative or near-zero trace cases carefully
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            q.x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
            q.w = (R[2, 1] - R[1, 2]) / (4 * q.x)
            q.y = (R[0, 1] + R[1, 0]) / (4 * q.x)
            q.z = (R[0, 2] + R[2, 0]) / (4 * q.x)
        elif i == 1:
            q.y = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
            q.w = (R[0, 2] - R[2, 0]) / (4 * q.y)
            q.x = (R[0, 1] + R[1, 0]) / (4 * q.y)
            q.z = (R[1, 2] + R[2, 1]) / (4 * q.y)
        else:
            q.z = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
            q.w = (R[1, 0] - R[0, 1]) / (4 * q.z)
            q.x = (R[0, 2] + R[2, 0]) / (4 * q.z)
            q.y = (R[1, 2] + R[2, 1]) / (4 * q.z)

    pose.orientation = q
    
    # Translation
    t = matrix[:3, 3]
    pose.position.x = t[0]
    pose.position.y = t[1]
    pose.position.z = t[2]
    
    return pose

def apply_EE_T_US(payload_pose):
    """
    Converts the desired payload pose in end-effector frame.

    Args:
        payload_pose (Pose): The target pose of the payload in the ee frame.

    Returns:
        Pose: The corresponding pose of the end-effector in the ee frame.
    """

    EE_T_US = get_EE_T_US()

    # Transform the end payload pose to the required end-effector pose
    payload_matrix = pose_to_matrix(payload_pose)
    ee_matrix = np.dot(payload_matrix, EE_T_US)
    ee_pose = matrix_to_pose(ee_matrix)

    return ee_pose

def apply_US_T_EE(ee_pose):
    """
    Converts the desired end-effector pose in payload frame.
    
    Args:
        ee_pose (Pose): The pose of the end-effector in the payload frame.
        
    Returns:
        PoseStamped: The corresponding pose of the payload in the payload frame.
    """

    # Get the transformation matrix from payload to end-effector (inverse of ee to payload)
    EE_T_US = get_EE_T_US()

    # Calculate the inverse of the transformation matrix to go from end-effector to payload
    US_T_EE = np.linalg.inv(EE_T_US)

    # Convert ee_pose to a transformation matrix
    ee_matrix = pose_to_matrix(ee_pose)
    payload_matrix = np.dot(ee_matrix, US_T_EE)
    payload_pose = matrix_to_pose(payload_matrix)

    return payload_pose

def slerp(q0, q1, t, threshold=0.1):
    """Spherical Linear Interpolation between two quaternions."""

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.75:
        result = q0 + t * (q1 - q0)
        result /= np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)

    result = q0 * np.cos(theta) + q2 * np.sin(theta)
    result /= np.linalg.norm(result)
    
    # Directly set the final quaternion if the interpolation is near completion
    if t > 1.0 - threshold:
        return q1
    
    return result

def calculate_ee_offset(start_pose, distance):
    """
    Calculate the target position a certain distance away in the direction to the EE z direction.
    This is calculated based on the z-axis of the end-effector's frame.

    Args:
        start_pose (Pose): The current pose of the robot's end-effector in the base frame.
        distance (float): The distance to move in meters.

    Returns:
        Pose: The new pose adjusted by 'distance' perpendicularly to the end-effector's current orientation.
    """

    # Extract quaternion from start_pose
    qx = start_pose.orientation.x
    qy = start_pose.orientation.y
    qz = start_pose.orientation.z
    qw = start_pose.orientation.w

    # Convert the quaternion to a rotation matrix
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    # The Z-axis in the end-effector's frame represents the forward direction
    position_vector_ee = np.array([0, 0, distance])
    position_vector_base = np.dot(R, position_vector_ee)

    # Apply this to the current position in the base frame
    start_position = np.array([start_pose.position.x, 
                               start_pose.position.y, 
                               start_pose.position.z])
    end_position = start_position + position_vector_base

    # Create the new pose with the adjusted position
    end_pose = copy.deepcopy(start_pose)
    end_pose.position.x = end_position[0]
    end_pose.position.y = end_position[1]
    end_pose.position.z = end_position[2]

    return end_pose, position_vector_base

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    # Normalize the quaternion to avoid errors due to drift
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    # Convert quaternion (qx, qy, qz, qw) to a rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def plot_live_3d_trajectory(robot_poses, robot_targets):
    """
    Updates the 3D trajectory plot in real-time using the positions from the list of PoseStamped messages.
    Also calculates and displays the RMSE between the actual robot positions and the target positions.
    The first robot pose is aligned with the first target, and the same offset is applied to all other points.
    
    :param robot_poses: List of Pose messages containing the actual trajectory.
    :param robot_targets: List of Pose messages containing the target trajectory.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize the plot
    plt.ion()  # Interactive mode on
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x, y, z positions from the robot poses
    px = np.array([pose.position.x for pose in robot_poses])
    py = np.array([pose.position.y for pose in robot_poses])
    pz = np.array([pose.position.z for pose in robot_poses])

    # Extracting x, y, z positions from the robot targets
    tx = np.array([pose.position.x for pose in robot_targets])
    ty = np.array([pose.position.y for pose in robot_targets])
    tz = np.array([pose.position.z for pose in robot_targets])

    # Plotting the target trajectory as a line connecting the points
    ax.plot(tx, ty, tz, marker='o', linestyle='-', color='r', label='Target Path')
    
    # Plotting the aligned robot trajectory as a line connecting the points
    ax.plot(px, py, pz, marker='o', linestyle='-', color='b', label='Actual Path')

    # Calculate RMSE for each axis
    rmse_x = np.sqrt(np.mean((px - tx)**2)) * 1000
    rmse_y = np.sqrt(np.mean((py - ty)**2)) * 1000
    rmse_z = np.sqrt(np.mean((pz - tz)**2)) * 1000

    # Overall RMSE (Euclidean distance)
    rmse_overall = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)

    # Display RMSE on the plot
    rmse_text = f'\n RMSE (mm):\nX={rmse_x:.2f},\nY={rmse_y:.2f},\nZ={rmse_z:.2f},\nOverall={rmse_overall:.2f}'
    ax.text2D(0.05, 0.95, rmse_text, transform=ax.transAxes)

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Keep the plot open after the loop
    plt.ioff()
    plt.show()

def read_trajectory_from_file(file_path):
    """Reads the trajectory.xyz file and stores each line as a PoseStamped variable."""
    poses = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            if len(data) == 7:
                # Extract position (x, y, z) and orientation (quaternion x, y, z, w)
                x, y, z = float(data[0]), float(data[1]), float(data[2])
                qx, qy, qz, qw = float(data[3]), float(data[4]), float(data[5]), float(data[6])

                # Create a PoseStamped message
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "panda_link0"
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw

                # Append the PoseStamped message to the list
                poses.append(pose)
    
    return poses

def get_current_external_force():
    """
    Retrieves the current external force applied to the robot's end-effector.
    """

    msg = rospy.wait_for_message('/franka_state_controller/F_ext', WrenchStamped)
    ext_force = msg.wrench.force
    return np.array([ext_force.x, ext_force.y, ext_force.z])

def get_EE_T_US():
    """Return the transformation matrix from payload (US) frame frame to end-effector (EE)."""

    EE_T_US = np.array([
        [0.707, 0.707, 0.0, 0.0],
        [-0.707, 0.707, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.179],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return EE_T_US

def get_T_S():
    """Return the image Scale Transformation."""

    T_S = np.array([
        [2.38238604e-04, 0.0, 0.0, 0.0],
        [0.0, 2.99864728e-04, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return T_S

def get_O_T_EE():
    msg = rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
    O_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T
    return O_T_EE

def get_EE_T_C():
    """Return the transformation matrix from camera (C) frame to end-effector (EE) frame."""

    # Computed by follwing the tutorial at: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-calibration-extrinsic.html
    # EE_T_C = np.array([
    #     [0.7144664408, -0.6996531067, -0.00482030742, 0.02339265235],
    #     [0.699666528, 0.7144254519, 0.007938726206, -0.0748114657],
    #     [-0.002110604146, -0.009044561213, 0.9999568697, 0.01558410105],
    #     [0, 0, 0, 1]
    # ])
    # EE_T_C = np.array([
    #     [0.7029283164,  -0.7112483258,  0.004195356826,  0.01943622883],
    #     [0.7112257036,  0.7028229919,  -0.01406558053,  -0.06868686773],
    #     [0.00705552737,  0.01287094045,  0.9998922734,  0.01729763919],
    #     [0,  0,  0,  1]
    # ])
    # EE_T_C = np.array([
    #     [0.7029200982,  -0.7111337457,  0.01386113997,  0.01333930614],
    #     [0.7112536662,  0.7028988485,  -0.007171549375,  -0.07448362188],
    #     [-0.004643048551,  0.01489981281,  0.9998782114,  0.01579639855],
    #     [0,  0,  0,  1]
    # ])
    EE_T_C = np.array([
        [0.7029200982,  -0.7111337457,  0.01386113997,  0.0115],
        [0.7112536662,  0.7028988485,  -0.007171549375,  -0.065],
        [-0.004643048551,  0.01489981281,  0.9998782114,  0.032],
        [0,  0,  0,  1]
    ])
    return EE_T_C
