# Gaze-AR-Robot

Gaze-AR-Robot is a distributed shared-autonomy demonstrator for human--robot collaboration. An operator wearing a HoloLens 2 authorizes targets using gaze or hand-ray input, receives spatial augmented-reality (AR) feedback, and collaborates with a Franka Research 3 (FR3) robot equipped with a Franka Hand and an eye-in-hand Intel RealSense D435 camera. The robot-side computer runs the Python middleware and motion interface; an RTX 4060 PC provides the HoloLens hotspot, a Tailscale relay, and a WSL2 instance for Contact-GraspNet inference.

> **Safety notice.** This repository drives physical robot hardware. Run it only with a trained operator, an enabled emergency stop, validated collision settings, and an unobstructed workspace. The supplied code is a research prototype, not a safety-rated robot-control system.

## Capabilities

- HoloLens--robot point-based calibration and cached rigid transform.
- Eye-in-hand RealSense acquisition, point-cloud accumulation, plane removal, clustering, and AR point-cloud feedback.
- Gaze-ray dwell and hand-ray/pinch target authorization.
- Skeleton tracking and reaching-intent detection.
- Contact-GraspNet inference from a cleaned point cloud, followed by object-specific grasp-pool caching.
- AR trajectory teaching, robot state streaming, gripper control, and a non-blocking HRI state machine.

## System topology

| Node | Primary responsibility | Main software |
|---|---|---|
| HoloLens 2 | AR interface, gaze/hand input, point-cloud and state visualization | Unity UWP client |
| Robot-side PC | TCP middleware, perception, coordinate transforms, state machine, robot commands | Python, ROS, `franka_ros` |
| RTX 4060 PC | Mobile hotspot, Tailscale/port forwarding, GPU inference host | Windows, WSL2, Contact-GraspNet |
| FR3 + Franka Hand + D435 | Motion execution, gripper actions, wrench/joint feedback, RGB-D sensing | Franka control stack, RealSense SDK |

The HoloLens connects to the mobile hotspot on TCP port `8848`. The robot-side PC and RTX PC communicate through a private overlay network. The point-cloud grasp request is forwarded to the WSL2 Contact-GraspNet service on TCP port `5000`. Use local configuration values for all hosts and addresses; do **not** commit private addresses, VPN credentials, calibration matrices, or robot-specific safety limits.

## Repository layout

The current implementation is split across three working trees. Keep the relative Python imports intact.

```text
TCPIP/                                      # Robot-side Python project
├── Main_HRI_Demo.py                        # Main TCP server and HRI state machine
├── Main_HRI_Demo_Exp1.py                   # Experimental variant
├── Main_Calibration_Only.py                # Calibration-only workflow
├── robotController.py                      # ROS Cartesian/gripper interface
├── udp_from_tf.py                          # TF/joint-state UDP broadcaster
├── BodyPointCloud_dual.py                  # RealSense, body/skeleton processing
├── gaze_interaction.py                     # Ray--point-cloud intersection and mapping
├── request_grasps.py                       # HTTP client for WSL inference
├── compute_alignment.py                    # HoloLens--robot alignment
├── pathInterpolation.py                    # Waypoint interpolation/orientation handling
├── config.yaml                             # Runtime configuration; machine-specific
├── environment.yaml                        # Recorded Windows/vision Conda environment
├── hand_landmarker.task                    # MediaPipe hand-landmark model asset
└── checkpoints/                            # Local YOLO weights (not necessarily distributable)

MT20262019/                                 # Unity HoloLens project
├── Assets/Scripts/TCPClient.cs             # UWP TCP client and packet dispatcher
└── Packages/manifest.json                  # Unity package versions

contact_graspnet/                           # WSL2 Contact-GraspNet checkout
└── server_wsl.py                           # Flask wrapper for point-cloud inference
```

## Hardware and operating-system requirements

- Microsoft HoloLens 2, developer mode enabled for UWP deployment.
- Franka Research 3, Franka Hand, control box, Ethernet connection, and a FR3-compatible `libfranka`/`franka_ros` installation.
- Intel RealSense D435 rigidly mounted to the end effector, with a stable USB 3 connection.
- Robot-side Ubuntu computer with ROS, a built `catkin_ws`, and access to the Franka control network.
- Windows PC with NVIDIA RTX 4060 GPU, Windows mobile-hotspot support, Tailscale, WSL2, and a CUDA-compatible NVIDIA driver.
- A router-free local link between the HoloLens and RTX PC, plus the private overlay link between the RTX PC and robot-side PC.

## Dependencies

### 1. Python middleware and perception environment

`TCPIP/environment.yaml` records the vision environment used for the current project: **Python 3.9.21**, NumPy 1.26.4, SciPy 1.13.1, Open3D 0.19.0, RealSense Python bindings 2.55.1.6486, MediaPipe 0.10.21, PyTorch 2.0.1 + CUDA 11.7, and Ultralytics 8.3.92. Edit the `prefix` field in that file before creating an environment on a different machine.

```bash
conda env create -f environment.yaml
conda activate <your-environment>
```

The following packages are required by the active HRI path in `Main_HRI_Demo.py` and its imported modules:

| Package | Used for |
|---|---|
| `numpy`, `scipy` | Transformations, point-cloud processing, interpolation |
| `PyYAML` | `config.yaml` loading |
| `opencv-python` / `opencv-contrib-python` | RGB-D image processing and calibration utilities |
| `open3d` | Point-cloud filtering, mapping, segmentation, and visualization |
| `pyrealsense2` | Intel RealSense D435 acquisition |
| `scikit-learn` | PCA and point-cloud/body-processing utilities |
| `torch`, `torchvision`, `torchaudio` | GPU model runtime |
| `ultralytics` | YOLO pose/object models used by the local perception modules |
| `mediapipe` | Hand landmark detection (`hand_landmarker.task`) |
| `requests` | HTTP request to the grasp-inference service |
| `matplotlib` | Optional debugging/visualization path in the main server |

Useful optional packages present in the recorded environment include `vtk` (alignment visualization), `pynput` (legacy keyboard-driven scripts), `pillow`, `pandas`, `tensorrt`, `onnxruntime-gpu`, and `pycuda`. `apriltag` is only needed by utilities that import `helper_functions.py`.

For a minimal manual installation, use the package versions in `environment.yaml` as the source of truth rather than assuming the newest packages are compatible:

```bash
python -m pip install numpy scipy PyYAML opencv-python open3d pyrealsense2 \
    scikit-learn requests matplotlib mediapipe ultralytics
```

Install PyTorch separately with the CUDA build compatible with the NVIDIA driver and the selected CUDA runtime. The recorded environment uses `torch==2.0.1+cu117`.

### 2. ROS and Franka robot-control dependencies

The robot-side process imports ROS Python APIs and expects the following packages in the sourced ROS/catkin workspace:

- `rospy`, `tf`, `tf2_ros`, and `actionlib`
- `geometry_msgs`, `sensor_msgs`, and `std_msgs`
- `franka_msgs` and `franka_gripper`
- `franka_control` and `franka_gripper` launch files
- An FR3-compatible `libfranka` and `franka_ros` build

The code uses a Cartesian target topic, external wrench feedback, joint states, TF between `panda_link0` and `panda_link8`, and Franka gripper action messages. Verify the topic/frame names against the installed controller before enabling motion.

### 3. WSL2 grasp-inference environment

The WSL2 service in `server_wsl.py` accepts a serialized NumPy dictionary containing an `N x 3` point cloud (and optional RGB/intrinsics) through the `/predict_grasp` Flask endpoint. It creates a temporary `real_time_cloud.npy` file and invokes a generated `contact_graspnet/inference_pcd.py` script.

Install the following in the dedicated WSL2/Conda environment:

- [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet), its compatible TensorFlow/CUDA stack, and pretrained checkpoint `scene_test_2048_bs3_hor_sigma_001`.
- `Flask` and `numpy` for the HTTP wrapper.
- A CUDA-enabled NVIDIA WSL driver compatible with the Contact-GraspNet runtime.

The wrapper is a local adaptation: it injects point-cloud loading into a copy of the upstream inference script. Keep the upstream checkout unmodified and ensure that the server can create `contact_graspnet/inference_pcd.py` at launch. The `checkpoints/` and inference assets are required but should be acquired under the upstream project's licence; they are not recreated by this repository.

### 4. Unity / HoloLens application dependencies

The Unity project was created with **Unity 2019.4.22f1** and targets UWP/HoloLens. The manifest pins:

- Microsoft Mixed Reality Toolkit Foundation 2.6.1
- Microsoft Mixed Reality Toolkit Standard Assets 2.6.1
- Windows Mixed Reality XR Plugin 4.2.3
- Unity Barracuda 2.1.0-preview (used by local visual inference scripts)
- TextMeshPro and Unity UI

`TCPClient.cs` uses `Windows.Networking.Sockets`, `Windows.Storage.Streams`, and `Newtonsoft.Json` in its `WINDOWS_UWP` build branch. Confirm that the project's UWP capabilities permit network client access and that the Newtonsoft assembly/package is available in the Unity project. Rebuild and deploy the UWP project from Unity after changing package versions.

### 5. Required third-party software and services

| Component | Purpose | Notes |
|---|---|---|
| [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) | D435 drivers and `pyrealsense2` bindings | Match the installed camera firmware/SDK. |
| [ROS](https://www.ros.org/) | Robot middleware | Use the distribution compatible with the built `catkin_ws`. |
| [libfranka / franka_ros](https://github.com/frankaemika/franka_ros) | FR3 control, state, and gripper interface | Must match the robot's supported version. |
| [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) | 6-DoF grasp proposal inference | Requires model weights and its own licence/runtime. |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO pose/object detection | Include the local model weights or document how to obtain them. |
| [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Hand landmark tracking | `hand_landmarker.task` must be present. |
| [Open3D](https://www.open3d.org/) | 3-D geometry processing | Used for scene mapping and grasp visualization. |
| [Tailscale](https://tailscale.com/) | Private link between the two PCs | Use site-local policy and credentials. |
| [Unity](https://unity.com/) + [MRTK](https://github.com/microsoft/MixedRealityToolkit-Unity) | HoloLens AR client | Preserve the manifest versions above. |

## Configuration before first run

1. Copy `config.yaml` to a machine-local configuration if the repository is shared. Set the TCP bind host/port, output directory, calibration-file paths, and the locally measured hand--eye transform. Never reuse another rig's calibration matrix.
2. Verify that `tm_matrix.txt`, `robotPosition.txt`, and any calibration-record files belong to the current robot, camera mount, and HoloLens session. Generate a fresh transform if the HoloLens world frame is reset or the camera mount changes.
3. Configure the HoloLens `TCPClient` inspector fields with the local bridge endpoint and port `8848`. Do not hard-code personal addresses into source control.
4. Create the RTX PC forwarding rules for the HoloLens-to-robot service and robot-to-WSL grasp service. Store only a sanitized template/configuration guide in the repository.
5. Confirm that the camera serial number, YOLO weight files, `hand_landmarker.task`, Contact-GraspNet checkpoint, and ROS launch files are present.

## Recommended start-up order

1. **Prepare the physical cell.** Check the emergency stop, workspace, mounting rigidity, robot network, D435 USB link, and approved collision configuration.
2. **Start the relay node.** Enable the RTX PC hotspot, establish the private overlay connection, and apply the local forwarding configuration for ports `8848` and `5000`.
3. **Start WSL2 inference.** Activate the Contact-GraspNet environment and run:

   ```bash
   cd ~/contact_graspnet
   conda activate graspnet
   python server_wsl.py
   ```

4. **Start the robot stack.** On the robot-side Ubuntu computer, bring up the overlay connection, source the catkin workspace, launch the Cartesian controller and gripper, then apply the lab-approved collision configuration. Do not enable the robot until the actual controller, robot address, and safety configuration have been reviewed.

5. **Start robot state publishing.** In a separate sourced terminal:

   ```bash
   cd <TCPIP_ROOT>
   python udp_from_tf.py
   ```

6. **Start the HRI middleware.** In another sourced terminal:

   ```bash
   cd <TCPIP_ROOT>
   python Main_HRI_Demo.py
   ```

7. **Start the HoloLens client.** Open the deployed app, connect it to the middleware, and complete the five-point HoloLens--robot calibration. The workflow uses four corresponding points to estimate the rigid transform and reserves one point for validation. Continue only when the validation error is within the lab's acceptance criterion.

8. **Verify before task execution.** Confirm robot-pose feedback, joint-state rendering, D435 stream, skeleton/AR feedback, grasp-service response, and correct cursor registration before enabling autonomous motion.

## Runtime protocol notes

`Main_HRI_Demo.py` and `TCPClient.cs` exchange header-prefixed binary TCP packets. Main flows include AR calibration/records, gaze and hand-ray input, hand position, waypoint paths, robot pose/joints, skeletons, point clouds, HRI state, cursor targets, and motion-complete notifications. The WSL service is a separate HTTP request/response channel for grasp inference.

The current Unity client treats header `t` as a three-float AR cursor/target point. Before extending the protocol, keep one authoritative definition for each header and update both Python and C# dispatchers together; in particular, do not reuse `t` for a transform matrix on the same stream.

## Quick diagnostics

| Symptom | First checks |
|---|---|
| HoloLens cannot connect | Hotspot association, local forwarding rule, TCP port `8848`, endpoint configured in Unity. |
| No robot motion or state | ROS master, sourced workspace, controller/gripper launch status, TF frames, topic names. |
| Empty/unstable point cloud | D435 USB 3 connection, RealSense permission/driver, camera mount, `EE_T_C`, point-cloud limits. |
| No grasp candidates | WSL service running, port `5000` forwarding, checkpoint path, CUDA/TensorFlow compatibility, accepted point-cloud shape. |
| Misregistered AR cursor | Repeat five-point calibration; ensure the HoloLens app stayed open and the camera mount has not moved. |
| YOLO/MediaPipe errors | Check model weights, `hand_landmarker.task`, GPU-capable PyTorch build, and matching package versions. |

## Data, models, and reproducibility

Do not commit recordings, participant data, camera serial numbers, network addresses, VPN credentials, calibration matrices, or robot-specific collision values. Track model provenance and licences separately for YOLO weights, MediaPipe assets, and Contact-GraspNet checkpoints. For a reproducible deployment, record the operating-system version, ROS distribution, `libfranka`/`franka_ros` commit, NVIDIA driver/CUDA version, camera firmware, Unity package manifest, and the exact local configuration used for the run.

## Acknowledgements and third-party notices

This project relies on ROS, libfranka/franka_ros, Intel RealSense, Open3D, Contact-GraspNet, PyTorch, Ultralytics YOLO, MediaPipe, Unity, MRTK, Tailscale, and their respective dependencies. Retain their licence notices and cite their associated publications/software when using this system in an academic artifact.
