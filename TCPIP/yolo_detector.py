from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import math

class YoloGraspDetector:
    def __init__(self, weights_filename="yolo_weights.pt", conf=0.25):
        """
        初始化 YOLO 抓取检测器
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [YOLO] Loading model on {self.device}...")
        
        # 🌟 自动获取当前脚本所在的绝对目录，防止相对路径报错
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        
        # 如果 checkpoints 文件夹不存在，自动创建
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            print(f"📁 [YOLO] Created checkpoints directory at: {checkpoints_dir}")
            
        self.weights_path = os.path.join(checkpoints_dir, weights_filename)
        
        # 🌟 防崩溃机制：如果你还没放自己的权重文件，自动用官方的基础 OBB 模型测试
        if not os.path.exists(self.weights_path):
            print(f"⚠️ [YOLO] Warning: Your custom weights file '{weights_filename}' was not found at {self.weights_path}")
            print("🔄 [YOLO] Fallback: Using default 'yolov8n-obb.pt' for testing purposes...")
            self.weights_path = "yolov8n-obb.pt" # ultralytics 会自动下载这个基础旋转框模型
            
        # 加载模型
        self.model = YOLO(self.weights_path)
        self.conf = conf
        print("✅ [YOLO] Model loaded successfully.")

    def detect(self, color_image):
        """
        传入彩色图像，返回检测到的物体列表。
        返回格式: [{'x': px, 'y': py, 'angle': angle_rad, 'label': class_name, 'conf': confidence}, ...]
        """
        # 高斯模糊去噪
        yolo_input = cv2.GaussianBlur(color_image, (5, 5), 2)

        # 运行推理
        results = self.model(
            yolo_input,
            conf=self.conf,
            iou=0.25,
            max_det=2,
            verbose=False
        )

        detections = []
        for r in results:
            if r.obb is None:
                continue

            # 🌟 提取 xywhr：中心x, 中心y, 宽, 高, 旋转弧度
            for box_data, conf, cls in zip(r.obb.xywhr, r.obb.conf, r.obb.cls):
                px = int(box_data[0].item())
                py = int(box_data[1].item())
                angle_rad = box_data[4].item() # 提取 OBB 的旋转角度 (弧度)
                
                # 限制坐标在画面内，防止越界
                h, w = color_image.shape[:2]
                px = int(np.clip(px, 0, w - 1))
                py = int(np.clip(py, 0, h - 1))

                class_id = int(cls)
                label = self.model.names[class_id] if class_id in self.model.names else str(class_id)

                detections.append({
                    'x': px,
                    'y': py,
                    'angle': angle_rad, # 将角度保存给主程序
                    'label': label,
                    'conf': float(conf)
                })

        # 按置信度从高到低排序，优先抓取最确定的物体
        detections.sort(key=lambda item: item['conf'], reverse=True)
        return detections


# =====================================================================
# 🛠️ 独立测试主程序 (Unit Test)
# 运行方式: python3 yolo_detector.py
# 架构: 严格遵守“生产者-消费者”模式，由 BodyPointCloud_dual 提供底层数据
# =====================================================================
if __name__ == "__main__":
    print("🛠️ 启动 YOLO 独立测试可视化模式 (基于 RealSense 底层数据流)...")
    
    # 1. 导入你的底层相机数据驱动模块
    try:
        import BodyPointCloud_dual
    except ImportError:
        print("❌ 无法导入 BodyPointCloud_dual.py，请确保它在同一目录下！")
        exit()
    
    # 2. 初始化 YOLO 检测器
    detector = YoloGraspDetector(weights_filename="yolo_weights.pt", conf=0.30)
    
    # 构造一个虚拟的 T_M 矩阵，仅仅是为了让底层的函数能跑通，不影响画面
    dummy_T_M = np.eye(4)
    
    print("▶️ 正在启动 RealSense 硬件并实时推理... 按 'q' 键或 'Esc' 键退出。")

    while True:
        # 🌟 核心架构：驱动 BodyPointCloud_dual 去硬件拿一帧新鲜的数据
        # 这里默认 use_dual_camera=False (单目测试)，如果你是双目可以改为 True
        _, should_quit = BodyPointCloud_dual.Body3DSkeletonProcess_dual(dummy_T_M, use_dual_camera=False)
        
        if should_quit:
            print("⚠️ 收到退出信号。")
            break
            
        # 🌟 从“全局橱窗”拿取原汁原味的彩色画面
        frame = BodyPointCloud_dual.global_raw_color_image
        
        if frame is None or frame.shape[0] == 0:
            continue
            
        # 3. 运行 YOLO 推理
        detections = detector.detect(frame)
        
        # 4. 在画面上绘制结果
        for i, det in enumerate(detections):
            px = det['x']
            py = det['y']
            angle = det['angle']
            label = det['label']
            conf = det['conf']
            
            # --- 绘制 A: 中心点 (绿色圆点) ---
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
            
            # --- 绘制 B: 信息标签 ---
            text = f"#{i+1} {label} {conf:.2f}"
            cv2.putText(frame, text, (px + 10, py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # --- 绘制 C: 旋转角度与趋势指示 (红色箭头) ---
            arrow_length = 60
            end_x = int(px + arrow_length * math.cos(angle))
            end_y = int(py + arrow_length * math.sin(angle))
            cv2.arrowedLine(frame, (px, py), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)
            
            print(f"🎯 识别到 {label}: 中心({px}, {py}), 夹爪需旋转: {math.degrees(angle):.1f}°")

        # 5. 显示画面
        cv2.imshow("YOLO OBB Grasp Test (RealSense Aligned)", frame)
        
        # 6. 按键退出逻辑
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("👋 退出测试程序。")
            break

    cv2.destroyAllWindows()