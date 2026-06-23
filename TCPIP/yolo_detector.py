from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os

class YoloGraspDetector:
    def __init__(self, weights_path="checkpoints/yolo_weights.pt", conf=0.25):
        """
        初始化 YOLO 抓取检测器
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [YOLO] Loading model on {self.device}...")
        
        # 确保权重文件存在
        if not os.path.exists(weights_path):
            print(f"⚠️ [YOLO] Warning: Weights file not found at {weights_path}")
            
        self.model = YOLO(weights_path)
        self.conf = conf
        print("✅ [YOLO] Model loaded successfully.")

    def detect(self, color_image):
        """
        传入彩色图像，返回检测到的物体列表。
        返回格式: [{'x': px, 'y': py, 'label': class_name, 'conf': confidence}, ...]
        """
        # 按照你原脚本的逻辑，先做一次高斯模糊去噪
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

            # 遍历 OBB 框
            for corners, conf, cls in zip(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls):
                pts = corners.detach().cpu().numpy()
                
                # 计算中心点坐标
                px = int(np.mean(pts[:, 0]))
                py = int(np.mean(pts[:, 1]))
                
                # 限制坐标在画面内
                h, w = color_image.shape[:2]
                px = int(np.clip(px, 0, w - 1))
                py = int(np.clip(py, 0, h - 1))

                class_id = int(cls)
                label = self.model.names[class_id] if class_id in self.model.names else str(class_id)

                detections.append({
                    'x': px,
                    'y': py,
                    'label': label,
                    'conf': float(conf)
                })

        # 按置信度从高到低排序，优先抓取最确定的物体
        detections.sort(key=lambda item: item['conf'], reverse=True)
        return detections