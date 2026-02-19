
import torch
import numpy as np
import cv2
from ultralytics import YOLO  # 直接使用 YOLOv8 加载模型
import threading


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"YOLO segment on: {device}")


model_seg_engine = None
_model_seg_lock = threading.Lock()

def get_model_seg():
    global model_seg_engine
    with _model_seg_lock:
        if model_seg_engine is None:
            print("📦 Loading YOLOv8-Seg TensorRT model...")
            model_seg_engine = YOLO("yolo11n-seg.pt").to(device)
            #model_seg_engine = YOLO("yolo11n-seg.engine")  # 这里无需使用 .to(device)
            _ = model_seg_engine.predict(np.zeros((480, 640, 3), dtype=np.uint8))  # 模型 warmup
        return model_seg_engine


def segment_and_crop(color_image, depth_intrinsics, ero_para):
    model_seg = get_model_seg()


    results = model_seg(color_image, verbose=False)
    w, h = depth_intrinsics.width, depth_intrinsics.height

    mask = None
    for result in results:
        masks = result.masks
        classes = result.boxes.cls if result.boxes is not None else []

        if masks is not None and len(masks.data) > 0:
            for i, cls_id in enumerate(classes):
                if int(cls_id) == 0:
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (w, h))
                    break

    if mask is not None:
        mask = (mask > 0.9).astype(np.uint8)
        kernel = np.ones((ero_para, ero_para), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.erode(mask, kernel, iterations=1)
        indices = np.where(mask.flatten() == 1)[0]
    else:

        indices = np.array([], dtype=int)

    return indices

