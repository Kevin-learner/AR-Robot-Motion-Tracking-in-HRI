# from ultralytics import YOLO
#
# model = YOLO('yolo11n-seg.pt')
# model.export(format='onnx', opset=12, simplify=True, imgsz=(640, 640), dynamic=False)
#


from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n-pose.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11n-pose.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)


