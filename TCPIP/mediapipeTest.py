import pyrealsense2 as rs
import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
import mediapipe as mp


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2)
detector = HandLandmarker.create_from_options(options)


while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result = detector.detect(mp_image)
    h, w, _ = img.shape
    for hand in result.hand_landmarks:
        for i, lm in enumerate(hand):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.imshow("Hand Keypoints", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
