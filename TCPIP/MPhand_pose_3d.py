import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions


class MPhandDetector:
    def __init__(self, model_path='hand_landmarker.task'):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=2
        )
        self.detector = HandLandmarker.create_from_options(options)

    def detect_2d(self, image):

        h, w, _ = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.detector.detect(mp_image)


        all_points = np.zeros((42, 2), dtype=np.float32)
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            if i >= len(result.handedness):
                continue
            score = result.handedness[i][0].score
            if score < 0.3:
                continue
            if i > 1:
                break
            for j, lm in enumerate(hand_landmarks):
                x, y = int(lm.x * w), int(lm.y * h)
                all_points[i * 21 + j] = (x, y)

        return np.array(all_points, dtype=np.float32)


def MPhand2D_to_3D(pose_2d_42, verts, image_shape, depth_threshold=0.2):

    h, w = image_shape[:2]
    pose_3d = []

    palm_indices = [0, 21]
    palm_depths = {}

    for i in palm_indices:
        x, y = pose_2d_42[i]
        if x > 0 and y > 0:
            idx = int(y) * w + int(x)
            if 0 <= idx < verts.shape[0]:
                z = verts[idx][2]
                if z > 0:
                    palm_depths[i] = z

    for i, (x, y) in enumerate(pose_2d_42):
        if x <= 0 or y <= 0:
            pose_3d.append([0.0, 0.0, 0.0])
            continue

        x, y = int(round(x)), int(round(y))
        idx = y * w + x
        pt = [0.0, 0.0, 0.0]

        if 0 <= idx < verts.shape[0]:
            raw_pt = verts[idx]
            z_valid = raw_pt[2] > 0

            palm_idx = 0 if i < 21 else 21
            palm_z = palm_depths.get(palm_idx, None)

            if z_valid:
                if palm_z is not None and abs(raw_pt[2] - palm_z) > depth_threshold:
                    pt = [raw_pt[0], raw_pt[1], palm_z]
                else:
                    pt = raw_pt
            elif palm_z is not None:
                pt = [raw_pt[0], raw_pt[1], palm_z]

        pose_3d.append(pt)

    return np.array(pose_3d, dtype=np.float32)



hand_detector = MPhandDetector()

def MPhandDetect2D(image):
    return hand_detector.detect_2d(image)
