"""Body landmark extraction with MediaPipe Pose Landmarker (33 points)."""
from __future__ import annotations

import cv2
import numpy as np

from .video import iter_frames

N_LANDMARKS = 33


def extract_landmarks(video_path: str, model_path: str, fps: float, stride: int, max_width: int):
    """Returns an array [T, 33, 3]: x, y in pixels of the downscaled frame, and visibility.

    Frames where nobody was detected are NaN.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    out, size = [], None
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(iter_frames(video_path, max_width, stride)):
            h, w = frame.shape[:2]
            size = (w, h)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int(i * stride * 1000 / fps))
            arr = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks[0]):
                    arr[j] = (lm.x * w, lm.y * h, lm.visibility)
            out.append(arr)
    return np.stack(out), size
