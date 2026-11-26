"""Shared frame reader, so pose, camera-motion and rendering see exactly the same frames."""
from __future__ import annotations

import cv2


def video_info(path: str, target_fps: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(fps / target_fps))
    return fps, stride, fps / stride


def iter_frames(path: str, max_width: int, stride: int = 1):
    """Yields downscaled BGR frames (every `stride`-th frame). OpenCV applies phone rotation metadata."""
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
            yield frame
        i += 1
    cap.release()
