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


def extract_landmarks_zoomed(video_path: str, model_path: str, fps: float, stride: int, max_width: int,
                             lm_coarse: np.ndarray, crop_size: int = 512, margin: float = 0.6):
    """Second pass for a small, far-away climber (typical for rope climbing filmed from below).

    The first pass (whole frame) tells roughly where the climber is. Here every frame is cropped around the
    climber from the FULL-resolution video, scaled up to `crop_size` and given to the pose model again, so the
    model sees the climber close up. Landmarks are mapped back to the downscaled frame used everywhere else.
    Frames where the first pass found nobody reuse the nearest known position.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    T = len(lm_coarse)
    # climber box per frame (downscaled coords), filled over gaps and smoothed
    pts = lm_coarse[:, :, :2]
    good = ~np.isnan(pts[:, 0, 0])
    if not good.any():
        return lm_coarse
    cx = np.full(T, np.nan); cy = np.full(T, np.nan); hh = np.full(T, np.nan)
    for t in np.where(good)[0]:
        p = pts[t][~np.isnan(pts[t, :, 0])]
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        cx[t], cy[t], hh[t] = (x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, x1 - x0)
    idx = np.arange(T)
    for a in (cx, cy, hh):
        a[~good] = np.interp(idx[~good], idx[good], a[good])
        k = max(1, int(fps * 0.3))
        a[:] = np.convolve(np.pad(a, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    hh = np.maximum(hh, np.median(hh))  # never crop tighter than the typical body size

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_tracking_confidence=0.3,
    )
    out = np.full_like(lm_coarse, np.nan)
    cap = cv2.VideoCapture(video_path)
    i = j = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while j < T:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride:
                i += 1
                continue
            H, W = frame.shape[:2]
            s = min(1.0, max_width / W)  # full-res -> downscaled factor
            side = hh[j] * (1 + 2 * margin) / s  # crop side in full-res pixels
            x0 = int(round(cx[j] / s - side / 2)); y0 = int(round(cy[j] / s - side / 2)); side_i = int(round(side))
            pad = max(0, -x0, -y0, x0 + side_i - W, y0 + side_i - H)
            src = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else frame
            crop = src[y0 + pad: y0 + pad + side_i, x0 + pad: x0 + pad + side_i]
            crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int(i * 1000 / fps))
            if res.pose_landmarks:
                for k, l in enumerate(res.pose_landmarks[0]):
                    out[j, k] = ((x0 + l.x * side_i) * s, (y0 + l.y * side_i) * s, l.visibility)
            i += 1
            j += 1
    cap.release()
    return out
