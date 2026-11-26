"""Camera-motion compensation.

Lesson learned: nobody in our test videos used a tripod – the person filming
follows the climber, pans and zooms. Without compensation every camera move
looks like body movement.

We track feature points on the wall (background), masking out the climber,
and chain frame-to-frame similarity transforms into "wall coordinates"
(the coordinate system of the first frame).
"""
from __future__ import annotations

import cv2
import numpy as np

from .video import iter_frames


def estimate_camera_motion(video_path: str, lm: np.ndarray | None, max_width: int, stride: int,
                           flow_width: int = 480) -> np.ndarray:
    """Returns [T, 3, 3] matrices mapping pixels of frame t to pixels of frame 0."""
    mats, prev_gray, cum, s = [], None, np.eye(3), 1.0
    for i, frame in enumerate(iter_frames(video_path, max_width, stride)):
        h, w = frame.shape[:2]
        s = flow_width / w
        gray = cv2.cvtColor(cv2.resize(frame, (flow_width, int(h * s))), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mask = np.full(prev_gray.shape, 255, np.uint8)
            if lm is not None and i - 1 < len(lm):
                pts = lm[i - 1, :, :2]
                pts = pts[~np.isnan(pts[:, 0])] * s
                if len(pts):  # mask out the climber (plus margin) – we only want the wall
                    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                    pw, ph = (x1 - x0) * 0.35 + 20, (y1 - y0) * 0.2 + 20
                    cv2.rectangle(mask, (int(x0 - pw), int(y0 - ph)), (int(x1 + pw), int(y1 + ph)), 0, -1)
            step = np.eye(3)
            p0 = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
                good = st.ravel() == 1
                if good.sum() >= 12:
                    M, _ = cv2.estimateAffinePartial2D(p1[good], p0[good], method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.0)
                    if M is not None:
                        step = np.vstack([M, [0, 0, 1]])  # frame t -> frame t-1
            cum = cum @ step
        mats.append(cum.copy())
        prev_gray = gray
    S, Sinv = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    return np.stack([Sinv @ m @ S for m in mats])


def to_wall(lm: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Maps landmarks [T, 33, 3] into wall coordinates."""
    out = lm.copy()
    T = min(len(lm), len(mats))
    xy1 = np.concatenate([lm[:T, :, :2], np.ones((T, lm.shape[1], 1))], axis=-1)
    out[:T, :, :2] = np.einsum("tij,tkj->tki", mats[:T], xy1)[:, :, :2]
    return out


def motion_summary(mats: np.ndarray, fps: float, width: int) -> dict:
    pan = np.linalg.norm(np.diff(mats[:, :2, 2], axis=0), axis=-1) * fps / width  # frame widths per second
    scale = np.sqrt(np.abs(np.linalg.det(mats[:, :2, :2])))
    p95 = float(np.percentile(pan, 95)) if len(pan) else 0.0
    return {
        "pan_p95_per_s": round(p95, 3),
        "zoom_range": round(float(scale.max() / max(scale.min(), 1e-6)), 2),
        "handheld": p95 > 0.05,
    }
