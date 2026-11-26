"""Annotated video: skeleton, on-screen messages and an event timeline."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detect import L_EL, L_FOOT, R_EL, R_FOOT
from .video import iter_frames

FONTS = ["/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
ARM_BONES = {(11, 13), (13, 15), (12, 14), (14, 16)}

OK = (180, 220, 120)  # BGR
COLORS = {"bent_arms": (60, 60, 235), "foot_slip": (40, 150, 255), "foot_readjust": (40, 210, 240)}


def _font(size):
    for p in FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(d, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def _banner(frame, text, color_bgr, font, dim=False, tip=None, tip_font=None):
    """Message card at the top: what happened + (optionally) what to do about it."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    W = img.width
    pad, x0, y0 = 14, 20, 20
    max_w = W - 2 * x0 - 2 * pad - 22
    tip_lines = _wrap(d, tip, tip_font, max_w) if tip else []
    tw = max([d.textlength(text, font=font)] + [d.textlength(l, font=tip_font) for l in tip_lines])
    h = font.size + 2 * pad + len(tip_lines) * int(tip_font.size * 1.35) + (6 if tip_lines else 0)
    d.rounded_rectangle([x0, y0, x0 + tw + 2 * pad + 22, y0 + h], 12, fill=(20, 20, 20, 210))
    cy = y0 + pad + font.size / 2
    fill = (color_bgr[2], color_bgr[1], color_bgr[0], 120 if dim else 255)
    d.ellipse([x0 + pad, cy - 6, x0 + pad + 12, cy + 6], fill=fill)
    d.text((x0 + pad + 22, y0 + pad - 2), text, font=font, fill=(255, 255, 255, 170 if dim else 255))
    ty = y0 + pad + font.size + 6
    for line in tip_lines:
        d.text((x0 + pad + 22, ty), line, font=tip_font, fill=(225, 225, 225, 255))
        ty += int(tip_font.size * 1.35)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _rolling_median(x: np.ndarray, half: int) -> np.ndarray:
    """Centred running median along time for a [T] array with NaNs."""
    T = len(x)
    pad = np.pad(x, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)[:T]
    with np.errstate(all="ignore"):
        return np.nanmedian(win, axis=1)


def smooth_for_display(lm: np.ndarray, fps: float, window_s: float = 0.25, median_s: float = 0.8,
                       jump_torso: float = 0.4) -> np.ndarray:
    """Removes skeleton jitter for drawing (the video is offline, so centred windows are fine).

    1. Outliers: the pose model is sometimes *confidently wrong* for a few frames – e.g. it swaps the left
       and right leg, so a foot jumps half a wall while the climber stands still. A running median ignores
       such short spikes; any point further than `jump_torso` torso lengths from it is dropped and filled
       in from the neighbouring frames.
    2. Wobble: a running median of the cleaned track (edge-preserving: a step to a new hold stays sharp).
    3. Jitter: each point = average of the neighbouring frames, weighted by visibility squared.
    """
    T = len(lm)
    xy = lm[:, :, :2].astype(float).copy()
    vis = np.nan_to_num(lm[:, :, 2], nan=0.0)
    sh = (xy[:, 11] + xy[:, 12]) / 2
    hip = (xy[:, 23] + xy[:, 24]) / 2
    torso = np.nanmedian(np.linalg.norm(sh - hip, axis=-1))
    torso = torso if np.isfinite(torso) and torso > 0 else 100.0

    mhalf = max(1, int(round(median_s * fps / 2)))
    t = np.arange(T)
    for j in range(lm.shape[1]):
        med = np.stack([_rolling_median(xy[:, j, c], mhalf) for c in range(2)], axis=1)
        bad = np.linalg.norm(xy[:, j] - med, axis=1) > jump_torso * torso
        xy[bad, j] = np.nan
        vis[bad, j] *= 0.3  # a corrected point is drawn more faintly / may be hidden
        for c in range(2):
            col = xy[:, j, c]
            good = ~np.isnan(col)
            if good.sum() >= 2:
                col[~good] = np.interp(t[~good], t[good], col[good])
        # Climbing is mostly holds and steps: a running median keeps a real step to a new hold sharp
        # but removes the model's wobble while a limb is actually still.
        xy[:, j] = np.stack([_rolling_median(xy[:, j, c], mhalf) for c in range(2)], axis=1)

    half = max(1, int(round(window_s * fps / 2)))
    kernel = np.hanning(2 * half + 3)[1:-1]
    w = vis ** 2
    xy0 = np.nan_to_num(xy, nan=0.0)
    out = np.full_like(lm, np.nan, dtype=float)
    for j in range(lm.shape[1]):
        wj = np.convolve(w[:, j], kernel, mode="same")
        for c in range(2):
            out[:, j, c] = np.convolve(xy0[:, j, c] * w[:, j], kernel, mode="same") / np.maximum(wj, 1e-6)
        out[:, j, 2] = np.convolve(vis[:, j], kernel, mode="same") / kernel.sum()
    out[np.isnan(lm[:, 0, 0])] = np.nan  # nobody detected -> draw nothing
    return out


def render(video_path, lm, report, out_path, fps, stride, max_width, show_after=1.0, low_conf_text="?"):
    lm = smooth_for_display(lm, fps)
    T = len(lm)
    events = report["events"]
    dur = T / fps
    tmp_dir = Path(tempfile.mkdtemp())
    tmp = tmp_dir / "raw.mp4"
    writer, font = None, None
    for i, frame in enumerate(iter_frames(video_path, max_width, stride)):
        if i >= T:
            break
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            font = _font(max(18, w // 32))
            tip_font = _font(max(15, w // 42))
        t = i / fps
        active = [e for e in events if e["start"] <= t <= e["end"] + show_after]
        kinds = {e["kind"] for e in active}
        thick = max(2, w // 300)

        pts = lm[i]
        if not np.isnan(pts[0, 0]):
            for a, b in BONES:
                if pts[a, 2] > 0.6 and pts[b, 2] > 0.6:
                    col = COLORS["bent_arms"] if ("bent_arms" in kinds and (a, b) in ARM_BONES) else OK
                    cv2.line(frame, tuple(pts[a, :2].astype(int)), tuple(pts[b, :2].astype(int)), col, thick, cv2.LINE_AA)
            for e in active:
                if e["kind"] in ("foot_slip", "foot_readjust"):
                    idx = L_FOOT if e["side"] == "left" else R_FOOT
                    if not np.isnan(pts[idx, 0]):
                        cv2.circle(frame, tuple(pts[idx, :2].astype(int)), max(14, w // 40), COLORS[e["kind"]], thick + 1, cv2.LINE_AA)
            if "bent_arms" in kinds:
                for idx in (L_EL, R_EL):
                    if pts[idx, 2] > 0.4:
                        cv2.circle(frame, tuple(pts[idx, :2].astype(int)), max(8, w // 70), COLORS["bent_arms"], -1, cv2.LINE_AA)

        # timeline
        bar_h = max(10, h // 60)
        y = h - bar_h - 16
        cv2.rectangle(frame, (20, y), (w - 20, y + bar_h), (60, 60, 60), -1)
        span = w - 40
        for e in events:
            x0 = 20 + int(span * e["start"] / dur)
            x1 = max(x0 + 5, 20 + int(span * e["end"] / dur))
            col = COLORS[e["kind"]] if not e.get("low_confidence") else tuple(int(c * 0.5) for c in COLORS[e["kind"]])
            cv2.rectangle(frame, (x0, y), (x1, y + bar_h), col, -1)
        px = 20 + int(span * t / dur)
        cv2.rectangle(frame, (px - 2, y - 6), (px + 2, y + bar_h + 6), (255, 255, 255), -1)

        if active:
            e = active[-1]
            text = e.get("label", e["kind"]) + (f" ({low_conf_text})" if e.get("low_confidence") else "")
            frame = _banner(frame, text, COLORS[e["kind"]], font, dim=e.get("low_confidence", False),
                            tip=e.get("tip"), tip_font=tip_font)
        writer.write(frame)
    if writer:
        writer.release()
    if shutil.which("ffmpeg"):  # H.264 so it plays in browsers and on phones
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], check=True)
    else:
        shutil.copy(tmp, out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
