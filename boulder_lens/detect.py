"""Rule-based technique detectors working on landmarks in wall coordinates.

Design choices (see docs/lessons-learned.md):
* "Hand above the shoulder" is measured along the body axis, not the image
  vertical, so it also works on overhangs and roofs where the body is horizontal.
* An arm hidden behind the body is ignored instead of blocking detection.
* Errors depend on the phase of movement (Sensors 2023): bent arms are only an
  error while *holding a position*, not while pulling through a move.
* Every event gets a confidence score, so the UI can be honest about uncertainty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import DEFAULT, Config

NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
L_FOOT, R_FOOT = 31, 32


@dataclass
class Event:
    kind: str                 # bent_arms | foot_slip | foot_readjust
    start: float              # s
    end: float                # s
    side: str = ""            # left | right | both
    confidence: float = 1.0   # 0..1
    detail: dict = field(default_factory=dict)


# ---------- helpers ----------

def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Fills NaN gaps by linear interpolation, then moving average along time."""
    x = x.astype(np.float64).copy()
    T = len(x)
    flat = x.reshape(T, -1)
    t = np.arange(T)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = ~np.isnan(col)
        if good.sum() >= 2:
            col[~good] = np.interp(t[~good], t[good], col[good])
        if k > 1:
            pad = np.pad(col, k // 2, mode="edge")
            col[:] = np.convolve(pad, np.ones(k) / k, mode="valid")[:T]
    return flat.reshape(x.shape)


def angle(a, b, c):
    """Angle ABC in degrees for [T, 2] arrays."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def segments(mask: np.ndarray, fps: float, min_dur: float):
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) / fps >= min_dur:
                out.append((start, i - 1))
            start = None
    return out


# ---------- features ----------

def compute_features(lm: np.ndarray, fps: float, cfg: Config = DEFAULT) -> dict | None:
    T = len(lm)
    detected = ~np.isnan(lm[:, 0, 0])
    if detected.sum() < fps:
        return None
    vis = np.nan_to_num(lm[:, :, 2], nan=0.0)
    xy = smooth(lm[:, :, :2], k=max(1, int(round(fps / 10))))

    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    torso_vec = sh - hip
    torso = float(np.nanmedian(np.linalg.norm(torso_vec, axis=-1)))
    if not torso or np.isnan(torso):
        return None
    axis = torso_vec / (np.linalg.norm(torso_vec, axis=-1, keepdims=True) + 1e-9)  # hip -> shoulders

    sw = max(1, int(round(cfg.speed_window_s * fps)))  # speed window in frames

    def speed(p):
        v = np.zeros(T)
        if T > sw:
            v[sw:] = np.linalg.norm(p[sw:] - p[:-sw], axis=-1) / torso * fps / sw
            if sw > 1:
                v[:sw] = v[sw]
        return v

    # climbing window: first..last frame with a hand above the head
    head_y = xy[:, NOSE, 1]
    over_head = detected & ((xy[:, L_WR, 1] < head_y) | (xy[:, R_WR, 1] < head_y))
    idx = np.where(over_head)[0]
    climbing = np.zeros(T, bool)
    if len(idx):
        end = idx[-1]
        # Rope climbing: the video often ends with a fall or with being lowered. Find the final
        # continuous descent of the hips; if it is long (> lowering_drop), the climb ends where it starts.
        hip_s = smooth(hip[:, 1:2], k=max(1, int(fps)))[:, 0]
        j = end
        while j > idx[0] and (hip_s[j] - hip_s[j - 1]) * fps > 0.3 * torso:  # moving down faster than 0.3 torso/s
            j -= 1
        if hip_s[end] - hip_s[j] > cfg.lowering_drop * torso:
            end = j
        idx = idx[idx <= end]
        climbing[idx[0]: end + 1] = True

    # floor level: where the feet are before the climb starts (preferred – camera drift is smallest there)
    # or after it ends. None if the video starts and ends on the wall.
    # Measured per foot: on a thick mat seen in perspective each foot rests at a different image height.
    ground_y = {}
    if len(idx):
        before = detected & (np.arange(T) < idx[0])
        after = detected & (np.arange(T) > idx[-1])
        for part in (before, after):
            if part.sum() >= fps * 0.5:
                ground_y = {k: float(np.median(xy[part, k, 1])) for k in (L_FOOT, R_FOOT)}
                break

    arms = {}
    for side, (s, e, w) in {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}.items():
        visible = (vis[:, [s, e, w]] > cfg.vis_min).all(1)
        reach = ((xy[:, w] - xy[:, s]) * axis).sum(-1) / torso  # >0: hand beyond the shoulder along body axis
        arms[side] = {
            "angle": angle(xy[:, s], xy[:, e], xy[:, w]),
            "visible": visible,
            "hanging": visible & (reach > cfg.hanging_reach),
            "vis": vis[:, [s, e, w]].mean(1),
        }

    return {
        "T": T, "fps": fps, "torso": torso, "xy": xy, "vis": vis, "detected": detected,
        "climbing": climbing, "speed": speed, "body_speed": speed(hip),
        "body_static": (speed(hip) < cfg.body_static_speed) & detected,
        "arms": arms, "ground_y": ground_y, "wrists": (xy[:, L_WR] + xy[:, R_WR]) / 2,
        "climb_start": float(idx[0] / fps) if len(idx) else None,
        "climb_end": float(np.where(climbing)[0][-1] / fps) if climbing.any() else None,
    }


# ---------- detectors ----------

def detect_bent_arms(f: dict, cfg: Config):
    """Holding a position on bent arms.

    * both arms visible  -> both must hang on holds and both be bent
    * one arm hidden     -> judge the visible one (lower confidence)
    * one arm visible but not hanging (reaching / chalking) -> a lock-off, which is good technique: no event
    """
    L, R = f["arms"]["left"], f["arms"]["right"]
    bentL = L["hanging"] & (L["angle"] < cfg.bent_arm_angle)
    bentR = R["hanging"] & (R["angle"] < cfg.bent_arm_angle)
    both = bentL & bentR
    if cfg.bent_arm_any:  # one clearly bent arm, unless the other is visibly free (lock-off / reaching)
        lock_off_l = bentL & R["visible"] & ~R["hanging"]
        lock_off_r = bentR & L["visible"] & ~L["hanging"]
        both = (bentL & ~lock_off_l) | (bentR & ~lock_off_r)
    only_left = bentL & ~R["visible"]
    only_right = bentR & ~L["visible"]
    # the final match on the top hold is not an error: skip the last moments of the climb
    finishing = np.zeros(f["T"], bool)
    if f["climb_end"] is not None:
        finishing[int(max(0, f["climb_end"] - cfg.finish_ignore) * f["fps"]):] = True
    mask = f["climbing"] & ~finishing & f["body_static"] & (both | only_left | only_right)
    if cfg.gap_fill_s > 0:  # bridge short breaks (a beginner sways a little while holding a position)
        gap = int(round(cfg.gap_fill_s * f["fps"]))
        on = np.where(mask)[0]
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= gap + 1:
                mask[a:b] = True
    events = []
    for a, b in segments(mask, f["fps"], cfg.bent_arm_min_dur):
        both = (L["hanging"][a:b + 1] & R["hanging"][a:b + 1]).mean()
        angles = np.concatenate([L["angle"][a:b + 1][L["hanging"][a:b + 1]], R["angle"][a:b + 1][R["hanging"][a:b + 1]]])
        margin = np.clip((cfg.bent_arm_angle - np.median(angles)) / 30, 0, 1)  # how clearly bent
        vis = np.concatenate([L["vis"][a:b + 1], R["vis"][a:b + 1]]).mean()
        conf = vis * (0.7 + 0.3 * both) * (0.5 + 0.5 * margin)
        events.append(Event("bent_arms", a / f["fps"], b / f["fps"], "both" if both > 0.5 else "one",
                            round(float(conf), 2), {"median_elbow_angle": int(np.median(angles))}))
    return events


def detect_feet(f: dict, cfg: Config):
    fps, T, torso, xy, vis = f["fps"], f["T"], f["torso"], f["xy"], f["vis"]
    win = max(1, int(cfg.slip_window * fps))
    slips, readjusts, placements = [], [], 0
    for side, idx in (("left", L_FOOT), ("right", R_FOOT)):
        foot = xy[:, idx]
        on_wall = f["climbing"].copy()
        if idx in f["ground_y"]:  # ignore the foot while it is on the floor / mat
            on_wall &= foot[:, 1] < f["ground_y"][idx] - cfg.ground_margin * torso
        planted = (f["speed"](foot) < cfg.foot_planted_speed) & (vis[:, idx] > cfg.vis_min) & on_wall
        plants = segments(planted, fps, cfg.foot_planted_min_dur)
        placements += len(plants)

        for a, b in plants:  # slip: planted foot shoots down while the hands stay on the holds
            j = min(T - 1, b + win)
            drop = (foot[j, 1] - foot[b, 1]) / torso
            hands = np.linalg.norm(f["wrists"][j] - f["wrists"][b]) / torso
            if drop > cfg.slip_drop and hands < cfg.slip_hands_still and j < T - fps:
                conf = float(vis[b:j + 1, idx].mean() * min(1, drop / (2 * cfg.slip_drop)))
                slips.append(Event("foot_slip", b / fps, j / fps, side, round(conf, 2), {"drop": round(float(drop), 2)}))

        raw = []  # re-adjust: same foot placed again almost in the same spot
        for (a1, b1), (a2, b2) in zip(plants, plants[1:]):
            gap = (a2 - b1) / fps
            dist = np.linalg.norm(foot[a2] - foot[b1]) / torso
            if gap < cfg.readjust_window and cfg.readjust_min_dist < dist < cfg.readjust_max_dist:
                raw.append((b1 / fps, a2 / fps, float(dist), float(vis[b1:a2 + 1, idx].mean())))
        for r in raw:  # merge bursts into one event ("3 re-adjustments") – less noise in the UI
            if readjusts and readjusts[-1].side == side and r[0] - readjusts[-1].end < cfg.readjust_merge_gap:
                e = readjusts[-1]
                e.end = r[1]
                e.detail["count"] += 1
                e.confidence = round(max(e.confidence, r[3]), 2)
            else:
                readjusts.append(Event("foot_readjust", r[0], r[1], side, round(r[3], 2), {"count": 1}))
    return slips, readjusts, placements


def analyze(lm: np.ndarray, fps: float, cfg: Config = DEFAULT) -> dict:
    """Runs all detectors. `lm` should already be in wall coordinates."""
    f = compute_features(lm, fps, cfg)
    if f is None:
        return {"ok": False, "reason": "no_person", "events": []}

    bent = detect_bent_arms(f, cfg)
    slips, readjusts, placements = detect_feet(f, cfg)
    events = sorted(bent + slips + readjusts, key=lambda e: e.start)

    L, R = f["arms"]["left"], f["arms"]["right"]
    holding = f["climbing"] & f["body_static"] & (L["hanging"] | R["hanging"])
    straight = holding & ((L["hanging"] & (L["angle"] >= cfg.bent_arm_angle)) | (R["hanging"] & (R["angle"] >= cfg.bent_arm_angle)))
    n_readjust = sum(e.detail["count"] for e in readjusts)
    climbing = f["climbing"]

    return {
        "ok": True,
        "fps": fps,
        "duration_s": round(f["T"] / fps, 1),
        "climb_start_s": f["climb_start"],
        "climb_end_s": f["climb_end"],
        "events": [asdict(e) for e in events],
        "metrics": {
            "straight_arm_ratio": round(float(straight.sum() / max(1, holding.sum())), 2),
            "foot_placements": placements,
            "clean_foot_ratio": round(1 - n_readjust / max(1, placements), 2),
            "foot_slips": len(slips),
        },
        "quality": {
            "person_detected": round(float(f["detected"][climbing].mean()) if climbing.any() else 0.0, 2),
            "both_arms_visible": round(float((L["visible"] & R["visible"])[climbing].mean()) if climbing.any() else 0.0, 2),
            # per-limb visibility: reaching overhead briefly hides one arm, which is normal
            "arms_visible": round(float(((L["visible"].astype(float) + R["visible"]) / 2)[climbing].mean()) if climbing.any() else 0.0, 2),
            "feet_visible": round(float(((f["vis"][:, L_FOOT] > cfg.vis_min).astype(float) + (f["vis"][:, R_FOOT] > cfg.vis_min))[climbing].mean() / 2) if climbing.any() else 0.0, 2),
        },
    }
