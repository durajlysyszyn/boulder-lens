"""A synthetic stick-figure climber for testing detectors without real video."""
import numpy as np


def make_climber(duration=8.0, fps=30, bent=(1.0, 3.5), slip_at=None, readjust_at=None,
                 one_arm_hidden=False, lockoff=False, roof=False, noise=1.0, seed=0):
    T = int(duration * fps)
    lm = np.zeros((T, 33, 3), np.float32)
    lm[:, :, 2] = 0.9
    rng = np.random.default_rng(seed)
    for i in range(T):
        t = i / fps
        p = np.zeros((33, 2))
        hip = np.array([270.0, 560.0])
        sh = hip + [0, -160]
        p[11], p[12] = sh + [-45, 0], sh + [45, 0]
        p[23], p[24] = hip + [-30, 0], hip + [30, 0]
        if bent and bent[0] < t < bent[1]:
            p[13], p[14] = p[11] + [-70, -10], p[12] + [70, -10]
            p[15], p[16] = p[13] + [0, -80], p[14] + [0, -80]
        else:
            p[13], p[14] = p[11] + [-15, -90], p[12] + [15, -90]
            p[15], p[16] = p[13] + [-8, -90], p[14] + [8, -90]
        if lockoff:  # right hand down by the hip (reaching / chalking), still visible
            p[14], p[16] = p[12] + [20, 80], p[12] + [30, 160]
        p[0] = sh + [0, -60]
        p[25], p[26] = p[23] + [-50, 100], p[24] + [50, 100]
        lf = p[25] + [-10, 110]
        if slip_at and t > slip_at:
            lf = lf + [0, min(1, (t - slip_at) / 0.15) * 60]
        if readjust_at:
            for k, ra in enumerate(readjust_at):
                if ra < t < ra + 0.15:   # foot lifted and moving
                    lf = lf + [0, -25]
                if t >= ra + 0.15:
                    lf = lf + [((-1) ** k) * 20, 0]
        p[27], p[29], p[31] = lf + [0, -20], lf + [-15, -5], lf + [15, 0]
        p[28] = p[26] + [10, 90]
        p[30], p[32] = p[28] + [-15, 15], p[28] + [15, 20]
        if roof:  # rotate the whole body 80° (horizontal climber under a roof)
            c, s = np.cos(np.radians(80)), np.sin(np.radians(80))
            p = (p - hip) @ np.array([[c, -s], [s, c]]).T + hip
        lm[i, :, :2] = p
        if one_arm_hidden:
            lm[i, [12, 14, 16], 2] = 0.1
    lm[:, :, :2] += rng.normal(0, noise, (T, 33, 2))
    return lm, fps
