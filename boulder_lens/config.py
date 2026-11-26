"""Detection thresholds.

All distances are expressed in *torso lengths* (shoulder-centre to hip-centre),
so thresholds do not depend on how far the camera is from the wall.
All values are assumptions calibrated on a handful of real videos – see
docs/lessons-learned.md. Override them by passing a modified Config.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # processing
    max_width: int = 960            # frames are downscaled to this width
    target_fps: float = 30.0        # 60 fps videos are sampled every 2nd frame
    zoom_below: float = 0.06            # torso shorter than this share of frame height -> zoomed second pass
    vis_min: float = 0.5            # minimum landmark visibility to trust a point

    # body "holding a position" (static phase)
    body_static_speed: float = 0.35     # torso/s

    # bent arms in a static position
    bent_arm_angle: float = 130.0       # degrees; below = bent
    bent_arm_min_dur: float = 1.0       # s
    finish_ignore: float = 2.0          # s – last moments of the climb (matching the top) are not judged
    hanging_reach: float = 0.25         # torso – hand must be clearly beyond the shoulder (along body axis)
                                        #         to count as hanging; hands at shoulder level = pressing/palming

    # feet
    foot_planted_speed: float = 0.35    # torso/s
    foot_planted_min_dur: float = 0.3   # s
    slip_drop: float = 0.12             # torso – downward drop of a planted foot...
    slip_window: float = 0.25           # s     – ...within this time
    slip_hands_still: float = 0.08      # torso – while the hands stay on the holds
    readjust_window: float = 1.2        # s     – foot placed again within this time...
    readjust_min_dist: float = 0.07     # torso – ...moved more than this (below = tracking noise)
    readjust_max_dist: float = 0.35     # torso – ...but less than this (more = a new foothold)
    readjust_merge_gap: float = 2.0     # s     – re-adjustments closer than this become one event
    ground_margin: float = 0.3          # torso – feet this close to the floor level are not judged

    lowering_drop: float = 3.0          # torso – hips dropping this much after the high point = lowered on the rope

    # report
    show_after: float = 2.5             # s – how long a message + tip stays on screen after an event (time to read)


DEFAULT = Config()
