"""End-to-end analysis: video -> landmarks -> wall coordinates -> detections -> feedback -> outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camera import estimate_camera_motion, motion_summary, to_wall
from .config import DEFAULT, Config
from .detect import analyze
from .feedback import TEXT, build_feedback
from .html_report import write_html
from .pose import extract_landmarks
from .render import render
from .video import video_info


def run(video: str, model: str, out_dir: str, lang: str = "pl", cfg: Config = DEFAULT,
        render_video: bool = True, cache: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem
    src_fps, stride, fps = video_info(video, cfg.target_fps)

    cache_file = out / f"{stem}.landmarks.npz"
    if cache and cache_file.exists():
        c = np.load(cache_file)
        lm, mats, size = c["lm"], c["mats"], tuple(c["size"])
    else:
        lm, size = extract_landmarks(video, model, src_fps, stride, cfg.max_width)  # image coords (for drawing)
        mats = estimate_camera_motion(video, lm, cfg.max_width, stride)              # camera motion
        np.savez_compressed(cache_file, lm=lm, mats=mats, size=np.array(size))

    report = analyze(to_wall(lm, mats), fps, cfg)                                    # analysis in wall coords
    report["video"] = Path(video).name
    report["camera"] = motion_summary(mats, fps, size[0])
    if report["ok"]:
        report["feedback"] = build_feedback(report, report["camera"], lang)
        if render_video:
            vid = out / f"{stem}_analysis.mp4"
            render(video, lm, report, vid, fps, stride, cfg.max_width, cfg.show_after, TEXT[lang]["low_conf"])
            report["annotated_video"] = vid.name
            write_html(report, out / f"{stem}_report.html", lang)
    (out / f"{stem}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
