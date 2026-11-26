"""Analyze a bouldering video.

    python analyze.py my_climb.mov
    python analyze.py videos/*.mov --lang en --out results
"""
import argparse
import json
from pathlib import Path

from boulder_lens.pipeline import run

DEFAULT_MODEL = Path(__file__).parent / "models" / "pose_landmarker_full.task"


def main():
    ap = argparse.ArgumentParser(description="Bouldering technique analysis from video")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="MediaPipe pose model (.task)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lang", default="pl", choices=["pl", "en"])
    ap.add_argument("--no-video", action="store_true", help="only write the JSON report")
    args = ap.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Pose model not found: {args.model}\nDownload it with: python scripts/download_model.py")
    for v in args.videos:
        r = run(v, args.model, args.out, args.lang, render_video=not args.no_video)
        fb = r.get("feedback", {})
        print(f"\n== {Path(v).name}")
        print(json.dumps({"main_issue": fb.get("main_issue_label"), "strengths": fb.get("strengths"),
                          "events": len(r["events"]), "metrics": r.get("metrics")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
