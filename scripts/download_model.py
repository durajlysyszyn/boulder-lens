"""Downloads the MediaPipe Pose Landmarker model into models/.

    python scripts/download_model.py          # full model, ~9 MB (default)
    python scripts/download_model.py heavy    # heavy model, ~30 MB (more accurate on hard footage)
"""
import sys
import urllib.request
from pathlib import Path

variant = sys.argv[1] if len(sys.argv) > 1 else "full"
if variant not in ("full", "heavy"):
    raise SystemExit("usage: download_model.py [full|heavy]")
URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")
dest = Path(__file__).resolve().parent.parent / "models" / f"pose_landmarker_{variant}.task"
dest.parent.mkdir(exist_ok=True)
print(f"Downloading {URL}\n -> {dest}")
urllib.request.urlretrieve(URL, dest)
print("Done.")
