"""Downloads the MediaPipe Pose Landmarker model (~9 MB) into models/."""
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
dest = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"
dest.parent.mkdir(exist_ok=True)
print(f"Downloading {URL}\n -> {dest}")
urllib.request.urlretrieve(URL, dest)
print("Done.")
