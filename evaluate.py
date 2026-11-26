"""Compare detections with human annotations (ground truth) and report precision / recall.

1. Create an annotation file pre-filled with the app's detections:
       python evaluate.py template results/ > annotations/labels.json
2. Watch each *_analysis.mp4 and edit the file: delete false detections,
   add the mistakes the app missed (kind, start, end in seconds).
3. Score:
       python evaluate.py score results/ annotations/labels.json
"""
import json
import sys
from pathlib import Path

TOLERANCE = 0.75  # s – a detection matches a label if their time ranges overlap within this margin
KINDS = ["bent_arms", "foot_slip", "foot_readjust"]


def load_reports(results_dir):
    return {json.loads(p.read_text())["video"]: json.loads(p.read_text())
            for p in Path(results_dir).glob("*_report.json")}


def template(results_dir):
    out = {}
    for video, r in sorted(load_reports(results_dir).items()):
        out[video] = [{"kind": e["kind"], "start": round(e["start"], 1), "end": round(e["end"], 1), "side": e["side"]}
                      for e in r["events"]]
    print(json.dumps(out, indent=2))


def overlaps(a, b):
    return a["start"] - TOLERANCE <= b["end"] and b["start"] - TOLERANCE <= a["end"]


def score(results_dir, labels_file):
    reports = load_reports(results_dir)
    labels = json.loads(Path(labels_file).read_text())
    stats = {k: {"tp": 0, "fp": 0, "fn": 0} for k in KINDS}
    for video, truth in labels.items():
        pred = reports.get(video, {}).get("events", [])
        for k in KINDS:
            P = [e for e in pred if e["kind"] == k]
            G = [e for e in truth if e["kind"] == k]
            used = set()
            for g in G:
                hit = next((i for i, p in enumerate(P) if i not in used and overlaps(p, g)), None)
                if hit is None:
                    stats[k]["fn"] += 1
                else:
                    used.add(hit)
                    stats[k]["tp"] += 1
            stats[k]["fp"] += len(P) - len(used)
    print(f"{'kind':<15}{'TP':>4}{'FP':>4}{'FN':>4}{'precision':>11}{'recall':>8}")
    for k, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
        print(f"{k:<15}{s['tp']:>4}{s['fp']:>4}{s['fn']:>4}{p:>11.2f}{r:>8.2f}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "template":
        template(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
