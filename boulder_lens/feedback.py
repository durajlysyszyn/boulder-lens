"""Turns raw detections into feedback a climber can act on.

UX principles (see docs/lessons-learned.md):
1. Strengths first – a report that is only criticism makes people stop filming.
2. One main thing to work on (inspired by VBeta's "the one issue that hurt your send").
3. Honesty about limits – tell the user when the video makes the analysis less reliable.
4. Low-confidence events are kept but marked, so the user can confirm or reject them.
"""
from __future__ import annotations

TEXT = {
    "pl": {
        "label": {
            "bent_arms": "Ugięte ręce w pozycji",
            "foot_slip": "Zsunięcie stopy",
            "foot_readjust": "Poprawianie stopy",
        },
        "tip": {
            "bent_arms": "Gdy stoisz w pozycji, wisz na prostych rękach. Nogi i biodra pracują, ręce tylko trzymają.",
            "foot_slip": "Patrz na stopień aż do postawienia stopy i obciąż go czubkiem buta.",
            "foot_readjust": "Wybierz miejsce na stopniu, zanim podniesiesz nogę, i postaw stopę raz, cicho i precyzyjnie.",
        },
        "strength": {
            "straight_arms": "Dobra praca na prostych rękach ({p}% czasu w pozycji)",
            "quiet_feet": "Precyzyjne stopy ({p}% postawień bez poprawiania)",
            "no_slips": "Stabilne stopy – żadnego zsunięcia",
        },
        "warning": {
            "handheld": "Kamera ruszała się w trakcie nagrania – wyniki skompensowano, ale statyw da dokładniejszą analizę.",
            "arms_hidden": "Ręce były często zasłonięte ({p}% klatek z obiema rękami) – ocena rąk jest mniej pewna.",
            "person_lost": "Sylwetka nie była widoczna w części nagrania – spróbuj kadru z całym ciałem.",
            "feet_hidden": "Stopy były słabo widoczne – praca nóg nie została oceniona.",
        },
        "readjust_count": "{n}× poprawka",
        "low_conf": "niepewne",
    },
    "en": {
        "label": {
            "bent_arms": "Bent arms while holding",
            "foot_slip": "Foot slip",
            "foot_readjust": "Foot re-adjustment",
        },
        "tip": {
            "bent_arms": "When holding a position, hang on straight arms. Legs and hips do the work, arms just hold.",
            "foot_slip": "Watch the foothold until your foot is on it and load it with the toe of the shoe.",
            "foot_readjust": "Pick the spot on the foothold before lifting your leg, then place the foot once, quietly and precisely.",
        },
        "strength": {
            "straight_arms": "Good straight-arm climbing ({p}% of holding time)",
            "quiet_feet": "Precise feet ({p}% of placements without re-adjusting)",
            "no_slips": "Stable feet – no slips",
        },
        "warning": {
            "handheld": "The camera moved during the video – motion was compensated, but a tripod gives a more accurate analysis.",
            "arms_hidden": "Arms were often hidden ({p}% of frames with both arms visible) – arm feedback is less certain.",
            "person_lost": "The climber was not visible for part of the video – try framing the whole body.",
            "feet_hidden": "Feet were hard to see – footwork was not assessed.",
        },
        "readjust_count": "{n}× re-adjust",
        "low_conf": "uncertain",
    },
}

SEVERITY = {"foot_slip": 3.0, "bent_arms": 2.0, "foot_readjust": 1.0}
LOW_CONFIDENCE = 0.5


def build_feedback(report: dict, camera: dict, lang: str = "pl") -> dict:
    t = TEXT[lang]
    events = report["events"]
    for e in events:
        e["label"] = t["label"][e["kind"]]
        if e["kind"] == "foot_readjust" and e["detail"].get("count", 1) > 1:
            e["label"] += " (" + t["readjust_count"].format(n=e["detail"]["count"]) + ")"
        e["low_confidence"] = e["confidence"] < LOW_CONFIDENCE

    # main issue: severity x confidence, summed per kind
    score = {}
    for e in events:
        w = e["detail"].get("count", 1)
        score[e["kind"]] = score.get(e["kind"], 0) + SEVERITY[e["kind"]] * e["confidence"] * w
    main = max(score, key=score.get) if score else None

    m, q = report["metrics"], report["quality"]
    climb_s = (report["climb_end_s"] or 0) - (report["climb_start_s"] or 0)
    feet_hidden = climb_s > 8 and m["foot_placements"] < climb_s / 4

    strengths = []
    if m["straight_arm_ratio"] >= 0.7 and q["both_arms_visible"] >= 0.3:
        strengths.append(t["strength"]["straight_arms"].format(p=int(m["straight_arm_ratio"] * 100)))
    if not feet_hidden:  # never praise what we could not see
        if m["foot_placements"] >= 4 and m["clean_foot_ratio"] >= 0.85:
            strengths.append(t["strength"]["quiet_feet"].format(p=int(m["clean_foot_ratio"] * 100)))
        if m["foot_slips"] == 0 and m["foot_placements"] >= 4:
            strengths.append(t["strength"]["no_slips"])

    warnings = []
    if camera.get("handheld"):
        warnings.append(t["warning"]["handheld"])
    if q["both_arms_visible"] < 0.6:
        warnings.append(t["warning"]["arms_hidden"].format(p=int(q["both_arms_visible"] * 100)))
    if feet_hidden:
        warnings.append(t["warning"]["feet_hidden"])
    if q["person_detected"] < 0.9:
        warnings.append(t["warning"]["person_lost"])

    return {
        "main_issue": main,
        "main_issue_label": t["label"][main] if main else None,
        "main_tip": t["tip"][main] if main else None,
        "strengths": strengths,
        "warnings": warnings,
    }
