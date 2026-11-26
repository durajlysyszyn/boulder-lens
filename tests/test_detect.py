import numpy as np

from boulder_lens.camera import to_wall
from boulder_lens.detect import analyze
from boulder_lens.feedback import build_feedback
from tests.synthetic import make_climber


def kinds(r):
    return [e["kind"] for e in r["events"]]


def test_bent_arms_detected_in_time_window():
    lm, fps = make_climber(bent=(1.0, 3.5))
    r = analyze(lm, fps)
    bent = [e for e in r["events"] if e["kind"] == "bent_arms"]
    assert len(bent) == 1
    assert abs(bent[0]["start"] - 1.0) < 0.2 and abs(bent[0]["end"] - 3.5) < 0.2


def test_straight_arms_no_event():
    lm, fps = make_climber(bent=None)
    r = analyze(lm, fps)
    assert "bent_arms" not in kinds(r)
    assert r["metrics"]["straight_arm_ratio"] > 0.95


def test_bent_arms_on_roof():
    """Lesson learned: 'hand above shoulder' must follow the body axis, not the image vertical."""
    lm, fps = make_climber(bent=(1.0, 3.5), roof=True)
    assert "bent_arms" in kinds(analyze(lm, fps))


def test_bent_arms_with_one_arm_hidden():
    """Lesson learned: one arm hidden behind the body must not block detection."""
    lm, fps = make_climber(bent=(1.0, 3.5), one_arm_hidden=True)
    r = analyze(lm, fps)
    ev = [e for e in r["events"] if e["kind"] == "bent_arms"]
    assert ev and ev[0]["side"] == "one"
    assert ev[0]["confidence"] < 0.9  # less certain than with both arms


def test_foot_slip():
    lm, fps = make_climber(bent=None, slip_at=5.0)
    slips = [e for e in analyze(lm, fps)["events"] if e["kind"] == "foot_slip"]
    assert len(slips) == 1 and slips[0]["side"] == "left"


def test_tracking_noise_is_not_a_readjust():
    lm, fps = make_climber(bent=None, noise=2.0)
    assert "foot_readjust" not in kinds(analyze(lm, fps))


def test_readjust_burst_is_merged():
    lm, fps = make_climber(bent=None, readjust_at=[2.0, 2.8, 3.6])
    ev = [e for e in analyze(lm, fps)["events"] if e["kind"] == "foot_readjust"]
    assert len(ev) == 1 and ev[0]["detail"]["count"] >= 2


def test_camera_pan_is_compensated():
    """A panning camera alone must not create events once mapped to wall coordinates."""
    lm, fps = make_climber(bent=None)
    T = len(lm)
    shift = np.linspace(0, 300, T)  # camera pans 300 px down over the clip
    panned = lm.copy()
    panned[:, :, 1] -= shift[:, None]
    mats = np.tile(np.eye(3), (T, 1, 1))
    mats[:, 1, 2] = shift  # what the camera tracker would estimate
    assert analyze(to_wall(panned, mats), fps)["events"] == []


def test_no_person():
    lm = np.full((100, 33, 3), np.nan, np.float32)
    assert analyze(lm, 30)["ok"] is False


def test_feedback_strengths_first_and_main_issue():
    lm, fps = make_climber(bent=(1.0, 3.5))
    r = analyze(lm, fps)
    fb = build_feedback(r, {"handheld": True}, "en")
    assert fb["main_issue"] == "bent_arms"
    assert fb["main_tip"]
    assert any("camera" in w for w in fb["warnings"])


def test_lockoff_is_not_an_error():
    """Lesson learned: one bent arm while the other is visibly free is a lock-off – good technique."""
    lm, fps = make_climber(bent=(1.0, 3.5), lockoff=True)
    assert "bent_arms" not in kinds(analyze(lm, fps))
