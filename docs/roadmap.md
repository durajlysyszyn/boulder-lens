# Roadmap

## ✅ Phase 0 – Engine prototype (done)
- Pose estimation + camera motion compensation
- 3 detectors, strengths, confidence, quality warnings
- Annotated video, HTML report, evaluation tooling, tests

## Phase 1 – Validate with climbers
- [ ] Interviews and observation at the gym: how and why people film, what they do with the videos
- [ ] Label 20–30 videos (`evaluate.py`), publish precision / recall in the README
- [ ] Review rules with a coach; tune `config.py`
- [ ] User test the report: is the tip understood? is the tone right for beginners and kids?

## Phase 2 – App MVP
- [ ] Record / import flow designed for the wall (chalky hands, phone far away: one-tap or voice start)
- [ ] Results screen: strengths, one tip, timeline, "this was intentional" correction
- [ ] On-device analysis, video never leaves the phone by default

## Phase 3 – More technique
- [ ] Hips far from the wall (needs side view or depth – research)
- [ ] Stiff body / no hip rotation (variance of hip and knee angles)
- [ ] Looking at hands instead of feet (needs a side or front view)
- [ ] Kicking the wall (combine foot deceleration with audio)
- [ ] Beginner / advanced presets (different thresholds and tone)

## Phase 4 – Walls, boulders, grades
- [ ] Photo of the wall → tap one hold → find all holds of the same colour
- [ ] Save a boulder, share it, rate it (amateur route setting)
- [ ] Link analysis to holds: "your foot slips on *this* foothold every time"
- [ ] Gym grade translator: crowd-sourced mapping of a gym's colours to Font / V-scale

## Phase 5 – Personal training path
- [ ] History of detected issues across sessions
- [ ] Recurring issue → drills (e.g. silent feet, straight-arm traverses)
- [ ] Progress view: is the issue becoming less frequent?

## Parked
- Shoe advice – low value without manufacturer data; may return as content, not a feature.
