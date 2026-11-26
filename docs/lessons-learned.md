# Lessons learned

The first version of the engine was written from assumptions. Then it met three real videos:
an overhang, a vertical wall with volumes, and a beginner (a child) on a vertical wall.
This is what changed and why.

## The assumptions that broke

| # | Assumption (v1) | What the videos showed | Change (v2) |
|---|---|---|---|
| 1 | The phone stands on a tripod | **All three** videos were hand-held: the camera followed the climber and zoomed (up to 1.45×) | Camera motion compensation: track the wall, measure the body relative to the wall |
| 2 | Hanging = hands above the shoulders in the image | On an overhang/roof the body is horizontal – "above" in the image means nothing | "Above the shoulder" is measured along the body axis (hips → shoulders) |
| 3 | Both arms are always visible | Both arms were visible in only ~55% of frames (the body hides one arm) | Judge the visible arm; lower confidence when only one is seen |
| 4 | Any bent arm while still is a mistake | Two of the "errors" were a **lock-off** (one arm bent, the other reaching) and **palming a volume** at shoulder height – both correct technique | Single bent arm is only judged when the other arm is hidden, not when it is visibly free; hands must be clearly above the shoulder |
| 5 | Everything in the video is climbing | The beginner was "re-adjusting feet" – **on the mat, before starting** | Detect the floor level from the moments before the climb; feet near it are not judged |
| 6 | The end of the climb is like the rest | Matching the top hold was flagged as "bent arms" | The last 2 s of the climb are not judged |
| 7 | Small foot movements are re-adjustments | Many were 5–7% of torso length – within tracking noise | Minimum shift raised to 7%; bursts are merged into one event ("3× re-adjust") |
| 8 | Feet are always trackable | On the overhang only 2 foot placements were found in 20 s | Report says "footwork was not assessed" instead of praising perfect feet |

## v1 → v2 on the same videos

| Video | v1 events | v2 events | v1 straight-arm ratio | v2 straight-arm ratio |
|---|---|---|---|---|
| Overhang (22 s) | 0 | 0 + warning: feet not visible | 49% | 98% |
| Vertical + volumes (47 s) | 3 re-adjusts | 0 | 88% | 52% |
| Beginner, child (40 s) | 7 re-adjusts | 1 foot slip (91% confidence) | 38% | 85% |

**How to read this honestly**
- v2 is more conservative: fewer, more defensible events. That raises precision, but
  we have **no ground truth yet**, so recall is unknown – v2 may now miss real mistakes.
- The straight-arm ratio swings a lot between versions. That shows the metric depends heavily on
  definitions (what counts as "holding" and "hanging"). It should not be shown to users as a precise
  number until it is validated against a coach's judgement.
- Visual inspection of frames confirmed that the removed v1 events (lock-off, palming, mat, top-out)
  were false alarms, and that the v2 foot slip looks real.

## Design lessons

1. **The real user behaviour beats the planned one.** The product idea started from "tripods at the gym".
   The data said "a friend films with a phone". This changes the recording flow, not just the algorithm.
2. **Correct technique can look like a mistake out of context.** Every rule needs a phase:
   on the floor, holding, moving, locking off, finishing.
3. **Silence is better than a wrong compliment.** When feet are not visible, saying "perfect feet" is worse than
   saying nothing. Quality warnings are part of the product.
4. **A false alarm costs more than a miss** for a beginner – it teaches the wrong thing and erodes trust.
   For the MVP we optimise for precision and let the user confirm or reject uncertain events.
5. **Kids and beginners need a different tone and thresholds** (smaller bodies, different movement patterns,
   more re-adjustments are normal while learning).

## Next validation step

Label every attempt in a small set of videos with a coach or experienced climber
(`evaluate.py template` → edit → `evaluate.py score`) and publish precision / recall per issue.
