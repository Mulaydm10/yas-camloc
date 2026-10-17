# Slide outline: 5 minutes, 6 slides

All results below are from the training run (night) with held-out laps: the map is rebuilt without the scored
lap, processing is causal, and only the allowed car signals are used.

## 1. Problem and approach (45 s)
- Task: 2D pose of the race car at 10 Hz from its cameras on Yas Marina, causal, with no GNSS at inference.
- Idea: a **known map plus odometry plus a filter**. Car odometry predicts, cameras correct heading
  (relative rotation) and absolute position (a geo-referenced feature map built from the training run).
- Constraint-driven: CPU only, real time, robust to lighting changes (the test may be in daylight).

## 2. System diagram (60 s)
- Wheel speeds and steering feed a calibrated bicycle model (EKF predict).
- `fl` / `fr` / `r` images feed two updates: keyframe rotation (LK + essential matrix) for relative yaw, and
  ORB features (CLAHE) matched to the 3D map, solved with PnP + RANSAC, as an absolute fix weighted by quality.
- Gates: cross-camera consensus, then an NIS test with recovery; the track model handles the pit lane and gives
  heading/lateral priors only when there are no fixes.
- (Diagram: README "Method".)

## 3. Results (60 s)
- Held-out laps 2 / 4 / 6: **0.674 / 0.613 / 0.577 m** position RMSE. Pooled **0.630 m**, lateral 0.477 m,
  heading 0.26°, **91 % within 1 m**.
- Progression: 528 m (hold) → 144 (odometry) → 55 (map priors) → 21.5 (visual heading) → about 1 m (map
  fixes) → 0.63 m (gating, CLAHE, quality weighting).
- Plot: `docs/traj_lap4_heldout.png` (trajectory on the track map, coloured by error, plus error vs time).

## 4. Runtime (30 s)
- Apple M4 CPU, no GPU: tens of ms per frame (see `out.run.json`), against a 100 ms frame period, so real
  time. Per-camera work runs in threads, images are decoded once at half resolution, and messages are parsed
  directly.

## 5. Robustness and failure cases (60 s)
- Lighting and blur proxy: CLAHE raises map-matching success under washed-out images from 20 % to 60 %, and
  under 9 px blur from 11 % to 43 %.
- Map-fix loss: 30 s / 60 s blackouts give 2.3 / 2.9 m and recover. Total map failure falls back to 21.5 m.
- Failure cases:
  - the feature-poor back straight at night;
  - a residual lateral bias of about −0.2 m from sideslip (model fitted, inconsistent on held-out laps);
  - lateral position from paint is unobservable (camera height above the paint about 0.2 m).

## 6. Lessons learned (45 s)
- Measure observability before building: paint-based lateral looked natural and was geometrically useless.
- Honest validation changes decisions: leave-one-lap-out and strict calibration hold-out (leakage about 0.02 m).
- Small, principled gates (consensus, NIS, quality weighting) beat a heavier model; there are no pre-trained
  networks.
- Open: daylight data, and the rules question on fused car-state fields (ADR-0003).
