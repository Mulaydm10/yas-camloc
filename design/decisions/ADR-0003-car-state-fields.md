# ADR-0003: Which car-state fields the submitted localiser uses

- Status: **Accepted (default), owner to confirm the submitted variant**
- Date/time (TZ): 2026-10-18 01:10 CEST
- Deciders: Claude (Opus) on the owner's instruction to work through the Devin review (GitHub issue #1)
- Related: REPORT.md §5, §9 Q1, §13; EXP-0004, EXP-0022, EXP-0023

## Context
The challenge brief (notes/brief/*.pdf, "Minimal requirements") says: "Only the camera images, camera
calibration and track borders may be used. Car state data (accelerations, speed, and other fields) can be used
for camera-based localization. Ground truth may not be used at inference, except the first pose of the run."
It also says the ground-truth pose comes "from the car's state estimation (RTK GNSS, lidar map matching, IMU
and odometry fusion)". The dataset README calls the `state_estimation` topic ground truth.

The fused fields `v_mps`, `vx/vy_mps`, `wz_radps`, `ax/ay_mps2`, `beta_rad` (`config.STATE_AMBIGUOUS`) are
outputs of that same estimator. They are "car state" in the brief's wording, but they come from the GT
pipeline. Pose fields and anything computed from the pose (`x/y/z`, attitude, `sn_map_state`, ...) are
forbidden in every reading, and the replay never exposes them.

## Options considered
- A: allowed signals only (wheel speeds, steering, pedals, gear, brake pressures). `--estimator vismap`.
- B: A plus the fused speed / yaw rate in the prediction step. `--estimator vismap_est --state ambiguous`.

## Evidence (held-out laps 2/4/6 pooled, CLAHE map, threshold 30, quality weighting)
| variant | position | lateral | heading | within 1 m | 30 s / 60 s blackout, lap 4 |
|---|---|---|---|---|---|
| A | **0.630 m** | 0.477 m | 0.26° | **91.0 %** | 2.32 / 2.95 m (whole run 2.90 / 4.73 m) |
| B | 0.639 m | **0.430 m** | 0.25° | 87.2 % | 2.45 / 3.90 m (whole run 3.10 / **9.32 m**) |

## Decision
Default and submitted variant: **A**.
- It is better on the two largest metrics (position and within 1 m), and close on lateral.
- It carries no rules risk.
- It degrades more gracefully when map fixes are lost: in a 60 s blackout over the whole run, 4.7 m against 9.3 m.
- B stays implemented and declared as an option. If the organisers confirm the fused fields are allowed,
  the owner can switch with `./run.sh <test> <out> -- --estimator vismap_est --state ambiguous`.

## Consequences
- The README declares the fields used.
- `tests/test_smoke.py` keeps checking that pose-derived fields are never whitelisted.
- The question to the organisers stays open (REPORT.md §9 Q1).
