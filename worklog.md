# worklog.md - APPEND-ONLY, never edit past entries

Entries are dated **and timestamped with timezone** (hackathon history moves hourly). Newest at the bottom.
Format: `## YYYY-MM-DD HH:MM TZ - author - summary`, then bullets citing IDs.

## 2026-10-16 - hackathon-setup agent - initial scaffold
- Created repo skeleton per ADR-0001. Stack undecided (ADR-0002). No event facts known (Q-0002). Clock time not recorded; timezone unknown.

## 2026-10-17 05:54 CEST - Claude (Opus) - camloc dataset analysis (analysis only, no method)
- Brief: `notes/brief/claude-code-camloc-analysis-prompt.md` + challenge PDF (Constructor Labs, camera-based localization, Yas Marina).
- Downloaded dataset to `data/` (git-ignored, licence: hackathon only). Venv `.venv` (py3.12), pins in `requirements.txt`.
- Scripts in `analysis/` (README there); results in `REPORT.md`.
- Key numbers: dead reckoning v+wz 65.5 m RMSE (1 m exceeded at 12.6 s); allowed-only wheels+bicycle 154 m; GT resets every 5 s 0.64 m.
- Open rules question: README calls all of state_estimation GT; PDF allows car state. wz_radps/v_mps are fused estimator outputs.
- Event facts in `COMPETITION.md` still TODO (LOCKED file, not edited); the PDF in notes/brief has them.

## 2026-10-17 06:06 CEST - Claude (Opus) - D1: camloc package (ADR-0002, EXP-0002)
- `camloc/data.py`: causal replay (stamp-ordered heap, lag 250 ms, 0 late drops on the training run), 7-camera FrameSets stamped with the latest camera stamp, car-state whitelist (allowed / ambiguous opt-in).
- `camloc/track.py`: track (3006 m, CCW) + pit lane (1122 m) centrelines, (x,y)<->(s,n,epsi), widths, curvature, pit branch points (entry s~1980, exit s~96 on the track).
- `camloc/evaluate.py` + `groundtruth.py` (eval only): official metric reimplemented and cross-checked, breakdown by GT quality, lane, sector, speed, lap.
- `camloc/run.py`: single entry point MCAP -> CSV with per-frame timing; `hold` estimator only (smoke).
- 8 pytest tests green. LOCKED `tests/README.md` still says "no baseline": needs the Main Agent to update it.

## 2026-10-17 06:19 CEST - Claude (Opus) - D2: odometry, EKF, map priors, ego-body masks (EXP-0003, EXP-0004)
- `camloc/odometry.py` + `tools/calibrate_odometry.py` (training-only, GT): params in `camloc/assets/odometry.json`.
- `camloc/filter.py`: PoseEKF (x, y, yaw) with track-coordinate measurements; straights-only map priors.
- Masks: automatic extraction (temporal std, median frame difference) failed at night (noise-dominated); hand polygons on mean images in `camloc/assets/body_polygons.json`, rendered by `tools/make_masks.py`.
- Results: allowed-only 55.1 m, ambiguous 13.8 m whole-run RMSE. 12 tests green.

## 2026-10-17 06:35 CEST - Claude (Opus) - D3: geometry, line measurement, camera self-calibration; plan pivot (EXP-0005..0008)
- New: `camloc/geometry.py` (projection), `lines.py` (paint detector), `align.py` (chamfer, ground ROI), `measure.py` (near-border residuals), `road.py` (map-based z/roll/pitch), `data.load_bounds3d`.
- Training tools: `calibrate_cameras.py` (rotations -> camloc/assets/camera_adjust.json), `fit_road_attitude.py`, `overlay_borders.py`, `vo_rotation_check.py`, `lateral_sensitivity.py`.
- Finding: lateral from ground paint is unobservable (cameras ~0.2 m above the paint). REPORT.md section 10 revises D4/D5: visual rotation for heading, height structures/landmarks for s and lateral.
- 15 tests green.

## 2026-10-17 19:30 CEST - Claude (Opus) - D4: visual heading (EXP-0009)
- `camloc/vision_rot.py` KeyframeRotation; PoseEKF state extended with a heading clone (`clone_heading`, `update_relative_yaw`).
- `run.VisualHeading` (`--estimator vis`): fl/fr/r keyframe rotation, scale-corrected, consensus-gated; tracked lane state with speed-gated pit switching; heading-aware lane projection (`Lane.project(..., yaw)`).
- Allowed-only whole-run RMSE 55.1 -> 21.5 m (yaw 1.60 deg). 17 tests green.
- New dataset `23-00-38_cut_120m_update.zip` (SharePoint) not downloaded yet: needs a Microsoft sign-in (Playwright browser opened at the login page).
