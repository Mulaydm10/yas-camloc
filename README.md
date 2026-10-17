# yas-camloc: camera-based localisation on the Yas Marina track

Constructor Labs challenge, National AI Hackathon 2026. A causal localiser estimates the race car's 2D pose
(`x, y, yaw` of `constructor0/base_link` in `map`) at camera rate from its 7 cameras. It runs on CPU in real time.

**Results:** see [Results](#results). The full analysis is in `REPORT.md`, and the experiment history is in
`experiments/experiment_log.md`.

## Run it

```bash
./run.sh <test.mcap | test_dir> <out.csv> [training.mcap]
```

- `run.sh` creates the Python 3.12 venv from `requirements.txt` if missing.
- It builds the training-derived visual map if missing. That needs the training bag; the map is derived from
  the dataset, so it is not committed.
- It then writes `out.csv` (`timestamp_ns,x,y,yaw`) and `out.run.json`, which holds the per-frame processing
  time and the hardware.
- If the test file has no `/tf` first pose, pass it with
  `.venv/bin/python -m camloc.run --mcap … --out … --estimator vismap --init x,y,yaw`.

## Method (all causal: the pose at t uses only data stamped ≤ t)

```
 wheel speeds + steering ──► odometry (anti-slip axle choice, bicycle + understeer + steering lag)
                                   │ predict
 fl / fr / r images ──► keyframe rotation (LK + essential matrix) ──► relative-yaw update ─┐
                   └──► ORB vs geo-referenced 3D map ──► PnP + RANSAC ──► absolute pose fix ─┼─► EKF [x, y, yaw, yaw_kf]
 track borders (map) ──► lane model: track / pit lane, speed-gated switching; ────────────────┘
                         heading + lateral priors on straights when no map fix for > 2 s
```

1. **Odometry from allowed car signals only.**
   - Wheel speeds and steering, with parameters fitted on the training run (`tools/calibrate_odometry.py`).
   - The fused `state_estimation` speed and yaw-rate fields are **not used** by default. Their status is an
     open rules question; see `REPORT.md` §5.
2. **Visual heading.**
   - The rotation since a keyframe at most 1.5 s old is measured in `fl`, `fr` and `r` and fused through a
     cloned heading state.
   - Per-camera rotation scale (`fr` under-reports by 7.5 %), with a two-camera consensus.
3. **Map-based absolute fixes** (images contrast-normalised with CLAHE at build and query time).
   - At training time, ORB features are triangulated with the GT camera poses into a 3D map
     (`tools/build_map.py`).
   - At run time, per-camera PnP + RANSAC, gated by cross-camera consensus and an NIS test that can recover
     if the filter itself is wrong.
4. **Track model.**
   - The surveyed borders give the track and pit-lane centrelines, with (s, n) projection that respects heading.
   - Lane membership is tracked and switches at the pit branch points only when the car is slow, at pit speed.
   - The 3D borders give the road z, roll and pitch.
5. **Camera self-calibration.**
   - Rotation corrections of 1° or less per camera, fitted on training GT (`tools/calibrate_cameras.py`).
   - Ego-body masks are hand-drawn polygons (`camloc/assets/body_polygons.json`).

Pre-trained models and external data: **none**. The training run's GT is used only at training time (odometry,
camera and road calibration, the visual map), for scoring, and for the first pose. The test
`tests/test_smoke.py::test_inference_modules_do_not_import_groundtruth` enforces that inference code never
imports the GT modules.

## Results

All numbers come from the training run with an **honest hold-out**: for each lap the visual map is rebuilt
**without that lap**, the localiser runs causally over the whole recording using only the allowed car signals,
and only the held-out lap is scored with the official metric (`camloc/evaluate.py`, which reproduces the
organiser's `tools/evaluate.py`).

| held-out lap | position RMSE | lateral | longitudinal | heading RMSE | p95 | within 1 m |
|---|---|---|---|---|---|---|
| 2 | 0.905 m | 0.536 m | 0.729 m | 0.29° | 1.82 m | 86.3 % |
| 4 | 0.657 m | 0.480 m | 0.449 m | 0.20° | 1.20 m | 86.8 % |
| 6 | 0.659 m | 0.475 m | 0.456 m | 0.22° | 1.28 m | 90.8 % |
| **pooled (3 laps)** | **0.78 m** | | | | | **88 %** |

- **Strict hold-out.** Lap 4 is also scored with every calibration (odometry, camera rotations, rotation
  scales) refitted without it (`tools/holdout.py`): 0.677 m against 0.654 m with the same pipeline, so
  calibration leakage is about 0.02 m.

![lap 4 held out](docs/traj_lap4_heldout.png)

- **Pit lane:** 1.13 m (whole run, map without lap 6).
- **Organiser (s, n) scorer:** agrees with the x/y numbers within 1 %. GT (s, n) is re-projected onto
  `map.mcap`, because the training bag's `sn_map_state` does not match it (`REPORT.md` §11).
- **Speed:** 22–27 ms per frame mean, 35 ms p95, on an Apple M4 CPU (input at 10 Hz). The full 20-minute
  run takes about 5 minutes.

How we got here (whole run, allowed signals only; `experiments/experiment_log.md`):

| step | position RMSE |
|---|---|
| hold first pose | 528 m |
| + wheel / steering odometry | 144 m |
| + map priors on straights | 55.1 m |
| + visual heading | 21.5 m |
| + map-based PnP fixes | ≈1 m |
| + NIS gate, pit lane in map, lower single-camera threshold | 0.81 m held out |
| + contrast-normalised (CLAHE) map | **0.78 m held out** |

### Robustness (the test may differ from the night training run)

**Image changes.** Query images were altered on held-out lap 4 (`tools/vmap_check.py --aug`); the map is
built from the unchanged night images. Columns give the `fr` map-localisation success rate:

| query images | plain ORB map | CLAHE-normalised map (default) |
|---|---|---|
| unchanged | 60 % | 82 % |
| darker (gamma 2) | 42 % | 72 % |
| contrast halved | 22 % | 66 % |
| washed out ("day": gamma 0.4, contrast 0.6) | 20 % | 60 % |
| 9 px motion blur | 11 % | 43 % |

When a fix succeeds it stays accurate: median 0.4–0.5 m in every case.

**Losing map fixes.** These are switches for testing only (`--set FIX_DROP=…` / `FIX_GAP_S=… FIX_PERIOD_S=…`):

| map fixes | lap 4 | whole run |
|---|---|---|
| all | 0.65 m | 0.84 m |
| 50 % dropped at random | 0.70 m | 0.97 m |
| 90 % dropped at random | 0.88 m | 2.34 m |
| none for 30 s of every 60 s | 2.48 m | 3.10 m |
| none for 60 s of every 120 s | 2.86 m | 4.60 m |

The filter recovers as soon as fixes return. Only a total, permanent map failure would fall back to the
visual-heading-only result (21.5 m).

**Known failure cases** (`docs/traj_lap2_heldout.png`):
- The back straight (around x ≈ −300, y ≈ 0–250) is feature-poor at night. With the plain map, lap 2 drifted
  2–5 m sideways there. The CLAHE map removes that drift, but one 3.7 m along-track step remains for 3 s.
- Two brief lateral spikes of about 2.5 m occur in the slow corners. A real daylight recording has not been tested; the augmentations above
are only a proxy for it.

## Repository

| path | content |
|---|---|
| `camloc/` | the localiser: `data.py` (causal replay), `track.py`, `odometry.py`, `filter.py`, `vision_rot.py`, `vmap.py`, `road.py`, `geometry.py`, `run.py` |
| `camloc/tools/` | training-time calibration, map building and diagnostics (these may use GT) |
| `camloc/evaluate.py` | scoring harness: the official metric (verified equal to `tools/evaluate.py`) plus breakdowns by sector, speed, lane, lap and GT quality, and the organiser (s, n) scorer via `--sn` |
| `analysis/`, `REPORT.md` | dataset analysis and the decisions behind the design |
| `docs/` | trajectory plots of validation laps |
| `tests/` | `pytest -q tests` |

Agent and team workflow files: `CLAUDE.md`, `STATE.md`, `worklog.md`, `AGENTS.md`, `GOVERNANCE.md`.
