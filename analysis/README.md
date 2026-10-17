# analysis/: dataset analysis scripts (camera-only localization)

Analysis only, no localization method. Outputs feed `REPORT.md` at the repo root.

## Setup
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p data && curl -C - -L -o data/camloc.zip "https://eu1-s3.virtuozzo.com/dev-auto-mobility-data/hackaton_2026/c0_2026-10-13_22-53-42_camloc.zip"
cd data && unzip camloc.zip
```
The dataset licence allows hackathon use only, so `data/` and `analysis/out/` are git-ignored. Never commit or upload them.

## Run (from `analysis/`, in order; total ≈ 4 min on an M4)
| Step | Script | Time | Produces |
|---|---|---|---|
| 1 | `../.venv/bin/python dump_meta.py` | ~50 s | `out/meta.json`, `out/gt.{npy,csv}`, `out/state_est.npz`, `out/state_est_sample.txt`, `out/track_bounds.npz`, `out/image_index.npz` |
| 2 | `../.venv/bin/python dump_frames.py` | ~60 s (7 processes) | `out/cam_stats.npz`, `out/pixstats_*.npz`, `montage_{60,330,600,900}s.png` |
| 3 | `../.venv/bin/python track_gt.py` | ~60 s | `out/track_model.npz`, `out/gt_track.npz`, `out/track_gt.json`, `traj_speed.png`, `time_series.png`, `track_profile.png` |
| 4 | `../.venv/bin/python cameras.py` | ~5 s | `out/cameras.json`, `body_mask.png`, `cam_stats_time.png` |
| 5 | `../.venv/bin/python fields.py` | ~5 s | `out/fields.json` |
| 6 | `../.venv/bin/python deadreckon.py` | ~60 s | `out/sub_*.csv`, `out/deadreckon.json`, `deadreckon_err.png`, `deadreckon_traj.png` |
| 7 | `../.venv/bin/python sectors.py` | ~5 s | `out/sectors.json`, `sectors.png` |
| 8 | `../.venv/bin/python sector_frames.py [k ...]` | ~10 s | `montage_S<k>.png` |

Every script makes a streaming pass, or reads only the products of earlier steps, so the 7 GB MCAP is never loaded into memory.

## GT-use rules followed
GT (`/tf`, and the pose fields of `state_estimation`) is used only for:
- statistics and plots;
- scoring;
- the first pose;
- clearly labelled diagnostics: `deadreckon.py` run (c) resets from GT; the rolling-radius, wheelbase and steering-offset fit is a training-time calibration.
