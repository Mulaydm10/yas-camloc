# Claude Code prompt: Yas Marina camera-only localization, dataset analysis

Paste everything below the line into Claude Code, started in an empty working folder on the Mac mini (e.g. `~/camloc`).

---

You are helping me prepare for the Constructor Labs "Camera-based localization on a race track" challenge (National AI Hackathon 2026). **This task is analysis only: do not build the localization system yet.** At the end, give me a written report plus plots. Be honest: report only what you measured and its scope, and never present a narrow metric as "overall accuracy".

## Context
- Goal of the challenge: estimate a full-size autonomous race car's 2D pose (`x, y, yaw` in the `map` frame) at 10 Hz from its 7 onboard cameras, on the Yas Marina circuit.
- Not allowed as method input: GNSS, LiDAR, radar, IMU, ground-truth poses (the first pose is the only exception). The PDF says car-state data (speed, accelerations) **is** allowed. The README calls the whole `/constructor0/state_estimation` topic ground truth. Treat this as an **open rules question** and keep "allowed" and "GT-derived" fields strictly separate.
- Processing must be causal: the pose at time t may only use data stamped ≤ t.
- Submission is a CSV `timestamp_ns,x,y,yaw`. At least 90% of 1-second bins must contain a pose.
- Scoring: 70% metrics (position RMSE 25%, lateral RMSE 20%, heading RMSE 10%, share of poses within 1 m 10%, runtime 5%) and 30% jury.
- The training run is at night. The hidden test may be faster and/or in different lighting (assume daylight is possible).
- Dataset licence: hackathon use only. Do not redistribute or upload it anywhere.

## Step 0: Setup
1. Report the machine: `sysctl -n machdep.cpu.brand_string`, RAM, macOS version, free disk space. You need about 16 GB free.
2. Create a Python venv and install `mcap mcap-ros2-support numpy scipy opencv-python matplotlib pandas`. Record the exact versions in `requirements.txt`.
3. Download the dataset (7.4 GB ZIP, resumable). A plain HEAD request returns 403, which is normal; GET works:
   `curl -C - -L -o data/camloc.zip "https://eu1-s3.virtuozzo.com/dev-auto-mobility-data/hackaton_2026/c0_2026-10-13_22-53-42_camloc.zip"`
   Then unzip into `data/`. You should get `c0_2026-10-13_22-53-42_camloc/` containing one `.mcap` file (about 7.5 GB), `README.md`, and `tools/extract_gt.py` + `tools/evaluate.py`.
4. Read `README.md` and both tools fully before writing any code. Summarise anything that differs from this prompt.

## Step 1: Inventory (put all scripts in `analysis/` and make them re-runnable)
Write `analysis/dump_meta.py`, which makes **one** streaming pass over the MCAP (it is large, so never load it all into memory) and saves:
- Topic list with message counts, rates and message types.
- `/tf` `map -> constructor0/base_link` as `gt.npy` / `gt.csv` (t_ns, x, y, z, yaw). Cross-check it against `tools/extract_gt.py`.
- `/constructor0/state_estimation`: flatten every numeric field into `state_est.npz` (keys + matrix). Also dump one full message as text.
- `/constructor0/track_bounds`: the polylines `track_left`, `track_right`, `pitlane_left`, `pitlane_right`.
- `/tf_static`: camera extrinsics (base_link -> camera_*). `camera_info`: K, D, width/height for each camera.
- Save everything to `meta.json`.

Write `analysis/dump_frames.py`: for each of the 7 `compressed_image` topics, record header stamp, log time, JPEG size, mean brightness, and Laplacian variance (sharpness) for every frame (`cam_stats.npz`). Decode full images only at a few timestamps (e.g. t = 60, 330, 600, 900 s) and save a labelled 7-camera montage for each.

## Step 2: Analysis to report (with numbers)
1. **Cameras:**
   - Per camera: fx/fy, distortion, mount position/orientation, frame period (median and max gap), dropped frames.
   - Sync across cameras: nearest-stamp difference against `fl`.
   - Header-to-log latency, brightness range, sharpness.
   - Which image regions show the car's own body (to be masked).
   - The README calibration caveats (which yaw values are reliable, roll/translation uncalibrated), and how much 1° of yaw error costs in metres at 10/30/60 m range.
2. **Ground truth:**
   - Duration, rate, max gap (the evaluator rejects samples with a GT gap > 50 ms).
   - Speed stats (max, mean, p95), yaw-rate max, total distance.
   - Number of laps, lap times, which circuit layout this is.
3. **Track map:**
   - Total length; width distribution (left vs right border distance, sampled along the centreline).
   - Curvature profile along the track.
   - Pit-lane share of time and number of pit passes.
   - Plot the trajectory coloured by speed over the borders, plus a time plot of speed, yaw rate and the in-pit flag.
4. **Car-state fields:**
   - For every field: range, rate, and whether it is constant, flagged invalid, or derived from the GT pose. For example, `sn_map_state` (track coordinates) is computed from the pose, so it must **never** be an input. `valid_imu_b` is always false.
   - Make a table: **allowed / ambiguous / forbidden**, with the reasoning for each.
5. **Dead-reckoning baselines** (causal, initialised from the first GT pose only; output at the `camera_fl` stamps):
   - (a) speed `v_mps` + yaw rate `wz_radps`;
   - (b) wheel speeds (fit the rolling radius) + steering angle with a bicycle model (fit the wheelbase);
   - (c) the same as (a) but re-initialised from GT every N seconds (N = 5, 10, 30), to show how quickly drift grows.
   - Score each with `tools/evaluate.py`. Also report the time until the error first exceeds 1 m / 5 m / 20 m.
   - For reference, a previous run of (a) gave position RMSE ≈ 65 m, the 1 m error was first exceeded at ≈ 15 s, and only 3% of poses were within 1 m. Confirm or correct this.
6. **Visual conditions:**
   - Look at the montages and describe the usable cues per camera: kerbs, white lines, blue run-off paint, barriers, light poles, gantries such as the Etihad bridge, grandstands.
   - Describe the problems: glare, haze, blur at speed, low-texture straights, repeated structures.
   - Say which cues would survive daylight and which are night-only.
7. **Sector difficulty:**
   - Split the lap into ~10 sectors and report speed, curvature, and visual-cue richness for each (rough proxy: edge density or ORB keypoint count in the wide cameras).
   - Flag the sectors where localization will be hardest (long straights, high speed, pit entry/exit).

## Step 3: Recommendation (written, no implementation)
Based on the numbers, recommend an architecture and a first-week plan. My current hypothesis; confirm or challenge it with evidence:
- A causal filter (EKF or particle filter) in track coordinates: progress along the track s, lateral offset n, heading error.
- Prediction: car speed + yaw rate if those fields are allowed, otherwise visual odometry from the cameras.
- Lateral + heading update: detect track edges/kerbs/lines in the wide and front cameras, project them to the ground plane with the extrinsics, and match them to the surveyed borders (chamfer/ICP).
- Longitudinal update: match measured curvature/yaw rate to the map's curvature profile, match against geotagged keyframes from the training laps, and use fixed landmarks.
- Self-calibrate camera extrinsics from the training GT (allowed: it is training).
- No VLMs, no large models in the loop. It must run in real time on one GPU, ideally on the Mac mini CPU.
- Validation: hold out laps, and report error by sector, speed and pit lane, not just the overall RMSE. Add lighting/blur augmentation because we only have a night session.

Also list the questions I should ask the organisers. At minimum: exactly which `state_estimation` fields are allowed at inference, whether the test uses the same layout and camera calibration, and the test lighting and speed.

## Deliverables
- `analysis/*.py` (re-runnable, with a short README explaining how to run them), `requirements.txt`.
- `REPORT.md`: a 1-page executive summary first, then a section per step with tables and plots (`analysis/*.png`).
- Keep the raw data out of git. If you initialise a repo, add `data/` to `.gitignore`.
- Do not use future data, and do not use GT anywhere except the first pose, the scoring, and the explicitly labelled diagnostic experiments (5c, extrinsic self-calibration).
