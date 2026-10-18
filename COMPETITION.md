> **LOCKED governing file.** Do not edit in place. See `GOVERNANCE.md`.

# COMPETITION - single source of truth for event facts

Every other doc **links here** instead of restating a date, weight or rule. A duplicate found elsewhere should be deleted and replaced with a link. A bare time with no timezone has caused real missed submissions: always write date + time + timezone. Never guess a rule; a wrong assumed rule can disqualify. Registered as Q-0002.

| Field | Value |
|---|---|
| Event name | National AI Hackathon 2026 (licence text of the dataset, brief: `notes/brief/Constructor Labs - Camera-based localization on a race track.pdf`) |
| Host / organizer | SchubKI (hackathon host, "National AI Hackathon 2026 by SchubKI"); Constructor Labs (challenge owner) |
| Event link | TODO(<owner>) |
| Theme / track | Camera-based localization on a race track (Yas Marina Circuit) |
| Start | 2026-10-16 (opening). Brief: "16-21 of October 2026 (16 – opening, 17-18 – work, 19-20 – evaluation, 21 - closing)". Time and timezone: not stated, TODO(<owner>) |
| **Submission deadline** | **2026-10-18 23:59 CEST (21:59 UTC)**: organisers via Discord #camera-localization, relayed by Devin on GitHub #2 (TASK I, 2026-10-18 14:25 UTC). Submission form: https://forms.cloud.microsoft/r/Cu6YHzCwdZ (max 6 slides as PDF, video max 5 min, trajectory visualisation, all team members). The repo must be **public** at submission ("private repo = disqualification"). |
| Demo / judging time | Evaluation 2026-10-19 to 2026-10-20; closing 2026-10-21. Times / timezone: TODO(<owner>) |
| Submission format | Pose CSV for the hidden test run (`timestamp_ns, x [m], y [m], yaw [rad]`, track frame, stamp = image header stamp); full source code (Git link or archive) with a run script; validation results (scoring-script output); presentation (max. 5 min, max. 6 slides); trajectory-vs-GT video or plot. Upload channel / form: TODO(<owner>) |
| Team size limit | TODO(<owner>) |

## Judging rubric
| Criterion | Weight | Notes |
|---|---|---|
| Absolute trajectory error: position RMSE [m] | 25 % | quantitative, hidden test laps |
| Lateral error RMSE across the GT heading | 20 % | quantitative; **ties broken by lower lateral error** |
| Heading error RMSE [deg] | 10 % | quantitative |
| Robustness: share of poses with position error < 1 m | 10 % | quantitative |
| Runtime: average processing time per frame (reported) | 5 % | full points at or below the camera frame period |
| Technical soundness and novelty | 15 % | jury |
| Engineering quality and reproducibility of the code | 10 % | jury |
| Presentation, incl. failure cases and limits | 5 % | jury |

Quantitative total 70 %, qualitative 30 %. Each quantitative metric is normalised across teams (best = full points).

## Team roster
| Name | Role | Contact | Notes |
|---|---|---|---|
| Tamo Schäfers | team Rogue Ravens | Discord team channel | |
| Dhruv Mulay | team Rogue Ravens (repo owner) | GitHub Mulaydm10 | |
| Gagan Karthik Sripuram | team Rogue Ravens | Discord team channel | |
| Taa-seen Ahmed | team Rogue Ravens | Discord team channel | |

## Hard rules
| Rule | Value |
|---|---|
| What may be pre-built before the event | TODO(<owner>) |
| Licensing / open-source requirement | Dataset: for use in the National AI Hackathon 2026 only; no redistribution or commercial use without written permission from Constructor Labs. Code licence: TODO(<owner>) |
| API-key and rate-limit constraints | TODO(<owner>) |
| Rules on AI-generated code | TODO(<owner>) |
| Allowed/forbidden tools, datasets, models | Inputs: camera images, camera calibration, track borders; car state data (accelerations, speed and other fields) may be used; organisers (Discord, 18 Oct): speed, accelerations and yaw rate from state_estimation may be used, pose fields may not, and **no starting pose is given**. The organisers' test-domain sample may NOT be used to train, tune or validate. No LiDAR or radar. Ground truth only for training/validation, never at inference except the first pose of the run. Pre-trained models and public external datasets allowed but must be declared. |
| Causality | Causal (online) only: the estimate for frame t uses data up to t. Offline/smoothed results may be shown but are not ranked. |
| Output rate and coverage | Poses at camera rate (10 Hz expected); at least 90 % of 1 s intervals must have a pose, or the run is not ranked. |
| Reproducibility | Repository with README, dependency list and a single run script from the raw dataset; must run on a single-GPU workstation; organisers may re-run it. Measured average time per frame reported with the hardware. |
| Disqualification | Missing source code, results the code does not reproduce, ground truth at inference, or non-causal processing in the ranked result. |
| Test conditions | Hidden test run (separate file, first pose only) may include higher speeds and different lighting than the night training run. |
