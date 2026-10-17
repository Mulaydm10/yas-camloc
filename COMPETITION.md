> **LOCKED governing file.** Do not edit in place. See `GOVERNANCE.md`.

# COMPETITION - single source of truth for event facts

Every other doc **links here** instead of restating a date, weight or rule. A duplicate found elsewhere should be deleted and replaced with a link. A bare time with no timezone has caused real missed submissions: always write date + time + timezone. Never guess a rule; a wrong assumed rule can disqualify. Registered as Q-0002.

| Field | Value |
|---|---|
| Event name | National AI Hackathon 2026 (licence text of the dataset, brief: `notes/brief/Constructor Labs - Camera-based localization on a race track.pdf`) |
| Host / organizer | Constructor Labs (challenge owner) |
| Event link | TODO(<owner>) |
| Theme / track | Camera-based localization on a race track (Yas Marina Circuit) |
| Start | TODO(<owner>) (date, time, timezone) |
| **Submission deadline** | TODO(<owner>) (date, time, **timezone**) |
| Demo / judging time | TODO(<owner>) (date, time, timezone) |
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
| TODO(<owner>) | TODO(<owner>) | TODO(<owner>) | |

## Hard rules
| Rule | Value |
|---|---|
| What may be pre-built before the event | TODO(<owner>) |
| Licensing / open-source requirement | Dataset: for use in the National AI Hackathon 2026 only; no redistribution or commercial use without written permission from Constructor Labs. Code licence: TODO(<owner>) |
| API-key and rate-limit constraints | TODO(<owner>) |
| Rules on AI-generated code | TODO(<owner>) |
| Allowed/forbidden tools, datasets, models | Inputs: camera images, camera calibration, track borders; car state data (accelerations, speed and other fields) may be used. No LiDAR or radar. Ground truth only for training/validation, never at inference except the first pose of the run. Pre-trained models and public external datasets allowed but must be declared. |
| Causality | Causal (online) only: the estimate for frame t uses data up to t. Offline/smoothed results may be shown but are not ranked. |
| Output rate and coverage | Poses at camera rate (10 Hz expected); at least 90 % of 1 s intervals must have a pose, or the run is not ranked. |
| Reproducibility | Repository with README, dependency list and a single run script from the raw dataset; must run on a single-GPU workstation; organisers may re-run it. Measured average time per frame reported with the hardware. |
| Disqualification | Missing source code, results the code does not reproduce, ground truth at inference, or non-causal processing in the ranked result. |
| Test conditions | Hidden test run (separate file, first pose only) may include higher speeds and different lighting than the night training run. |
