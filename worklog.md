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

## 2026-10-17 23:45 CEST - Claude (Opus) - D5 finished: gating, folds, single-camera threshold, speed, submission README (EXP-0012..0016)
- `run.VisualMapLoc`: NIS gate with recovery, map priors only after 2 s without a fix, `MIN_INL_SINGLE` 30 -> 18; `--set NAME=VALUE` / `--init x,y,yaw` options.
- `tools/build_map.py` maps the pit lane too (pit lane 6.7 -> 1.05 m). `evaluate.py --sn`: organiser (s, n) scorer with GT re-projected on `map.mcap` (the bag's `sn_map_state` does not match it).
- Validation, each lap with a map built without it: lap 2 0.947 m, lap 4 0.654 m, lap 6 0.700 m, pooled 0.81 m (86 % within 1 m). Failure case: the feature-poor back straight.
- Speed: thread pool per camera, decode once, direct CDR image parsing: 38 -> 22 ms/frame, output bit-identical.
- `run.sh` single entry point (smoke-tested end to end); README results filled; plots in `docs/`; REPORT.md §11 (results, organiser update, questions 11-13).
- Two Claude sessions overlapped (screen session national-e8 wrote EXP-0012/0013 rows); duplicate row removed, session closed on the user's request. 19 tests green.

## 2026-10-18 00:45 CEST - Claude (Opus) - Devin review (issue #1) resolved (EXP-0017..0019)
- Stress test of the fallback: map-fix dropout / blackouts give metres, not tens of metres (EXP-0017).
- Robustness: query-image augmentations in `tools/vmap_check.py --aug`; CLAHE-normalised map (`build_map --norm`, flag in the map) raises map success under altered images 2-4x and improves the held-out pooled score 0.81 -> 0.78 m (EXP-0018). Now the default in `run.sh`.
- Strict hold-out: `tools/holdout.py`, `CAMLOC_EXCLUDE_LAPS` / `CAMLOC_ASSETS` (config.ASSETS); leakage about 0.02 m (EXP-0019).
- `build_map.py` exclusion leak fixed (both frames of a pair checked). 20 tests green. REPORT.md §12 maps each review point to its resolution.

## 2026-10-18 01:30 CEST - Claude (Opus) - Devin review round 2 (EXP-0020..0024, ADR-0003)
- 2x2 ablation found CLAHE behind the lap 4 lateral regression; threshold back to 30 with the CLAHE map (pooled 0.775 -> 0.673 m).
- Fix-quality weighting (inliers, reprojection RMS) on by default: pooled 0.630 m, lateral 0.477 m, 91.0 % within 1 m.
- Lateral bias traced to sideslip (0.23 deg + 0.14 deg per m/s^2 from GT); modelled but off (helps laps 4/6, hurts lap 2).
- Fused car-state variant `vismap_est`: lateral 0.430 m but worse in blackouts (9.3 m vs 4.7 m whole run) -> ADR-0003 submits allowed-only.
- Paint/kerb lateral update not built: EXP-0006 showed it is unobservable with these mounts.
- run.sh: caller-relative paths, `--` option passthrough; fresh-clone test (EXP-0024). submissions/validation, docs/SLIDES.md, README declarations.

## 2026-10-18 01:55 CEST - Claude (Opus) - Owner confirmed ADR-0003; COMPETITION.md filled
- ADR-0003 accepted: the submitted variant is allowed-only `vismap`.
- COMPETITION.md (LOCKED) filled from the challenge brief as Main Agent on the owner's instruction; audit row in GOVERNANCE.md. Deadline, dates, team size, roster and upload channel are not in the brief and stay TODO(<owner>).
- notes/judging_alignment.md: every rubric criterion mapped to an artifact.

## 2026-10-18 04:40 CEST - Claude (Opus, overnight in screen) - Devin tasks A-G (EXP-0025..0029)
- Handshake with Devin on #2 (TASK/ACK/DONE); watcher re-armed every 30 min.
- A: half-gain sideslip kept (pooled 0.586 m), later superseded. B: COMPETITION.md dates + host from the brief.
- D: fix log shows S7 fixes present and unbiased; the filter under-corrects -> online sideslip states b + c*a_y with lateral Q. Found and fixed a --set bug (constructor parameters applied too late; regression test).
- E: camera ~0.2-0.4 m higher above the paint than the map datum, but paint lateral scatter 0.44-0.74 m: closed.
- F: phase-based axle choice; root cause of the strict lap 2 blow-up was locked front wheels under 42 bar braking with a front-axle fit. Strict pooled 0.513 m / 0.303 m / 95.1 %.
- C: freeze pass from a clean clone (byte-identical map, 100 % coverage, 20.3 ms/frame). G: glare halves map success; extra blur for 45+ m/s nearly defeats ORB matching (open risk, mitigation proposed to Devin).

## 2026-10-18 08:25 CEST - Claude (Opus, screen session) - Devin TASK H (EXP-0030..0032)
- H.1: blur_vs_speed + exposure_fit: no measurable blur growth with speed; physical flow-field blur fits front exposure ~0.5-1 ms; EXP-0029 speed rows relabelled as an unphysical worst case.
- H.3: map-point tracking and lower FAST threshold explored; cannot seed under synthetic heavy blur; parked (default off).
- H.2: gated reset, corner lateral bounds, online wheel-scale state, DCS soft gate; plus wrong-fix injection test. None adopted (no clear win on normal + stress); default passes wrong-fix injection. Phase odometry has ~0 signed speed bias.
- Session restart mid-way (runs survived); orphan watcher cleaned; EXP timestamps corrected.

## 2026-10-18 10:10 CEST - Claude (Opus, screen session) - Devin TASK H (EXP-0030..0034)
- H.1: real blur does not grow with speed; a physical flow-field blur fitted to real sharpness gives front exposure ~0.5-1 ms; the V/3 px stress rows relabelled as an unphysical worst case.
- H.2: blackout regression traced (+6 % whole run; lap 4 window = phase-odometry trade-off). Gated reset, corner bounds, online wheel scale and DCS soft gate tested on strict folds + blackouts + wrong-fix injection: none passes.
- H.3: LK map-point tracking (age-scaled sigma) rejected; full-pipeline glare costs lap 4 0.46 -> 1.12 m.
- Defaults unchanged since f49ab7f; all options default-off and documented.

## 2026-10-18 13:40 CEST - Claude (Opus, screen session 57650) - duplicate-session cleanup
- A forked duplicate session (screen 68230) had worked TASK H in parallel on the same working tree (commits 280a091, 79c87e2; "DONE H.2" on #2 at 06:18 UTC). Peer session national-18 closed it on the owner's instruction; this session is now the only one working with Devin.
- Its EXP-0032 row duplicated an ID; renumbered to EXP-0035 with a provenance note (content overlaps EXP-0032/0033). No code conflict: the history is linear, and defaults are unchanged since f49ab7f.

## 2026-10-18 17:50 CEST - Claude (Opus) - TASK I: no starting pose (EXP-0036)
- Organisers (via Devin): the test file has no starting pose; the sample layout has no state_estimation, /tf or track_bounds; deadline 23:59 CEST; the repo must be public.
- Global initialisation over the whole map, locked on multi-frame consensus; no pose before the lock; re-localisation after 15 s without fixes (time-boxed: one camera per attempt). First lock rule gave 2/16 wrong locks; stricter rule gives 0/16. Strict laps 2/4/6 identical to GT init.
- I.0: topic resolution by suffix, bounds asset fallback (exported by run.sh), validation/twist as the state input when state_estimation is absent; run.sh falls back to Python >= 3.10 with requirements-compat.txt (tested on 3.10).
- Devin crash-tested the organisers' sample: right format, 74/74 1-s bins, 84 ms/frame after time-boxing.

## 2026-10-18 18:15 CEST - Claude (Opus) - TASK J and K: provisional lock, evidence-based switch (EXP-0037, EXP-0038)
- J (016ba24): two cameras agreeing in one frame give a provisional lock (sigma 3 m), confirmed by 2 accepted fixes; soft fallback at 6 s, best-hypothesis fallback at 10 s. The aliased lap 6 @ 70 % start recovers (1.42 m).
- K: Devin's sample re-run showed a wrong daylight revoke, with the region exclusion delaying recovery by about 25 s and a p95 of 1020 ms. Now a revoke needs >= 3 mutually consistent rejected fixes >= 5 m away, and it switches to that pose. There is no exclusion and no new search. Strict 0.513 / 0.303 / 95.1 % unchanged; lap 6 @ 70 % switches after 0.3 s (1.04 m); the other 15 starts are identical. 26 tests.
- L (18:30 CEST): on the sample, the early lock never fired after dfe3fdf. Its 0.6 s lock had been the pair-free 3-frame rule, which I.1 had made require a two-camera pair. That requirement is now off (GI_REQUIRE_PAIR=0) and the lock is provisional. Strict unchanged; 16/16 starts end correct; GI_LOG diagnostic added.
- Freeze (18:45 CEST): Devin verified f314e61 on the sample (strict lock 0.6 s, confirmed, 74/74 bins) and froze it. `./run.sh` with no flags checked on the training layout and on a synthetic test-domain cut. The 30 pre-lock poses at the stationary pit start of the training file are 15 m off (the never-silent best hypothesis); recorded, not changed (freeze).
