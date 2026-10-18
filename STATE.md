# STATE.md - live snapshot (OVERWRITE every session)

If STATE and `worklog.md` disagree about what is true *now*, STATE wins; the worklog explains how we got here.
Deadline: see `COMPETITION.md` (TODO(<owner>)). Time remaining: TODO(<owner>)

## Done
- Dataset analysis (`REPORT.md` §0-9), D1 package, D2 odometry + EKF, D3 camera geometry (pivot, §10), D4 visual heading.
- D5 done: geo-referenced ORB map (CLAHE) + quality-weighted PnP fixes, EKF with online sideslip states, braking-aware (phase) wheel odometry (`--estimator vismap`, `./run.sh`).
  **Strict** held-out validation (map AND odometry calibration rebuilt without the scored lap): lap 2 0.598 m, lap 4 0.460 m, lap 6 0.415 m, **pooled 0.513 m, lateral 0.303 m, 95.1 % within 1 m** (EXP-0026..0028).
- TASK I (deadline-critical): no starting pose in the test file -> global initialisation + re-localisation (EXP-0036); test-domain layout without state_estimation / tf / track_bounds (validation/twist, bounds asset); run.sh Python >= 3.10 fallback. Strict auto-init 0.513 / 0.303 / 95.1 %, 0 wrong locks in 16 mid-lap starts. Devin crash test on the organisers' sample: 84 ms/frame, 74/74 1-s bins.
- TASK J/K (EXP-0037/0038): provisional two-camera lock (sigma 3 m), confirmed by 2 accepted fixes; revoked only when >= 3 rejected fixes agree on a pose >= 5 m away, then switched to it (no exclusion, no new search). Strict unchanged; all 16 mid-lap starts end correct.
- Freeze pass (Devin TASK C): clean clone of `f49ab7f` -> byte-identical map, 100 % coverage, 20.3 ms/frame mean (M4 CPU) (`submissions/README.md`).
- Devin review loop (issues #1, #2; protocol TASK -> ACK -> DONE): tasks A-H done. H: real speed blur is small (front exposure ~0.5-1 ms, EXP-0030/0031); no fallback change beat the default (EXP-0032). Open risk: anything that removes map fixes (glare) -> 2-5 m dead reckoning.
- Organiser update (`23-00-38_cut_120m_update.zip`) analysed: `map.mcap` lines, (s, n) scorer integrated (`evaluate.py --sn`).
- Submission README with results and plots (`docs/`); 26 tests green; pushed to github.com/Mulaydm10/yas-camloc (private).

## In flight
- (none)

## Blocked
- `COMPETITION.md` filled from the brief (2026-10-18); still unknown: deadline, dates, team size, roster, upload channel (Q-0002). Thesis `VISION.md` (Q-0001).
- `tests/README.md` (LOCKED) still says there is no test baseline: needs a Main Agent edit + audit row.
- Organiser questions: state_estimation fields (REPORT.md §9 Q1, default allowed-only), authoritative (s, n) reference, missing 35 s bag, test-file contents (REPORT.md §11 Q11-13).
- A GitHub reviewer will look at the repo: add them as a collaborator (private repo) once the user gives the username.

## Next intended step
1. OWNER, before 2026-10-18 23:59 CEST (COMPETITION.md): make the repo PUBLIC; submit the form (<= 6 slides PDF, <= 5 min video, trajectory plot `docs/traj_lap*_heldout.png`, team members).
2. Devin: re-run the organisers' sample on the TASK L commit (guard: if not validated by 18:30 UTC, ship with the revoke disabled, PROV_REVOKE=0).

## Latest experiment
EXP-0039: pair-free 3-frame provisional lock restored (daylight sample fix); strict 0.513 / 0.303 / 95.1 %; 16/16 starts end correct.

## Work claims
| Surface / task | Claimed by | Since (with timezone) | IDs |
|---|---|---|---|
| (none) | | | |

Last updated: 2026-10-18 17:50 CEST (by Claude Opus, SSH session)
