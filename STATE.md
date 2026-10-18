# STATE.md - live snapshot (OVERWRITE every session)

If STATE and `worklog.md` disagree about what is true *now*, STATE wins; the worklog explains how we got here.
Deadline: see `COMPETITION.md` (TODO(<owner>)). Time remaining: TODO(<owner>)

## Done
- Dataset analysis (`REPORT.md` §0-9), D1 package, D2 odometry + EKF, D3 camera geometry (pivot, §10), D4 visual heading.
- D5 done: geo-referenced ORB map (CLAHE) + quality-weighted PnP fixes, EKF with online sideslip states, braking-aware (phase) wheel odometry (`--estimator vismap`, `./run.sh`).
  **Strict** held-out validation (map AND odometry calibration rebuilt without the scored lap): lap 2 0.598 m, lap 4 0.460 m, lap 6 0.415 m, **pooled 0.513 m, lateral 0.303 m, 95.1 % within 1 m** (EXP-0026..0028).
- Freeze pass (Devin TASK C): clean clone of `f49ab7f` -> byte-identical map, 100 % coverage, 20.3 ms/frame mean (M4 CPU) (`submissions/README.md`).
- Devin review loop (issues #1, #2; protocol TASK -> ACK -> DONE): tasks A-G done. Largest open risk: much higher test speed defeats ORB matching (EXP-0029).
- Organiser update (`23-00-38_cut_120m_update.zip`) analysed: `map.mcap` lines, (s, n) scorer integrated (`evaluate.py --sn`).
- Submission README with results and plots (`docs/`); 19 tests green; pushed to github.com/Mulaydm10/yas-camloc (private).

## In flight
- (none)

## Blocked
- `COMPETITION.md` filled from the brief (2026-10-18); still unknown: deadline, dates, team size, roster, upload channel (Q-0002). Thesis `VISION.md` (Q-0001).
- `tests/README.md` (LOCKED) still says there is no test baseline: needs a Main Agent edit + audit row.
- Organiser questions: state_estimation fields (REPORT.md §9 Q1, default allowed-only), authoritative (s, n) reference, missing 35 s bag, test-file contents (REPORT.md §11 Q11-13).
- A GitHub reviewer will look at the repo: add them as a collaborator (private repo) once the user gives the username.

## Next intended step
1. Wait for Devin / owner: proposed high-speed mitigation (coarser pyramid query above ~35 m/s, EXP-0029) needs a TASK.
2. Owner: slides from `docs/SLIDES.md`; exact submission time/timezone into `COMPETITION.md` (last work day 2026-10-18).
3. When the hidden test file arrives: `./run.sh <test.mcap> submissions/test_poses.csv` (`-- --init x,y,yaw` if no /tf).

## Latest experiment
EXP-0028: phase axle + online sideslip, strict pooled 0.513 m / lateral 0.303 m / 95.1 %. EXP-0029: speed blur is the largest open risk.

## Work claims
| Surface / task | Claimed by | Since (with timezone) | IDs |
|---|---|---|---|
| (none) | | | |

Last updated: 2026-10-18 04:40 CEST (by Claude Opus, overnight session in screen `camloc`)
