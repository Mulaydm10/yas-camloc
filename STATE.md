# STATE.md - live snapshot (OVERWRITE every session)

If STATE and `worklog.md` disagree about what is true *now*, STATE wins; the worklog explains how we got here.
Deadline: see `COMPETITION.md` (TODO(<owner>)). Time remaining: TODO(<owner>)

## Done
- Dataset analysis (`REPORT.md` §0-9), D1 package, D2 odometry + EKF, D3 camera geometry (pivot, §10), D4 visual heading.
- D5 done: geo-referenced ORB map (CLAHE) + PnP fixes, quality-weighted, fused in the EKF (`--estimator vismap`, `./run.sh`).
  Held-out validation (map rebuilt without the scored lap): lap 2 0.674 m, lap 4 0.613 m, lap 6 0.577 m, **pooled 0.630 m, lateral 0.477 m, 91.0 % within 1 m** (EXP-0020..0022).
- Devin review (GitHub issue #1), both rounds: stress test, CLAHE robustness, strict calibration hold-out, lateral-regression ablation, sideslip analysis, fused-state variant + ADR-0003, fresh-clone run.sh test, PDF deliverables (`REPORT.md` §12-13).
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
1. Owner: make the slides from `docs/SLIDES.md`; add the deadline / dates to `COMPETITION.md`. (ADR-0003 confirmed: submit allowed-only `vismap`.)
2. Lateral: residual bias about -0.2 m (sideslip, EXP-0021); a per-lane or speed-dependent sideslip model that holds on lap 2 too.
3. Run `./run.sh <test.mcap> submissions/test_poses.csv` when the hidden test file arrives (`-- --init x,y,yaw` if it has no /tf).

## Latest experiment
EXP-0022: fix-quality weighting, held-out pooled 0.630 m (91.0 % < 1 m). EXP-0023: fused car state, lateral 0.430 m but weaker in blackouts -> ADR-0003 keeps allowed-only.

## Work claims
| Surface / task | Claimed by | Since (with timezone) | IDs |
|---|---|---|---|
| (none) | | | |

Last updated: 2026-10-18 01:30 CEST (by Claude Opus, session national-d2)
