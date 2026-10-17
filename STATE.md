# STATE.md - live snapshot (OVERWRITE every session)

If STATE and `worklog.md` disagree about what is true *now*, STATE wins; the worklog explains how we got here.
Deadline: see `COMPETITION.md` (TODO(<owner>)). Time remaining: TODO(<owner>)

## Done
- Dataset analysis (`REPORT.md` §0-9), D1 package, D2 odometry + EKF, D3 camera geometry (pivot, §10), D4 visual heading.
- D5 done: geo-referenced ORB map + PnP fixes fused in the EKF (`--estimator vismap`, `./run.sh`).
  Held-out validation (map rebuilt without the scored lap): lap 2 0.947 m, lap 4 0.654 m, lap 6 0.700 m, **pooled 0.81 m, 86 % within 1 m**; pit lane 1.05 m; 22 ms/frame on the M4 (EXP-0012..0016, `REPORT.md` §11).
- Organiser update (`23-00-38_cut_120m_update.zip`) analysed: `map.mcap` lines, (s, n) scorer integrated (`evaluate.py --sn`).
- Submission README with results and plots (`docs/`); 19 tests green; pushed to github.com/Mulaydm10/yas-camloc (private).

## In flight
- (none)

## Blocked
- Event facts unknown: fill `COMPETITION.md` (Q-0002); thesis `VISION.md` (Q-0001).
- `tests/README.md` (LOCKED) still says there is no test baseline: needs a Main Agent edit + audit row.
- Organiser questions: state_estimation fields (REPORT.md §9 Q1, default allowed-only), authoritative (s, n) reference, missing 35 s bag, test-file contents (REPORT.md §11 Q11-13).
- A GitHub reviewer will look at the repo: add them as a collaborator (private repo) once the user gives the username.

## Next intended step
1. Back straight failure case (lap 2 lateral jump at the end of the straight): stricter acceptance of a lone fix after a long gap, or more map points there.
2. Daylight robustness is unvalidated (training run is at night): ask the organisers for the test conditions.
3. Run `./run.sh` on the organisers' test file when it arrives; pass `--init` if it has no `/tf`.

## Latest experiment
EXP-0015: single-camera fix threshold 30 -> 18, held-out pooled 1.06 -> 0.81 m. EXP-0016: speed 38 -> 22 ms/frame, bit-identical.

## Work claims
| Surface / task | Claimed by | Since (with timezone) | IDs |
|---|---|---|---|
| (none) | | | |

Last updated: 2026-10-17 23:45 CEST (by Claude Opus, session national-d2)
