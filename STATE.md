# STATE.md - live snapshot (OVERWRITE every session)

If STATE and `worklog.md` disagree about what is true *now*, STATE wins; the worklog explains how we got here.
Deadline: see `COMPETITION.md` (TODO(<owner>)). Time remaining: TODO(<owner>)

## Done
- Repo scaffolded (ADR-0001).
- Dataset analysis for the Constructor Labs camloc challenge: `REPORT.md`, scripts in `analysis/`, venv `.venv` (Python 3.12).
- D1 done: `camloc/` package (causal replay, track+pit model, eval harness, run script), ADR-0002 accepted (EXP-0002).
- D2 done: calibrated wheel/steering odometry, PoseEKF, straights-only map priors, ego-body masks. Best allowed-only: `odom_bounds` 55.1 m RMSE; ambiguous `odom_est_bounds` 13.8 m (EXP-0003/0004). 12 tests green.

## In flight
- (none)

## Blocked
- Event facts unknown: fill `COMPETITION.md` (Q-0002).
- Thesis unknown: fill `VISION.md` (Q-0001).
- `tests/README.md` (LOCKED) still says there is no test baseline: needs a Main Agent edit + audit row.
- Rules: which state_estimation fields are allowed (REPORT.md §9 Q1). Default: allowed-only.

## Next intended step
Get the organisers to answer the rules question (REPORT.md §9, Q1: which state_estimation fields are allowed). Fill `COMPETITION.md` from `notes/brief/*.pdf`. Then D3: line/edge detector (white line + blue run-off) in cl/cr/rl/rr, ground-plane projection with the extrinsics, extrinsic self-calibration on training GT.

## Latest experiment
EXP-0004: map priors on straights (allowed-only 55.1 m, ambiguous 13.8 m whole-run RMSE)

## Work claims
| Surface / task | Claimed by | Since (with timezone) | IDs |
|---|---|---|---|
| (none) | | | |

Last updated: TODO(<owner>) (by TODO(<owner>))
