# STATE.md - live snapshot (OVERWRITE every session)

If STATE and `worklog.md` disagree about what is true *now*, STATE wins; the worklog explains how we got here.
Deadline: see `COMPETITION.md` (TODO(<owner>)). Time remaining: TODO(<owner>)

## Done
- Repo scaffolded (ADR-0001).
- Dataset analysis for the Constructor Labs camloc challenge: `REPORT.md`, scripts in `analysis/`, venv `.venv` (Python 3.12).
- D1 done: `camloc/` package (causal replay, track+pit model, eval harness, run script), ADR-0002 accepted (EXP-0002).
- D4 done: visual heading (keyframe rotation fl/fr/r) + tracked lane state. Best allowed-only `vis`: 21.5 m whole-run RMSE, yaw 1.60 deg (EXP-0009). 17 tests green.
- D3 done (pivot): camera self-calibration + map-based attitude kept; lateral-from-paint shown unobservable (EXP-0006), see REPORT.md section 10. 15 tests green.
- D2 done: calibrated wheel/steering odometry, PoseEKF, straights-only map priors, ego-body masks. Best allowed-only: `odom_bounds` 55.1 m RMSE; ambiguous `odom_est_bounds` 13.8 m (EXP-0003/0004). 

## In flight
- (none)

## Blocked
- Event facts unknown: fill `COMPETITION.md` (Q-0002).
- Thesis unknown: fill `VISION.md` (Q-0001).
- `tests/README.md` (LOCKED) still says there is no test baseline: needs a Main Agent edit + audit row.
- Rules: which state_estimation fields are allowed (REPORT.md §9 Q1). Default: allowed-only.

## Next intended step
Get the organisers to answer the rules question (REPORT.md §9, Q1: which state_estimation fields are allowed). Fill `COMPETITION.md` from `notes/brief/*.pdf`. Then D5: absolute position from height structures (landmark map of gantries/poles/barrier tops from training GT) for s and lateral; analyse the new SharePoint dataset 23-00-38_cut_120m_update.zip once downloaded (needs sign-in).

## Latest experiment
EXP-0009: visual heading, allowed-only 21.5 m whole-run RMSE (yaw 1.60 deg)

## Work claims
| Surface / task | Claimed by | Since (with timezone) | IDs |
|---|---|---|---|
| (none) | | | |

Last updated: TODO(<owner>) (by TODO(<owner>))
