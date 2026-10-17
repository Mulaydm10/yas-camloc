# CLAUDE.md - cold-start entry point

**What this is:** a fixed-deadline hackathon repo (event: TODO(<owner>)). Worked by humans and AI agents at the same time.
Event facts live ONLY in `COMPETITION.md`. The thesis lives ONLY in `VISION.md`. Both may still be `TODO(<owner>)`.
If a doc is full of `TODO(<owner>)`, say so; do not invent values.

## Three-piece pattern (do not conflate)
- `CLAUDE.md` (this file): stable pointers and rules.
- `STATE.md`: live snapshot, **overwritten** every session. Wins over the worklog on "what is true now".
- `worklog.md`: append-only, dated + timestamped history. Explains how we got here.

## Read order
`CLAUDE.md` -> `STATE.md` -> `VISION.md` -> `COMPETITION.md` -> `GOVERNANCE.md` -> `AGENTS.md` -> tail of `worklog.md` -> `experiments/experiment_log.md`

## Governance (one paragraph)
Files starting with the LOCKED banner are edited only by the Main Agent (see `GOVERNANCE.md`), and every edit is a row in its audit table.
NOT locked on purpose: `CLAUDE.md`, `STATE.md`, `AGENTS.md`, `DEMO.md` (their value is staying current).
Append-only: `worklog.md`, `experiments/experiment_log.md`, `research/*`.

## ID scheme (single source; `grep -rn '<ID>' .` gives the full trace)
`ADR-####` decisions (`design/decisions/`) | `Q-####` open questions (`research/open_questions.md`) |
`EXP-####` experiments/spikes (`experiments/experiment_log.md`, logs in `logs/`) | `DEMO-####` demo scenarios (`DEMO.md`).
Reference IDs in commits, worklog entries and STATE rows.

## Canonical commands
Stack: Python 3.12 (ADR-0002). Dataset in `data/` (git-ignored; download steps in `analysis/README.md`).
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt   # setup
.venv/bin/python -m pytest -q tests                                     # smoke tests (dataset tests skip without data)
.venv/bin/python -m camloc.run --out cache/sub.csv [--estimator odom_bounds] [--start S --duration S]   # MCAP -> submission CSV
#   estimators: hold | odom | odom_bounds | vis (best, allowed signals only) | odom_est[_bounds], vis_est (need --state ambiguous, rules Q1)
.venv/bin/python -m camloc.tools.vis_heading_check    # training-time: visual rotation vs GT -> camloc/assets/vis_rot_scale.json
.venv/bin/python -m camloc.tools.calibrate_odometry   # training-time fit (uses GT) -> camloc/assets/odometry.json
.venv/bin/python -m camloc.tools.make_masks           # ego-body masks from camloc/assets/body_polygons.json
.venv/bin/python -m camloc.evaluate cache/sub.csv [--laps 4] [--official]   # metrics + sector/speed/lane/lap breakdown
```
Rule: inference code never imports `camloc.groundtruth` / `camloc.evaluate` (enforced by `tests/test_smoke.py`).

## Hard rules
1. Concurrency, claims and escalation: follow `AGENTS.md`.
2. `DEMO.md` must stay runnable once a path exists. Fixing a broken demo outranks new features.
3. Never state a deadline, weight or rule outside `COMPETITION.md`; link instead. Times always carry a timezone.
4. Never invent the thesis, an event rule, or a test baseline. Write `TODO(<owner>)` and register a `Q-####`.
5. No secrets in git. Respect API-key/rate-limit rules in `COMPETITION.md`.
6. Check `notes/judging_alignment.md` before starting a feature: build what is scored.

## End-of-session checklist
overwrite `STATE.md` | append dated+timestamped `worklog.md` entry | append any new `EXP-####` row | log any LOCKED-file edit in `GOVERNANCE.md` audit table | release finished claims in `STATE.md`
