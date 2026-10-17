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
