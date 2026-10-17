# Experiment log (APPEND-ONLY; schema in README.md)

| ID | Date/time (TZ) | Hypothesis | Method | Result | Decision/next | Log dir |
|---|---|---|---|---|---|---|
| EXP-0001 | 2026-10-17 06:06 CEST | Dead reckoning from car state is a usable baseline | analysis/deadreckon.py, whole run, causal, first pose only | v_mps+wz 65.5 m RMSE; wheels+bicycle (allowed) 154 m; GT reset every 5 s 0.64 m | Vision must correct every frame (REPORT.md §6) | analysis/out/ |
| EXP-0002 | 2026-10-17 06:06 CEST | D1 pipeline is correct end to end | camloc.run --estimator hold (first pose held) + camloc.evaluate --official | 11944 poses, 100 % coverage, 528.5 m RMSE; harness == tools/evaluate.py (whole run and lap 4: 530.038 m) | Reference floor only; D2 = wheel/bicycle predictor in (s, n) | cache/ |
