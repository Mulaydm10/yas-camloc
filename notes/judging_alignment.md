# Judging alignment

Criteria come from the rubric in `COMPETITION.md` (do not restate weights here). A criterion with no row, or a row with no artifact, is a gap to close **before submission**.

| Rubric criterion (from COMPETITION.md) | Artifact in this repo that satisfies it | Status |
|---|---|---|
| Position RMSE | `./run.sh`; validation in `submissions/validation/` (held-out pooled 0.630 m, EXP-0022) | Covered |
| Lateral error RMSE (also the tie-breaker) | quality-weighted map fixes; residual sideslip bias documented (EXP-0021) | Covered; the open improvement |
| Heading error RMSE | keyframe visual rotation + map fixes (0.26° pooled) | Covered |
| Robustness: poses within 1 m | NIS gate, consensus, CLAHE map (91.0 % pooled); stress tests EXP-0017/0018 | Covered; daylight unproven |
| Runtime per frame | `<out>.run.json`; 20.6 ms mean on an Apple M4 CPU (EXP-0024) | Covered |
| Technical soundness and novelty | README "Method", REPORT.md §10-13, ADR-0003 | Covered |
| Engineering quality and reproducibility | `run.sh` fresh-clone test (byte-identical map), 20 tests, causal replay, GT isolation test | Covered |
| Presentation incl. failure cases | `docs/SLIDES.md` outline, README "Known failure cases and limits" | Slides to make (owner) |
