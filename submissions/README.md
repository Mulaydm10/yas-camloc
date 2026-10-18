# submissions/

Deliverables required by the challenge brief (`notes/brief/`). Event dates and deadlines: `COMPETITION.md`.

| Deliverable | File / link | Status | Last verified (TZ) |
|---|---|---|---|
| Pose CSV for the hidden test run | `./run.sh <test.mcap> submissions/test_poses.csv` (the test file is not released yet) | Pipeline ready | 2026-10-18 01:20 CEST |
| Source code + run script | this repository, `run.sh`, `requirements.txt`, `README.md` | Done | 2026-10-18 01:20 CEST |
| Validation results (scoring-script output) | `submissions/validation/lap{2,4,6}_scores.txt` | Done | 2026-10-18 01:20 CEST |
| Visualisation (trajectory vs GT) | `docs/traj_lap{2,4,6}_heldout.png` | Done | 2026-10-18 01:20 CEST |
| Presentation (≤ 6 slides, 5 min) | outline in `docs/SLIDES.md` | Outline done; slides TODO(<owner>) | 2026-10-18 01:20 CEST |
| Declarations (pre-trained models / external data / car-state fields) | `README.md` "Declarations", ADR-0003 | Done | 2026-10-18 01:20 CEST |
| Runtime with hardware | `<out>.run.json` from every run; README "Speed" | Done | 2026-10-18 01:20 CEST |

## Freeze-readiness pass (Devin TASK C), 2026-10-18 04:15 CEST

Clean `git clone` of **`f49ab7f`** (no `.venv`, no `cache/`, no `vmap.npz`), then
`./run.sh <training.mcap> out_full.csv <training.mcap>` on the full training run (allowed signals only, `vismap`):

| item | value |
|---|---|
| Setup | venv created, GT caches extracted, visual map built **byte-identical** to the development copy (4440 keyframes, 299,900 points) |
| Poses / frames | 11,944 / 11,944 (one pose per image set, stamp = image header stamp) |
| Coverage (1 s intervals with a pose) | **100 %** (1195 / 1195); the brief's ranking threshold is in `COMPETITION.md` |
| Runtime | **20.3 ms per frame mean, 30.5 ms p95**, Apple M4 (arm64, macOS), CPU only, no GPU |
| Whole job incl. setup | 6 min 50 s |
| Whole-run score | 0.693 m position RMSE, 97.8 % within 1 m. **In-sample**: the full map contains these laps. The honest numbers are the strict held-out laps in `validation/` and the README. |
