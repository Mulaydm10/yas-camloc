> **LOCKED governing file.** Do not edit in place. See `GOVERNANCE.md`.

# experiments/ - EXP-#### schema (locked)

`experiment_log.md` is append-only. One row per experiment/spike, never edited after writing (add a follow-up row instead).
Columns: **ID | Date/time (TZ) | Hypothesis | Method | Result | Decision/next | Log dir**.
Raw output goes in `logs/EXP-####_YYYY-MM-DD_slug/` (untracked). Cite `EXP-####` in worklog, STATE, ADRs.
