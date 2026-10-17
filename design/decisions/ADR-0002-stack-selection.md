# ADR-0002: Stack selection

- Status: **Accepted**
- Date/time (TZ): 2026-10-17 06:06 CEST
- Deciders: user (via Claude Code session); event chosen = Constructor Labs camera-localization challenge
- Related: Q-0003

## Context
Team picks the stack after the idea firms up and theme/rules are known (COMPETITION.md may restrict tools).

## Options considered
- Option A: Python 3.12 + numpy/scipy/OpenCV, CPU-first (chosen)
- Option B: Python + PyTorch learned pose regression (rejected for now: night-only training data, test may be daylight; GPU dependency)
- Option C: C++/ROS 2 node (rejected: slower to iterate in a 2-day build; Python already reads the MCAP directly)

## Decision criteria
- Allowed by COMPETITION.md hard rules
- Team familiarity (fastest path to a demo)
- Time to first deployable demo
- Fit with the chosen idea and the rubric
- Ease of the reset procedure in DEMO.md

## Decision
Option A. Package `camloc/` (replay, track model, evaluation harness, run script), venv `.venv` via `uv`, pins in
`requirements.txt`, tests with pytest in `tests/`. The challenge requires a single run script on a single-GPU
workstation; a CPU pipeline at ~10 Hz satisfies that and also runs on the Mac mini (M4).

## Consequences / binding rule
Whoever resolves this ADR, in the **same change**: fills CLAUDE.md "Canonical commands", adds the toolchain config and source tree, and lands a **green smoke test** (run for real). Update tests/README.md.
