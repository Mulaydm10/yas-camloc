#!/usr/bin/env bash
# Demo: strict held-out lap 4, no start pose. The visual map and every calibration were rebuilt WITHOUT lap 4
# (cache/assets_strictp_4, cache/vmap_no_lap4_norm.npz; see camloc/tools/holdout.py). Starts 6 s before lap 4.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-submissions/build/lap4.csv}
CAMLOC_ASSETS=$PWD/cache/assets_strictp_4 .venv/bin/python -W ignore -m camloc.run --estimator vismap \
  --vmap cache/vmap_no_lap4_norm.npz --start 597 --duration 106 --out "$OUT" \
  | .venv/bin/python submissions/media/summary.py
