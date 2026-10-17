#!/usr/bin/env bash
# Single entry point: raw dataset -> submission CSV (timestamp_ns,x,y,yaw), causal.
#
#   ./run.sh <test.mcap | test_dir> <out.csv> [training.mcap]
#
# 1. creates the Python 3.12 venv (.venv) from requirements.txt if missing;
# 2. builds the training-derived assets if missing (visual map; needs the TRAINING bag with its ground truth,
#    which is allowed at training time) -- all other assets are versioned in camloc/assets/;
# 3. runs the causal localiser (estimator `vismap`: wheel/steering odometry + visual heading + map-based PnP)
#    and writes the CSV plus <out>.run.json (per-frame processing time, hardware).
set -euo pipefail
cd "$(dirname "$0")"
TEST=${1:?usage: ./run.sh <test.mcap|dir> <out.csv> [training.mcap]}
OUT=${2:?usage: ./run.sh <test.mcap|dir> <out.csv> [training.mcap]}
TRAIN=${3:-data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap}
[ -d "$TEST" ] && TEST=$(ls "$TEST"/*.mcap | head -1)

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
  else python3.12 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; fi
fi
PY=.venv/bin/python
if [ ! -f camloc/assets/vmap.npz ]; then
  echo "building the visual map from the training bag ($TRAIN) ..."
  CAMLOC_MCAP="$TRAIN" $PY -W ignore -m camloc.tools.build_map --out camloc/assets/vmap.npz
fi
$PY -W ignore -m camloc.run --mcap "$TEST" --out "$OUT" --estimator vismap
