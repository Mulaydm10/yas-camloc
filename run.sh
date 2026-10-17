#!/usr/bin/env bash
# Single entry point: raw dataset -> submission CSV (timestamp_ns,x,y,yaw), causal.
#
#   ./run.sh <test.mcap | test_dir> <out.csv> [training.mcap] [-- extra camloc.run options]
#   e.g. ./run.sh test.mcap out.csv -- --init 12.3,-4.5,1.57      (first pose, if the test file has no /tf)
#        ./run.sh test.mcap out.csv -- --state ambiguous --estimator vismap_est   (fused car state, ADR-0003)
#
# 1. creates the Python 3.12 venv (.venv) from requirements.txt if missing;
# 2. builds the training-derived assets if missing (visual map; needs the TRAINING bag with its ground truth,
#    which is allowed at training time) -- all other assets are versioned in camloc/assets/;
# 3. runs the causal localiser (estimator `vismap`: wheel/steering odometry + visual heading + map-based PnP)
#    and writes the CSV plus <out>.run.json (per-frame processing time, hardware).
set -euo pipefail
abspath() { case $1 in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }   # paths are relative to the caller
USAGE="usage: ./run.sh <test.mcap|dir> <out.csv> [training.mcap] [-- camloc.run options]"
TEST=$(abspath "${1:?$USAGE}"); OUT=$(abspath "${2:?$USAGE}"); shift 2
TRAIN=""
if [ $# -gt 0 ] && [ "$1" != "--" ]; then TRAIN=$(abspath "$1"); shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
cd "$(dirname "$0")"
TRAIN=${TRAIN:-$PWD/data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap}
[ -d "$TEST" ] && TEST=$(ls "$TEST"/*.mcap | head -1)

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
  else python3.12 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; fi
fi
PY=.venv/bin/python
if [ ! -f camloc/assets/vmap.npz ]; then
  echo "building the visual map from the training bag ($TRAIN) ..."
  CAMLOC_MCAP="$TRAIN" $PY -W ignore -m camloc.tools.build_map --norm --out camloc/assets/vmap.npz
fi
$PY -W ignore -m camloc.run --mcap "$TEST" --out "$OUT" --estimator vismap "$@"
