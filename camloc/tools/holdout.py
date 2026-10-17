"""TRAINING-TIME: strict hold-out support. With CAMLOC_EXCLUDE_LAPS=4 the calibration tools ignore every GT /
sensor sample inside lap 4, and with CAMLOC_ASSETS=<dir> they write (and the localiser reads) a separate asset set:

    cp -r camloc/assets cache/assets_nolap4
    export CAMLOC_ASSETS=cache/assets_nolap4 CAMLOC_EXCLUDE_LAPS=4
    python -m camloc.tools.calibrate_cameras && python -m camloc.tools.vis_heading_check \
      && python -m camloc.tools.calibrate_odometry && python -m camloc.tools.build_map --exclude-laps 4 --out $CAMLOC_ASSETS/vmap.npz
    python -m camloc.run --estimator vismap --vmap $CAMLOC_ASSETS/vmap.npz --out cache/sub_strict4.csv
"""
from functools import lru_cache

import numpy as np

from .. import config as C


@lru_cache(maxsize=1)
def windows():
    """(t0, t1) ns of the excluded laps."""
    if not C.EXCLUDE_LAPS:
        return ()
    from ..data import load_static
    from ..evaluate import gt_context, lap_windows
    from ..groundtruth import load_gt
    from ..track import TrackModel
    lw = lap_windows(gt_context(load_gt(), TrackModel(load_static(cams=())[1])))
    w = tuple((t0, t1) for n, t0, t1 in lw if n in C.EXCLUDE_LAPS)
    print(f'hold-out: excluding laps {C.EXCLUDE_LAPS} ({sum((b - a) for a, b in w) / 1e9:.0f} s)')
    return w


def excluded(t_ns):
    """Boolean (array or scalar): t inside an excluded lap."""
    t = np.asarray(t_ns)
    m = np.zeros(t.shape, bool)
    for a, b in windows():
        m |= (t >= a) & (t <= b)
    return m if m.shape else bool(m)
