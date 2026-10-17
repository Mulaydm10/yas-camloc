"""Single entry point: raw MCAP -> submission CSV (timestamp_ns,x,y,yaw), causally.

    python -m camloc.run --out sub.csv                       # whole run, default estimator
    python -m camloc.run --out sub.csv --start 600 --duration 100 --estimator hold

Only the camera images, calibration, track bounds, whitelisted car state (config.STATE_ALLOWED, plus
STATE_AMBIGUOUS with --state ambiguous) and the FIRST pose of the run are read. Per-frame processing
time (estimator + its image decoding) is measured and written next to the CSV.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import config as C
from .data import FrameSet, Replay, StateEvent, initial_pose, load_static
from .track import TrackModel


class Estimator:
    """Interface. `on_state` gets every allowed car-state sample in stamp order; `on_frames` gets each
    image set after all state samples with stamp <= its stamp and must return (x, y, yaw) or None."""

    def __init__(self, init, cameras, track):
        self.t0, self.x, self.y, self.yaw = init
        self.cameras, self.track = cameras, track

    def on_state(self, ev: StateEvent):
        pass

    def on_frames(self, fs: FrameSet):
        raise NotImplementedError


class HoldEstimator(Estimator):
    """Baseline: report the initial pose forever (pipeline smoke test, not a method)."""

    def on_frames(self, fs):
        return self.x, self.y, self.yaw


ESTIMATORS = {'hold': HoldEstimator}


def run(mcap, out, estimator='hold', state='allowed', start_s=0.0, duration_s=None, cams=C.CAMS):
    cameras, bounds = load_static(mcap, cams)
    track = TrackModel(bounds)
    init = initial_pose(mcap)
    est = ESTIMATORS[estimator](init, cameras, track)
    rows, proc = [], []
    rp = Replay(mcap, cams=cams, state=state, start_s=start_s, duration_s=duration_s)
    n_state = 0
    for ev in rp:
        if isinstance(ev, StateEvent):
            est.on_state(ev)
            n_state += 1
            continue
        t = time.perf_counter()
        pose = est.on_frames(ev)
        proc.append(time.perf_counter() - t)
        if pose is not None:
            rows.append((ev.stamp_ns, *pose))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        f.write('timestamp_ns,x,y,yaw\n')
        for t_, x, y, yaw in rows:
            f.write(f'{int(t_)},{x:.4f},{y:.4f},{yaw:.6f}\n')
    proc = np.array(proc) * 1e3
    info = {'estimator': estimator, 'state': state, 'frames': len(proc), 'poses': len(rows), 'state_samples': n_state,
            'late_drops': rp.late_drops, 'ms_per_frame_mean': float(proc.mean()) if len(proc) else None,
            'ms_per_frame_p95': float(np.percentile(proc, 95)) if len(proc) else None,
            'hardware': f'{platform.machine()} {platform.processor() or ""} {platform.platform()}'.strip()}
    out.with_suffix('.run.json').write_text(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcap', default=str(C.DEFAULT_MCAP))
    ap.add_argument('--out', required=True)
    ap.add_argument('--estimator', default='hold', choices=sorted(ESTIMATORS))
    ap.add_argument('--state', default='allowed', choices=['allowed', 'ambiguous'])
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--duration', type=float)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.mcap, a.out, a.estimator, a.state, a.start, a.duration), indent=1))


if __name__ == '__main__':
    main()
