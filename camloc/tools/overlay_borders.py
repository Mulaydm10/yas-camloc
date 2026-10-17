"""TRAINING DIAGNOSTIC (uses the full GT 6-DoF pose): project the surveyed 3D track borders into all
cameras with the given calibration and draw them on the images.

    python -m camloc.tools.overlay_borders 615.5 636.5 668.1     # times [s from first image]
Writes cache/overlay_<t>s.png
"""
import sys

import cv2
import numpy as np

from .. import config as C
from ..data import FrameSet, Replay, _reader, _stamp, load_bounds3d, load_static
from ..geometry import T_map_base, densify3, project


def gt6(mcap, t_ns):
    """Full GT pose at t (x, y, z, roll, pitch, yaw) from state_estimation (training diagnostic only)."""
    with open(mcap, 'rb') as f:
        for _, _, _, m in _reader(f).iter_decoded_messages(topics=[C.STATE_TOPIC], start_time=t_ns - 30_000_000):
            if _stamp(m.header) >= t_ns:
                return m.x_m, m.y_m, m.z_m, m.roll_rad, m.pitch_rad, m.yaw_rad


def bounds3d(mcap):
    return load_bounds3d(mcap)


COLORS = {'track_left': (255, 80, 0), 'track_right': (0, 80, 255), 'pitlane_left': (0, 200, 0), 'pitlane_right': (0, 200, 200)}


def overlay(times, mcap=C.DEFAULT_MCAP, cams=C.CAMS, adjust=None, tag=''):
    cameras, _ = load_static(mcap, cams)
    b3 = {k: densify3(v) for k, v in bounds3d(mcap).items()}
    for ts in times:
        ev = next(e for e in Replay(mcap, cams=cams, state=None, start_s=ts, duration_s=0.5) if isinstance(e, FrameSet))
        pose = gt6(mcap, ev.stamp_ns)
        T = T_map_base(*pose)
        tiles = []
        for c in cams:
            img = ev.decode(c)
            for k, pts in b3.items():
                uv, _, depth = project(cameras[c], T, pts, None if adjust is None else adjust.get(c))
                for (u, v), d in zip(uv.astype(int), depth):
                    cv2.circle(img, (u, v), 2 if d > 20 else 3, COLORS[k], -1)
            cv2.putText(img, f'camera_{c}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            tiles.append(cv2.resize(img, (753, 364)))
        info = np.zeros_like(tiles[0])
        for i, txt in enumerate([f't = {ts:.1f} s', 'blue: track_left  red: track_right', 'green/cyan: pit lane',
                                 f'roll {np.degrees(pose[3]):.2f} pitch {np.degrees(pose[4]):.2f} deg', tag]):
            cv2.putText(info, txt, (20, 60 + 55 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(info)
        out = C.CACHE / f'overlay_{ts:g}s{tag}.png'
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)]))
        print(out)


if __name__ == '__main__':
    overlay([float(a) for a in sys.argv[1:]] or [615.5, 636.5, 668.1])
