"""TRAINING DIAGNOSTIC: map-based PnP localisation accuracy vs GT on a time window, querying the map with the
GT pose perturbed by a given error (to mimic the filter's prior).

    python -m camloc.tools.vmap_check cache/vmap_no_lap4.npz 602.8 701.7 [prior_err_m]
"""
import json
import sys

import cv2
import numpy as np

from .. import config as C
from ..align import rot_adjust
from ..data import FrameSet, Replay, load_static
from ..groundtruth import load_gt
from ..vmap import MapLocalizer, VisualMap
from .make_masks import load_masks

CAMS = ('fl', 'fr', 'r')


def main(path, t0, t1, prior=10.0, step=3):
    cams, _ = load_static()
    gt = load_gt()
    adj = json.loads((C.ROOT / 'camloc' / 'assets' / 'camera_adjust.json').read_text())
    Tbc = {c: cams[c].T_base_cam @ rot_adjust(*adj.get(c, (0, 0, 0))) for c in CAMS}
    loc = MapLocalizer(VisualMap(path), cams, Tbc, {c: m[::2, ::2][:364, :753] for c, m in load_masks(CAMS).items()})
    rng = np.random.default_rng(0)
    yu = np.unwrap(gt['yaw'])
    rows, k = [], 0
    for ev in Replay(state=None, start_s=t0 - 1, duration_s=t1 - t0 + 1):
        if not isinstance(ev, FrameSet):
            continue
        k += 1
        if k % step:
            continue
        gx, gy = np.interp(ev.stamp_ns, gt['t'], gt['x']), np.interp(ev.stamp_ns, gt['t'], gt['y'])
        gyaw = np.interp(ev.stamp_ns, gt['t'], yu)
        dx, dy = rng.normal(0, prior / np.sqrt(2), 2)
        for c in CAMS:
            if c not in ev.images:
                continue
            g = cv2.imdecode(np.frombuffer(ev.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            r = loc.localize(c, g, gx + dx, gy + dy, gyaw)
            if r is None:
                rows.append((CAMS.index(c), np.nan, np.nan, np.nan, 0, np.nan))
                continue
            h = gyaw
            ex, ey = r['x'] - gx, r['y'] - gy
            rows.append((CAMS.index(c), np.cos(h) * ex + np.sin(h) * ey, -np.sin(h) * ex + np.cos(h) * ey,
                         np.degrees(np.angle(np.exp(1j * (r['yaw'] - gyaw)))), r['n'], r['median_depth']))
    a = np.array(rows)
    for i, c in enumerate(CAMS):
        m = a[:, 0] == i
        ok = m & ~np.isnan(a[:, 1])
        pos = np.hypot(a[ok, 1], a[ok, 2])
        print(f'{c}: success {ok.sum()}/{m.sum()} | pos err median {np.median(pos):.2f} p90 {np.percentile(pos, 90):.2f} m | '
              f'lon median|.| {np.median(np.abs(a[ok, 1])):.2f} lat {np.median(np.abs(a[ok, 2])):.2f} m | '
              f'yaw median|.| {np.median(np.abs(a[ok, 3])):.2f} deg | <2 m {np.mean(pos < 2) * 100:.0f}% | inliers {np.median(a[ok, 4]):.0f}')
    np.save(C.CACHE / 'vmap_check.npy', a)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), *(float(x) for x in sys.argv[4:]))
