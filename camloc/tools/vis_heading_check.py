"""TRAINING DIAGNOSTIC: keyframe visual rotation per camera vs the GT rotation over the same interval,
replayed along the run with the same keyframe policy as run.VisualHeading (1.5 s).

    python -m camloc.tools.vis_heading_check [start_s duration_s]
"""
import json
import sys

import cv2
import numpy as np

from .. import config as C
from ..align import rot_adjust
from ..data import FrameSet, Replay, StateEvent, load_static
from ..groundtruth import load_gt6
from ..vision_rot import KeyframeRotation
from .make_masks import load_masks

CAMS = ('fl', 'fr', 'r', 'cl', 'cr', 'rl', 'rr')


def fit_scales(a):
    """Robust per-camera scale of visual vs GT rotation (|GT| > 2 deg), written to camloc/assets."""
    out = {}
    for i, c in enumerate(CAMS):
        m = (a[:, 1] == i) & (np.abs(a[:, 3]) > np.radians(2))
        x, y = a[m, 3], a[m, 2]
        for _ in range(3):
            k = np.sum(x * y) / np.sum(x * x)
            r = y - k * x
            s = 1.4826 * np.median(np.abs(r))
            keep = np.abs(r) < 3 * s
            x, y = x[keep], y[keep]
        out[c] = float(k)
        print(f'{c}: rotation scale vis/gt {k:.4f}')
    (C.ROOT / 'camloc' / 'assets' / 'vis_rot_scale.json').write_text(json.dumps(
        {'_doc': 'visual/GT rotation scale per camera (training fit, tools/vis_heading_check.py); divide by it', **out}, indent=1))


def main(start=0.0, dur=None):
    cams, _ = load_static()
    g = load_gt6()
    yu = np.unwrap(g['yaw'])
    adj = json.loads((C.ROOT / 'camloc' / 'assets' / 'camera_adjust.json').read_text())
    masks = load_masks(CAMS)
    kr = {c: KeyframeRotation(cams[c], masks[c][::2, ::2][:364, :753],
                              (cams[c].T_base_cam @ rot_adjust(*adj.get(c, (0, 0, 0))))[:3, :3]) for c in CAMS}
    rows, kf_t, v = [], None, 0.0
    for ev in Replay(state='allowed', start_s=start, duration_s=dur):
        if isinstance(ev, StateEvent):
            v = 0.3052 * max(0.5 * (ev.fields['omega_w_fl'] + ev.fields['omega_w_fr']), 0)
            continue
        gray = {c: cv2.imdecode(np.frombuffer(ev.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2) for c in CAMS if c in ev.images}
        if kf_t is None or (ev.stamp_ns - kf_t) / 1e9 > 1.5:
            for c in gray:
                kr[c].reset(gray[c])
            kf_t = ev.stamp_ns
            continue
        gt_d = np.interp(ev.stamp_ns, g['t'], yu) - np.interp(kf_t, g['t'], yu)
        for i, c in enumerate(CAMS):
            if c not in gray:
                continue
            m = kr[c].track(gray[c])
            if m is not None:
                rows.append((ev.stamp_ns, i, m[0], gt_d, m[1], v))
    a = np.array(rows)
    np.save(C.CACHE / 'vis_heading_check.npy', a)
    fit_scales(a)
    e = np.degrees(a[:, 2] - a[:, 3])
    for i, c in enumerate(CAMS):
        m = a[:, 1] == i
        line = f'{c}: n {m.sum():5d} err median {np.median(e[m]):+.3f} MAD {np.median(np.abs(e[m] - np.median(e[m]))):.3f} deg |'
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45)):
            mm = m & (a[:, 5] >= lo) & (a[:, 5] < hi)
            if mm.sum() > 20:
                line += f' v{lo}-{hi}: {np.median(e[mm]):+.3f}/{np.median(np.abs(e[mm] - np.median(e[mm]))):.3f}'
        print(line)


if __name__ == '__main__':
    args = [float(x) for x in sys.argv[1:]]
    main(*args) if args else main()
