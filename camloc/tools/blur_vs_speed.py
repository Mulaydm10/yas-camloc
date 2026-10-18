"""TRAINING DIAGNOSTIC (Devin TASK H.1, EXP-0030): how much motion blur do the real images carry vs speed?

Proxy: in the outer thirds of each half-resolution image (where image motion is mostly horizontal for these
mounts), motion blur lowers the horizontal gradient energy but not the vertical one, so
    R = mean(gx^2) / mean(gy^2)
falls with blur. Calibration: frames driven slowly (< 12 m/s) are blurred with known horizontal box kernels
of k px, giving R(k). Each fast frame's R is mapped to an equivalent blur length k_eq, binned by speed, and
compared with the V/3 px model used by the EXP-0029 augmentation (blur ADDED on top of the real blur).

    python -m camloc.tools.blur_vs_speed [--step 10]
"""
import argparse
import json

import cv2
import numpy as np

from .. import config as C
from ..data import FrameSet, Replay
from ..groundtruth import load_gt
from .make_masks import load_masks

CAMS = ('fl', 'fr', 'r', 'cl', 'cr')
KS = (1, 3, 5, 7, 9, 11, 15, 19, 25)


def ratio(g, valid):
    f = g.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    w = g.shape[1]
    side = np.zeros_like(valid)
    side[:, : w // 3] = side[:, 2 * w // 3:] = True
    m = valid & side & (np.abs(gy) + np.abs(gx) > 4)       # ignore flat (dark) pixels
    if m.sum() < 500:
        return np.nan
    return float(np.mean(gx[m] ** 2) / np.mean(gy[m] ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=10)
    a = ap.parse_args(argv)
    gt = load_gt()
    ts = gt['t'] / 1e9
    v = np.hypot(np.gradient(gt['x'], ts), np.gradient(gt['y'], ts))
    masks = {c: m[::2, ::2][:364, :753].astype(bool) for c, m in load_masks(CAMS).items()}   # True = usable pixel
    rows, k = [], 0
    calib = {c: [] for c in CAMS}
    for ev in Replay(state=None):
        if not isinstance(ev, FrameSet):
            continue
        k += 1
        if k % a.step:
            continue
        sp = float(np.interp(ev.stamp_ns, gt['t'], v))
        for c in CAMS:
            if c not in ev.images:
                continue
            g = cv2.imdecode(np.frombuffer(ev.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)[:364, :753]
            valid = masks[c][: g.shape[0], : g.shape[1]]
            rows.append((CAMS.index(c), sp, ratio(g, valid)))
            if sp < 12 and len(calib[c]) < 60:
                calib[c].append([ratio(cv2.blur(g, (kk, 1)) if kk > 1 else g, valid) for kk in KS])
    R = np.array(rows)
    out = {}
    print(f'frames sampled: {k // a.step}')
    for i, c in enumerate(CAMS):
        cal = np.nanmedian(np.array(calib[c]), axis=0)          # R as a function of added blur k (slow frames)
        m = R[:, 0] == i
        sp, r = R[m, 1], R[m, 2]
        line = {}
        for lo, hi in ((0, 12), (12, 20), (20, 28), (28, 36), (36, 45)):
            mm = (sp >= lo) & (sp < hi) & np.isfinite(r)
            if mm.sum() < 10:
                continue
            rmed = float(np.median(r[mm]))
            # invert the monotone calibration curve R(k) (decreasing) -> equivalent extra blur vs slow frames
            keq = float(np.interp(-rmed, -cal, KS))
            line[f'{lo}-{hi}'] = {'n': int(mm.sum()), 'R': rmed, 'k_eq_px': keq, 'model_V/3_px': (lo + hi) / 2 / 3}
        out[c] = {'calib_R_vs_k': dict(zip(map(str, KS), map(float, cal))), 'by_speed': line}
        print(f'{c}: calib R(k=1..25) {np.round(cal, 3).tolist()}')
        for b, d in line.items():
            print(f'   v {b:>6} m/s  n {d["n"]:4d}  R {d["R"]:.3f}  equivalent extra blur vs slow frames {d["k_eq_px"]:5.1f} px'
                  f'  (V/3 model {d["model_V/3_px"]:4.1f} px)')
    (C.CACHE / 'blur_vs_speed.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
