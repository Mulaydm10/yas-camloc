"""TRAINING DIAGNOSTIC (Devin TASK H, EXP-0031): fit the camera exposure time from real sharpness vs speed.

Sharpness metrics on half-resolution grey images (usable pixels only): mean |Sobel x|, mean |Sobel y|, Laplacian
variance. Real frames are binned by GT speed. Slow frames (8-14 m/s) are blurred with the PHYSICAL flow-field
blur (camloc.augment.flow_blur: exposure x image flow from speed, yaw rate, camera intrinsics/extrinsics and
ground-plane depth) for a 37.5 m/s pass at candidate exposures; the exposure whose metric drops match the real
30-45 m/s frames best is the estimate (one physical constant per camera). Then the same model is used to
predict the extra loss at 60 m/s.

    python -m camloc.tools.exposure_fit [--step 7]
"""
import argparse
import json

import cv2
import numpy as np

from .. import config as C
from ..augment import flow_blur
from ..data import FrameSet, Replay, load_static
from ..groundtruth import load_gt
from .make_masks import load_masks

CAMS = ('fl', 'fr', 'r')
EXPOSURES_MS = (0.5, 1, 2, 3, 5, 8)
DATUM_DH = 0.3          # camera sits ~0.3 m higher above the road than the map datum implies (EXP-0027)


def metrics(g, valid):
    f = g.astype(np.float32)
    gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))[valid]
    gy = np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3))[valid]
    lap = cv2.Laplacian(f, cv2.CV_32F)[valid]
    return np.array([gx.mean(), gy.mean(), lap.var()])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=7)
    a = ap.parse_args(argv)
    cams, _ = load_static()
    gt = load_gt()
    ts = gt['t'] / 1e9
    v = np.convolve(np.hypot(np.gradient(gt['x'], ts), np.gradient(gt['y'], ts)), np.ones(51) / 51, 'same')
    masks = {c: m[::2, ::2][:364, :753].astype(bool) for c, m in load_masks(CAMS).items()}
    real = {c: {'slow': [], 'fast': []} for c in CAMS}
    sim = {c: {e: [] for e in EXPOSURES_MS} for c in CAMS}
    sim60 = {c: {e: [] for e in EXPOSURES_MS} for c in CAMS}
    k = 0
    for ev in Replay(state=None):
        if not isinstance(ev, FrameSet):
            continue
        k += 1
        if k % a.step:
            continue
        sp = float(np.interp(ev.stamp_ns, gt['t'], v))
        if not (8 <= sp < 14 or 30 <= sp < 45):
            continue
        for c in CAMS:
            if c not in ev.images:
                continue
            g = cv2.imdecode(np.frombuffer(ev.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)[:364, :753]
            valid = masks[c][: g.shape[0], : g.shape[1]]
            if sp < 14:
                real[c]['slow'].append(metrics(g, valid))
                if len(sim[c][EXPOSURES_MS[0]]) < 40:
                    h = cams[c].T_base_cam[2, 3] + DATUM_DH
                    for e in EXPOSURES_MS:
                        # extra blur of a 37.5 m/s pass relative to the slow frame's own ~11 m/s
                        sim[c][e].append(metrics(flow_blur(g, cams[c], 37.5 - sp, 0.0, e / 1e3, h), valid))
                        sim60[c][e].append(metrics(flow_blur(g, cams[c], 60.0 - sp, 0.0, e / 1e3, h), valid))
            else:
                real[c]['fast'].append(metrics(g, valid))
    out = {}
    names = ('gx', 'gy', 'lapvar')
    for c in CAMS:
        slow, fast = np.median(real[c]['slow'], axis=0), np.median(real[c]['fast'], axis=0)
        real_ratio = fast / slow
        base = np.median(np.array(real[c]['slow'][:40]), axis=0)
        rows = {}
        for e in EXPOSURES_MS:
            rows[e] = np.median(np.array(sim[c][e]), axis=0) / base
        # fit on gy and lapvar (the speed-sensitive metrics), in log space
        cost = {e: float(np.sum(np.log(rows[e][1:] / real_ratio[1:]) ** 2)) for e in EXPOSURES_MS}
        best = min(cost, key=cost.get)
        r60 = np.median(np.array(sim60[c][best]), axis=0) / base
        out[c] = {'real_fast_over_slow': dict(zip(names, real_ratio.tolist())), 'best_exposure_ms': best,
                  'sim_37p5_over_slow': {str(e): dict(zip(names, rows[e].tolist())) for e in EXPOSURES_MS},
                  'pred_60_over_slow_at_best': dict(zip(names, r60.tolist())), 'n_slow': len(real[c]['slow']),
                  'n_fast': len(real[c]['fast'])}
        print(f'{c}: real 30-45 / 8-14 m/s: gx {real_ratio[0]:.2f} gy {real_ratio[1]:.2f} lapvar {real_ratio[2]:.2f} '
              f'(n {len(real[c]["slow"])}/{len(real[c]["fast"])})')
        for e in EXPOSURES_MS:
            print(f'   exposure {e:4.1f} ms -> sim 37.5 m/s: gx {rows[e][0]:.2f} gy {rows[e][1]:.2f} lapvar {rows[e][2]:.2f}  cost {cost[e]:.3f}')
        print(f'   ==> best exposure {best} ms; predicted at 60 m/s: gx {r60[0]:.2f} gy {r60[1]:.2f} lapvar {r60[2]:.2f}')
    (C.CACHE / 'exposure_fit.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
