"""TRAINING DIAGNOSTIC: map-based PnP localisation accuracy vs GT on a time window, querying the map with the
GT pose perturbed by a given error (to mimic the filter's prior).

    python -m camloc.tools.vmap_check cache/vmap_no_lap4.npz 602.8 701.7 [prior_err_m [step]] [--aug gamma0.5]

--aug alters the QUERY images only (the map stays as built) to probe robustness to lighting / speed changes:
gamma0.5 (brighter), gamma2 (darker), contrast0.5, blur9 (9 px horizontal motion blur, half resolution),
noise10 (Gaussian, sigma 10 grey levels), day (gamma 0.4 + contrast 0.6: washed-out bright image);
spatial (TASK G): flood (glare blobs), shadow (dark band), sky (top clipped white), speedV (blur for V m/s).
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


def augment(g, aug):
    f = g.astype(np.float32)
    if aug.startswith('gamma'):
        f = 255 * (f / 255) ** float(aug[5:])
    elif aug.startswith('contrast'):
        f = 128 + float(aug[8:]) * (f - 128)
    elif aug.startswith('blur'):
        k = int(aug[4:])
        f = cv2.filter2D(f, -1, np.ones((1, k), np.float32) / k)
    elif aug.startswith('noise'):
        f = f + np.random.default_rng(int(f.sum()) % 2**32).normal(0, float(aug[5:]), f.shape)
    elif aug == 'day':
        f = 128 + 0.6 * (255 * (f / 255) ** 0.4 - 128)
    # spatial augmentations (Devin TASK G): local effects that a global CLAHE cannot undo
    elif aug == 'flood':          # floodlight glare: three bright Gaussian blobs, positions fixed per image size
        h, w = f.shape
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(h * 7 + w)
        for _ in range(3):
            cy, cx, r = rng.uniform(0.1, 0.6) * h, rng.uniform(0.1, 0.9) * w, rng.uniform(0.06, 0.12) * w
            f = f + 220 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    elif aug == 'shadow':         # cast shadow: a dark diagonal band over the lower half (55 % darker)
        h, w = f.shape
        yy, xx = np.mgrid[0:h, 0:w]
        band = (yy > 0.45 * h) & ((xx + 0.8 * yy) % (0.9 * w) < 0.45 * w)
        f = np.where(band, 0.45 * f, f)
    elif aug == 'sky':            # saturated sky / sun: the top 25 % clipped to white
        h = f.shape[0]
        f[: h // 4] = 255
    elif aug.startswith('speed'):  # motion blur scaled to speed: speedV = blur for V m/s (half-res px ~ V / 3)
        k = max(3, int(round(float(aug[5:]) / 3)))
        f = cv2.filter2D(f, -1, np.ones((1, k), np.float32) / k)
    elif aug != 'none':
        raise SystemExit(f'unknown --aug {aug}')
    return np.clip(f, 0, 255).astype(np.uint8)


def main(path, t0, t1, prior=10.0, step=3, aug='none'):
    cams, _ = load_static()
    gt = load_gt()
    adj = json.loads((C.ASSETS / 'camera_adjust.json').read_text())
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
            g = augment(cv2.imdecode(np.frombuffer(ev.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2), aug)
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
    argv = sys.argv[1:]
    aug = 'none'
    if '--aug' in argv:
        i = argv.index('--aug')
        aug = argv[i + 1]
        del argv[i:i + 2]
    main(argv[0], float(argv[2 - 1]), float(argv[2]), *(float(x) for x in argv[3:]), aug=aug)
