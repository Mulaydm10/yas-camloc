"""TRAINING DIAGNOSTIC (Devin TASK E.1, EXP-0026): is the camera really ~0.29 m above the border paint?

The surveyed border z and the GT base_link z may use different vertical datums, so the camera height above
the painted lines is fitted from the images instead. On straights (GT pose), the border points are shifted
down by dh (= the camera sits dh higher above the paint than the map says). For each dh, each camera's
rotation correction is refitted on the even frames (tools/calibrate_cameras.fit), and the near-border
vertical residuals are scored on the odd frames. A rotation cannot absorb a height error, because the
induced image shift falls off with distance (dv ~ f * dh / d) while a rotation shifts every point equally.
At the best dh the near-border lateral sensitivity (px per m, half resolution) is recomputed.

    python -m camloc.tools.fit_camera_height [--frames 320]
    python -m camloc.tools.fit_camera_height --scan     # then: per-frame lateral estimate scatter (E.1 verdict)
"""
import argparse
import json

import numpy as np

from .. import config as C
from ..align import ground_roi, rot_adjust
from ..data import load_bounds3d, load_static
from ..geometry import project
from ..groundtruth import load_gt6
from ..measure import vertical_residuals
from ..road import Road3D
from ..track import TrackModel
from . import calibrate_cameras as CC
from .calibrate_cameras import collect, fit, residuals
from .make_masks import load_masks

CAMS = ('cl', 'cr', 'fl', 'fr')
# --search-px N widens the residual search window (measure.SEARCH_PX, half-res px); results go to a suffixed json
import sys as _sys
from .. import measure as _measure
if '--search-px' in _sys.argv:
    _measure.SEARCH_PX = int(_sys.argv[_sys.argv.index('--search-px') + 1])
OUT_NAME = f'camera_height_fit_s{_measure.SEARCH_PX}.json' if '--search-px' in _sys.argv else 'camera_height_fit.json'
DH = np.round(np.arange(-0.3, 0.71, 0.1), 2)


def shifted(frames, c, dh):
    out = []
    for T, f in frames:
        th, valid, pts = f[c]
        out.append((T, {c: (th, valid, pts - [0.0, 0.0, dh])}))
    return out


def cost(cam, frames, c, a):
    """Robust score: RMS of the per-frame median residual and the pooled MAD (half-res px)."""
    per, allr = [], []
    for T, f in frames:
        r = residuals(cam, [(T, f)], c, a)
        if len(r) >= 5:
            per.append(np.median(r[:, 2]))
            allr.append(r[:, 2])
    if not per:
        return np.inf, np.inf, 0
    a_ = np.concatenate(allr)
    return float(np.sqrt(np.mean(np.square(per)))), float(np.median(np.abs(a_ - np.median(a_)))), len(a_)


def sensitivity(cam, frames, c, a, dh):
    """Median |dv| per metre of lateral offset (half-res px/m) for the near-border points."""
    A = rot_adjust(*a)
    s = []
    for T, f in frames:
        pts = f[c][2] - [0.0, 0.0, dh]
        if len(pts) < 5:
            continue
        T2 = T.copy()
        T2[:3, 3] += 0.5 * T[:3, 1]
        uv0, i0, _ = project(cam, T, pts, A, 80)
        uv1, i1, _ = project(cam, T2, pts, A, 80)
        common = np.intersect1d(i0, i1)
        if len(common) < 5:
            continue
        v0 = uv0[np.searchsorted(i0, common), 1]
        v1 = uv1[np.searchsorted(i1, common), 1]
        s.append(np.median(np.abs(v1 - v0)) * 0.5 / 0.5)
    return float(np.median(s)) if s else float('nan')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=320)
    ap.add_argument('--search-px', type=int)
    a = ap.parse_args(argv)
    cameras, bounds = load_static()
    tm = TrackModel(bounds)
    masks = {c: m[::2, ::2][:364, :753] for c, m in load_masks().items()}
    rois = {c: ground_roi(cameras[c], 2, 60) for c in CC.CAMS}       # collect() decodes all calibration cameras
    road = Road3D(load_bounds3d(C.DEFAULT_MCAP))
    frames = collect(a.frames, cameras, masks, rois, np.vstack([road.L, road.R]), load_gt6())
    keep = []
    for T, f in frames:
        x, y = T[0, 3], T[1, 3]
        yaw = np.arctan2(T[1, 0], T[0, 0])
        lane, s = tm.locate(x, y, yaw)[:2]
        if str(lane[0]) == 'track' and tm.lanes['track'].straight_at(s)[0]:
            keep.append((T, f))
    fitf, valf = keep[0::2], keep[1::2]
    print(f'{len(frames)} frames, {len(keep)} on straights ({len(fitf)} fit / {len(valf)} validate)')
    report = {}
    for c in CAMS:
        rows = []
        for dh in DH:
            fs, vs = shifted(fitf, c, dh), shifted(valf, c, dh)
            ang = fit(cameras[c], fs, c)
            rms, mad, n = cost(cameras[c], vs, c, ang)
            rows.append((float(dh), rms, mad, n, np.degrees(ang).tolist()))
            print(f'{c} dh {dh:+.1f} m: val frame-median rms {rms:.2f} px, MAD {mad:.2f} px, n {n}, rot deg {np.round(np.degrees(ang), 2)}')
        best = min(rows, key=lambda r: r[1])
        # bootstrap the argmin over validation frames for a rough confidence interval
        rng = np.random.default_rng(0)
        picks = []
        angs = {r[0]: np.radians(r[4]) for r in rows}
        per = {r[0]: [] for r in rows}
        for dh in DH:
            for T, f in shifted(valf, c, dh):
                r = residuals(cameras[c], [(T, f)], c, angs[float(dh)])
                per[float(dh)].append(np.median(r[:, 2]) if len(r) >= 5 else np.nan)
        M = np.array([per[float(dh)] for dh in DH])          # (n_dh, n_frames)
        ok = ~np.isnan(M).any(axis=0)
        M = M[:, ok]
        for _ in range(200):
            j = rng.integers(0, M.shape[1], M.shape[1])
            picks.append(DH[np.argmin(np.sqrt(np.mean(M[:, j] ** 2, axis=1)))])
        lo, hi = np.percentile(picks, [5, 95])
        sens = sensitivity(cameras[c], valf, c, np.radians(best[4]), best[0])
        sens0 = sensitivity(cameras[c], valf, c, np.radians(rows[list(DH).index(0.0)][4]), 0.0)
        noise = best[2] * 1.4826
        report[c] = {'best_dh_m': best[0], 'dh_90ci_m': [float(lo), float(hi)], 'val_rms_px': best[1],
                     'val_mad_px': best[2], 'sens_px_per_m_at_best': sens, 'sens_px_per_m_at_dh0': sens0,
                     'lateral_equiv_noise_m_per_point': noise / sens if sens > 0 else None, 'rows': rows}
        print(f'==> {c}: best dh {best[0]:+.1f} m (90 % CI {lo:+.1f}..{hi:+.1f}), sensitivity {sens0:.2f} -> {sens:.2f} px/m, '
              f'point noise ~{noise:.1f} px -> {noise / sens if sens > 0 else float("nan"):.1f} m lateral-equivalent per point')
    (C.CACHE / OUT_NAME).write_text(json.dumps(report, indent=1))


if __name__ == '__main__' and '--scan' not in __import__('sys').argv:
    main()


def lateral_scan(cam, frames, c, ang, dh, deltas=np.round(np.arange(-1.5, 1.51, 0.1), 2)):
    """Direct test of the measurement: per frame, shift the GT pose sideways by delta and pick the delta whose
    projected border best matches the image (median |dv| over points found at every delta). The spread of the
    picked delta around 0 (truth) is the per-frame lateral measurement noise of this camera."""
    A = rot_adjust(*ang)
    est = []
    for T, f in frames:
        th, valid, pts = f[c]
        pts = pts - [0.0, 0.0, dh]
        costs = []
        for d in deltas:
            T2 = T.copy()
            T2[:3, 3] += d * T[:3, 1]
            r = vertical_residuals(cam, th, valid, T2, pts, A)
            costs.append(np.median(np.abs(r[:, 2])) if len(r) >= 8 else np.inf)
        costs = np.array(costs)
        if np.isfinite(costs).sum() >= len(deltas) // 2:
            est.append(deltas[int(np.argmin(costs))])
    est = np.array(est)
    return est


def scan_main():
    rep = json.loads((C.CACHE / OUT_NAME).read_text())
    cameras, bounds = load_static()
    tm = TrackModel(bounds)
    masks = {c: m[::2, ::2][:364, :753] for c, m in load_masks().items()}
    rois = {c: ground_roi(cameras[c], 2, 60) for c in CC.CAMS}
    road = Road3D(load_bounds3d(C.DEFAULT_MCAP))
    frames = collect(320, cameras, masks, rois, np.vstack([road.L, road.R]), load_gt6())
    keep = [(T, f) for T, f in frames
            if str(tm.locate(T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0]))[0][0]) == 'track'
            and tm.lanes['track'].straight_at(tm.locate(T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0]))[1])[0]]
    valf = keep[1::2]
    for c in CAMS:
        for dh in (0.0, rep[c]['best_dh_m']):
            row = [r for r in rep[c]['rows'] if abs(r[0] - dh) < 1e-6][0]
            e = lateral_scan(cameras[c], valf, c, np.radians(row[4]), dh)
            mad = 1.4826 * np.median(np.abs(e - np.median(e))) if len(e) else np.nan
            print(f'{c} dh {dh:+.1f}: frames {len(e)}, lateral estimate median {np.median(e):+.2f} m, '
                  f'robust std {mad:.2f} m, RMS {np.sqrt(np.mean(e ** 2)):.2f} m, |err|<0.3 m: {np.mean(np.abs(e) < 0.3) * 100:.0f} %')
            rep[c][f'scan_dh{dh:+.1f}'] = {'median_m': float(np.median(e)), 'robust_std_m': float(mad),
                                           'rms_m': float(np.sqrt(np.mean(e ** 2))), 'n': int(len(e))}
    (C.CACHE / OUT_NAME).write_text(json.dumps(rep, indent=1))


if __name__ == '__main__' and __import__('sys').argv[1:2] == ['--scan']:
    scan_main()
