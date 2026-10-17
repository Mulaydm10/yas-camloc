"""TRAINING-TIME self-calibration of camera rotations (uses the GT 6-DoF pose; never at inference).

Measurement: near-border vertical residuals (camloc/measure.py). For each camera we solve for a small
optical-frame rotation (rx, ry, rz) minimising the Huber loss of the residuals over frames spread along
the laps, with numerical Jacobians and re-search after each Gauss-Newton step (IRLS, 4 iterations).
Even-indexed frames fit, odd-indexed frames validate.

    python -m camloc.tools.calibrate_cameras [--frames 160]
Writes camloc/assets/camera_adjust.json and cache/calib_report.json
(An earlier version minimised a global chamfer over all borders; it was flat and hit its bounds, EXP-0005.)
"""
import argparse
import json

import cv2
import numpy as np

from .. import config as C
from ..align import ground_roi, rot_adjust
from ..data import FrameSet, Replay, load_static
from ..geometry import T_map_base, project
from ..groundtruth import load_gt6
from ..lines import SCALE
from ..measure import near_border_points, tophat, vertical_residuals
from ..road import Road3D
from .make_masks import load_masks
from ..data import load_bounds3d as bounds3d

OUT = C.ROOT / 'camloc' / 'assets' / 'camera_adjust.json'
CAMS = ('cl', 'cr', 'rl', 'rr', 'fr', 'fl')
HUBER = 2.0          # px
PRIOR_DEG = 1.0      # Tikhonov prior on each correction angle (README: calibrated yaw/pitch good to ~1 deg)


def collect(n_frames, cameras, masks, rois, border, g6):
    t_lo, t_hi = 90.0, 1160.0
    step = (t_hi - t_lo) / n_frames
    out, nxt, t0 = [], t_lo, None
    for ev in Replay(state=None, start_s=t_lo):
        if not isinstance(ev, FrameSet):
            continue
        t0 = ev.stamp_ns if t0 is None else t0
        ts = t_lo + (ev.stamp_ns - t0) / 1e9
        if ts < nxt:
            continue
        nxt += step
        if ts > t_hi:
            break
        i = np.searchsorted(g6['t'], ev.stamp_ns)
        if g6['pos_acc'][i] > 0.5:
            continue
        T = T_map_base(*[g6[k][i] for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
        fr = {}
        for c in CAMS:
            img = cv2.imdecode(np.frombuffer(ev.images[c], np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
            fr[c] = (tophat(img), rois[c] & masks[c], near_border_points(T, border, c))
        out.append((T, fr))
    return out


def residuals(cam, frames, c, a):
    adj = rot_adjust(*a)
    r = [vertical_residuals(cam, f[c][0], f[c][1], T, f[c][2], adj) for T, f in frames]
    return np.vstack([x for x in r if len(x)]) if any(len(x) for x in r) else np.zeros((0, 4))


def stats(cam, frames, c, a):
    r = residuals(cam, frames, c, a)
    per = []
    adj = rot_adjust(*a)
    for T, f in frames:
        x = vertical_residuals(cam, f[c][0], f[c][1], T, f[c][2], adj)
        if len(x) >= 5:
            per.append(np.median(x[:, 2]))
    return {'n_pts': len(r), 'median_dv': float(np.median(r[:, 2])),
            'mad_dv': float(np.median(np.abs(r[:, 2] - np.median(r[:, 2])))),
            'frame_median_std': float(np.std(per)), 'frame_median_rms': float(np.sqrt(np.mean(np.square(per))))}


def fit(cam, frames, c, iters=4, eps=np.radians(0.05)):
    a = np.zeros(3)
    for _ in range(iters):
        J, r, w = [], [], []
        adj0 = rot_adjust(*a)
        for T, f in frames:
            res = vertical_residuals(cam, f[c][0], f[c][1], T, f[c][2], adj0)
            if len(res) < 5:
                continue
            # numerical d(v_pred)/d(angle) at the accepted points: reproject the same map points
            pts = f[c][2]
            uv0, idx0, _ = project(cam, T, pts, adj0, max_range=80.0)
            key0 = {(int(round(u * SCALE)), round(v * SCALE, 3)): k for k, (u, v) in enumerate(uv0)}
            sel = [key0.get((int(u), round(v, 3))) for u, v in res[:, :2]]
            m = np.array([s is not None for s in sel])
            if m.sum() < 5:
                continue
            sel = np.array([s for s in sel if s is not None])
            P = pts[idx0[sel]]
            cols = []
            for k in range(3):
                da = a.copy()
                da[k] += eps
                uv1, idx1, _ = project(cam, T, P, rot_adjust(*da), max_range=80.0)
                if len(idx1) != len(P):
                    break
                cols.append((uv1[:, 1] * SCALE - uv0[sel, 1] * SCALE) / eps)
            if len(cols) != 3:
                continue
            J.append(np.c_[cols].T)
            r.append(res[m, 2])
            w.append(res[m, 3])
        J, r = np.vstack(J), np.concatenate(r)
        hub = np.where(np.abs(r) <= HUBER, 1.0, HUBER / np.maximum(np.abs(r), 1e-9))
        W = hub
        # measured v = v_pred + dv; to move v_pred onto measurement: J @ da = dv
        # Gauss-Newton step with a zero-mean prior on the total correction a (sigma PRIOR_DEG, in units of
        # one Huber-weighted pixel residual per point), so weakly observed angles stay near the nominal value
        lam = np.sqrt(len(r)) / np.radians(PRIOR_DEG) * 0.5
        A = np.vstack([J * np.sqrt(W)[:, None], lam * np.eye(3)])
        b = np.r_[r * np.sqrt(W), -lam * a]
        da = np.linalg.lstsq(A, b, rcond=None)[0]
        a = a + np.clip(da, -np.radians(1.5), np.radians(1.5))
    return a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=160)
    args = ap.parse_args(argv)
    cameras, _ = load_static()
    masks = {c: m[::2, ::2][:364, :753] for c, m in load_masks().items()}
    rois = {c: ground_roi(cameras[c], 2, 60) for c in CAMS}
    road = Road3D(bounds3d(C.DEFAULT_MCAP))
    frames = collect(args.frames, cameras, masks, rois, np.vstack([road.L, road.R]), load_gt6())
    fitf, valf = frames[0::2], frames[1::2]
    print(f'{len(frames)} frames ({len(fitf)} fit / {len(valf)} validate)')
    res, report = {}, {}
    for c in CAMS:
        a = fit(cameras[c], fitf, c)
        b0, b1 = stats(cameras[c], valf, c, (0, 0, 0)), stats(cameras[c], valf, c, a)
        if b1['frame_median_rms'] >= b0['frame_median_rms']:      # no held-out improvement: keep nominal
            a = np.zeros(3)
            b1 = b0
        report[c] = {'adjust_deg': np.degrees(a).tolist(), 'val_before': b0, 'val_after': b1}
        res[c] = a.tolist()
        print(f"{c}: adjust deg {np.round(np.degrees(a), 2)} | val median dv {b0['median_dv']:+.2f} -> {b1['median_dv']:+.2f}, "
              f"frame-median rms {b0['frame_median_rms']:.2f} -> {b1['frame_median_rms']:.2f} px, MAD {b0['mad_dv']:.2f} -> {b1['mad_dv']:.2f}")
    OUT.write_text(json.dumps({'_doc': 'optical-frame rotation corrections (rad, Rodrigues rx ry rz) fitted on training GT '
                                       'by tools/calibrate_cameras.py (near-border vertical residuals)', **res}, indent=1))
    (C.CACHE / 'calib_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
