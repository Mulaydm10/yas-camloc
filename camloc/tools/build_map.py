"""TRAINING-TIME: build the geo-referenced visual feature map (uses the GT 6-DoF pose; never at inference).

Every STEP frames (whole run incl. the pit lane, GT accuracy <= 0.5 m, excluded laps skipped) each map camera's ORB features are matched to the
same camera PAIR_GAP frames later; matches are triangulated with the GT camera poses and kept when the
reprojection error is small in both views, the depth is 3-80 m and the triangulation angle >= 0.5 deg.

    python -m camloc.tools.build_map [--exclude-laps 4] [--out camloc/assets/vmap.npz]
"""
import argparse
import json

import cv2
import numpy as np

from .. import config as C
from ..align import rot_adjust
from ..data import FrameSet, Replay, load_static
from ..geometry import T_map_base
from ..groundtruth import load_gt6
from ..vmap import detect, orb
from .make_masks import load_masks

MAP_CAMS = ('fl', 'fr', 'r')
STEP, PAIR_GAP = 5, 2


def lap_windows(g6):
    """Lap boundary times (ns) from evaluate's GT context (track s wrap on track)."""
    from ..evaluate import gt_context, lap_windows as lw
    from ..groundtruth import load_gt
    from ..track import TrackModel
    return lw(gt_context(load_gt(), TrackModel(load_static(cams=())[1])))


def cam_pose(g6, t, Tbc):
    i = np.searchsorted(g6['t'], t)
    return T_map_base(*[g6[k][i] for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]) @ Tbc, i


def triangulate(c, T0, T1, uv0, uv1):
    """uv in full-res pixels; T = camera->map. Returns (P map (N,3), keep mask)."""
    und0 = cv2.undistortPoints(uv0.reshape(-1, 1, 2).astype(np.float64), c.K, c.D).reshape(-1, 2)
    und1 = cv2.undistortPoints(uv1.reshape(-1, 1, 2).astype(np.float64), c.K, c.D).reshape(-1, 2)
    M0, M1 = np.linalg.inv(T0)[:3], np.linalg.inv(T1)[:3]
    X = cv2.triangulatePoints(M0, M1, und0.T, und1.T)
    X = (X[:3] / X[3]).T
    keep = np.ones(len(X), bool)
    for M, und in ((M0, und0), (M1, und1)):
        Pc = (M[:, :3] @ X.T + M[:, 3:4]).T
        keep &= (Pc[:, 2] > 3) & (Pc[:, 2] < 80)
        proj = Pc[:, :2] / Pc[:, 2:3]
        keep &= np.linalg.norm(proj - und, axis=1) * c.K[0, 0] < 1.5
    r0 = X - T0[:3, 3]
    r1 = X - T1[:3, 3]
    ang = np.degrees(np.arccos(np.clip(np.sum(r0 * r1, 1) / (np.linalg.norm(r0, axis=1) * np.linalg.norm(r1, axis=1)), -1, 1)))
    keep &= ang > 0.5
    return X, keep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-laps', type=int, nargs='*', default=[])
    ap.add_argument('--norm', action='store_true', help='CLAHE-normalise images (stored in the map; EXP-0018)')
    ap.add_argument('--out', default=str(C.ASSETS / 'vmap.npz'))
    a = ap.parse_args(argv)
    cams, _ = load_static()
    g6 = load_gt6()
    adj = json.loads((C.ASSETS / 'camera_adjust.json').read_text())
    Tbc = {c: cams[c].T_base_cam @ rot_adjust(*adj.get(c, (0, 0, 0))) for c in MAP_CAMS}
    masks = {c: m[::2, ::2][:364, :753].astype(np.uint8) * 255 for c, m in load_masks(MAP_CAMS).items()}
    laps = lap_windows(g6)
    use = [(t0, t1) for n, t0, t1 in laps if n not in a.exclude_laps]
    excl = [(t0, t1) for n, t0, t1 in laps if n in a.exclude_laps]
    print(f'laps used {[n for n, _, _ in laps if n not in a.exclude_laps]}, excluded {a.exclude_laps}')
    det = orb()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    buf, k = [], 0
    kf_cam, kf_xy, kf_yaw, kf_ptr, Ps, Ds = [], [], [], [0], [], []
    for ev in Replay(state=None):
        if not isinstance(ev, FrameSet):
            continue
        buf.append(ev)
        if len(buf) > PAIR_GAP + 1:
            buf.pop(0)
        k += 1
        if len(buf) < PAIR_GAP + 1 or k % STEP:
            continue
        e0, e1 = buf[0], buf[-1]
        if any(t0 <= e.stamp_ns <= t1 for t0, t1 in excl for e in (e0, e1)):   # neither view in an excluded lap
            continue
        for c in MAP_CAMS:
            if c not in e0.images or c not in e1.images:
                continue
            T0, i0 = cam_pose(g6, e0.stamps[c], Tbc[c])
            T1, i1 = cam_pose(g6, e1.stamps[c], Tbc[c])
            if g6['pos_acc'][i0] > 0.5 or np.linalg.norm(T1[:3, 3] - T0[:3, 3]) < 1.0:
                continue
            g0 = cv2.imdecode(np.frombuffer(e0.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            g1 = cv2.imdecode(np.frombuffer(e1.images[c], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            uv0, d0 = detect(det, g0, masks[c], a.norm)
            uv1, d1 = detect(det, g1, masks[c], a.norm)
            if len(uv0) < 30 or len(uv1) < 30:
                continue
            mm = bf.match(d0, d1)
            if len(mm) < 30:
                continue
            q = np.array([m.queryIdx for m in mm])
            tr = np.array([m.trainIdx for m in mm])
            X, keep = triangulate(cams[c], T0, T1, uv0[q] * 2.0, uv1[tr] * 2.0)
            if keep.sum() < 15:
                continue
            kf_cam.append(MAP_CAMS.index(c))
            kf_xy.append([g6['x'][i0], g6['y'][i0]])
            kf_yaw.append(g6['yaw'][i0])
            Ps.append(X[keep].astype(np.float32))
            Ds.append(d0[q[keep]])
            kf_ptr.append(kf_ptr[-1] + int(keep.sum()))
    np.savez_compressed(a.out, kf_cam=np.array(kf_cam, np.int8), kf_xy=np.array(kf_xy), kf_yaw=np.array(kf_yaw),
                        kf_ptr=np.array(kf_ptr), P=np.vstack(Ps), D=np.vstack(Ds), cams=np.array(MAP_CAMS),
                        excluded_laps=np.array(a.exclude_laps), norm=np.array(a.norm))
    n = np.bincount(kf_cam, minlength=len(MAP_CAMS))
    print(f'keyframes {len(kf_cam)} {dict(zip(MAP_CAMS, n.tolist()))}, points {kf_ptr[-1]} '
          f'({kf_ptr[-1] / max(len(kf_cam), 1):.0f}/keyframe) -> {a.out}')


if __name__ == '__main__':
    main()
