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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config as C
from .data import FrameSet, Replay, StateEvent, initial_pose, load_static
from .filter import PoseEKF
from .odometry import Odometry
from .track import TrackModel
from .vision_rot import KeyframeRotation


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


class OdomEstimator(Estimator):
    """EKF driven by car-state odometry; optional track-bounds constraint (D2). No cameras yet."""
    source, bounds = 'wheels', False

    def __init__(self, init, cameras, track):
        super().__init__(init, cameras, track)
        self.odo = Odometry(source=self.source)
        self.ekf = PoseEKF(self.x, self.y, self.yaw, self.odo.p['q'])
        self.t = self.t0

    def _advance(self, t_ns):
        if self.t == 0:                      # --init without a stamp: start the clock at the first event
            self.t = t_ns
            return
        if t_ns > self.t:
            self.ekf.predict((t_ns - self.t) / 1e9, self.odo.v, self.odo.wz)
            self.t = t_ns

    def on_state(self, ev):
        self._advance(ev.stamp_ns)          # zero-order hold of the previous sample's (v, wz)
        self.odo.update(ev.stamp_ns, ev.fields)

    def on_frames(self, fs):
        self._advance(fs.stamp_ns)
        if self.bounds:
            self.ekf.constrain_to_lane(self.track)
        return self.ekf.pose


class OdomBounds(OdomEstimator):
    bounds = True


class OdomEst(OdomEstimator):
    source = 'estimator'


class OdomEstBounds(OdomEstimator):
    source, bounds = 'estimator', True


class VisualHeading(OdomBounds):
    """D4: odometry + straights-only map priors + visual heading increments against keyframes (fl, fr, r)."""
    VIS_CAMS = ('fl', 'fr', 'r')
    KF_MAX_AGE_S = 1.5
    KF_MAX_ROT = np.radians(10)          # renew the keyframe sooner in corners (large rotations track worse)
    AGREE = np.radians(0.5)              # cameras must agree within this (plus 2 % of the rotation)
    SIGMA_CAM = np.radians(0.25)        # per-camera keyframe rotation noise (EXP-0007 MAD 0.1-0.12 deg, inflated)
    GATE = np.radians(2.0)              # vs the filter's predicted rotation since the keyframe (plus 3 sigma)

    def __init__(self, init, cameras, track):
        super().__init__(init, cameras, track)
        import cv2
        from .align import rot_adjust
        from .tools.make_masks import load_masks
        self.cv2 = cv2
        adj = json.loads((C.ASSETS / 'camera_adjust.json').read_text())
        masks = load_masks(self.VIS_CAMS)
        self.kr = {}
        for c in self.VIS_CAMS:
            Rbc = (cameras[c].T_base_cam @ rot_adjust(*adj.get(c, (0, 0, 0))))[:3, :3]
            self.kr[c] = KeyframeRotation(cameras[c], masks[c][::2, ::2][:364, :753], Rbc)
        scales = json.loads((C.ASSETS / 'vis_rot_scale.json').read_text())
        self.scale = {c: scales.get(c, 1.0) for c in self.VIS_CAMS}
        self.kf_t = None
        # per-camera OpenCV work runs in threads (OpenCV releases the GIL); results do not depend on threading
        # because each job seeds OpenCV's (thread-local) RANSAC generator itself
        self.pool = ThreadPoolExecutor(max_workers=6)
        self._gray_for, self._gray_cache = None, {}
        self.lane = str(track.locate(self.x, self.y, self.yaw)[0][0])
        self.stats = {'updates': 0, 'rejected': 0, 'keyframes': 0, 'lane_switches': 0}

    DECODE_CAMS = VIS_CAMS

    def _gray(self, fs):
        """Half-resolution grayscale images of DECODE_CAMS, decoded once per frame set, in parallel."""
        if self._gray_for is not fs:
            cams = [c for c in self.DECODE_CAMS if c in fs.images]
            dec = lambda c: self.cv2.imdecode(np.frombuffer(fs.images[c], np.uint8), self.cv2.IMREAD_REDUCED_GRAYSCALE_2)
            self._gray_cache, self._gray_for = dict(zip(cams, self.pool.map(dec, cams))), fs
        return self._gray_cache

    def _seeded(self, fn, *a, **kw):
        self.cv2.setRNGSeed(1)
        return fn(*a, **kw)

    def on_frames(self, fs):
        self._advance(fs.stamp_ns)
        gray = {c: g for c, g in self._gray(fs).items() if c in self.VIS_CAMS}
        if self.kf_t is None:
            self._new_keyframe(fs.stamp_ns, gray)
        else:
            tracked = self.pool.map(lambda c: self._seeded(self.kr[c].track, gray[c]), list(gray))
            meas = [m[0] / self.scale[c] for c, m in zip(list(gray), tracked) if m is not None]
            pred = float(np.angle(np.exp(1j * (self.ekf.X[2] - self.ekf.X[3]))))
            sd = float(np.sqrt(self.ekf.P[2, 2] + self.ekf.P[3, 3] - 2 * self.ekf.P[2, 3]))
            good = [d for d in meas if abs(d - pred) < self.GATE + 3 * sd]
            if len(good) >= 2:                       # consensus: keep those agreeing with the median
                med = float(np.median(good))
                good = [d for d in good if abs(d - med) < self.AGREE + 0.02 * abs(med)]
                good = good if len(good) >= 2 else []
            elif good and abs(good[0] - pred) > np.radians(1.0):
                good = []                            # a lone camera must also agree with the odometry
            self.stats['rejected'] += len(meas) - len(good)
            if good:
                self.ekf.update_relative_yaw(float(np.mean(good)), self.SIGMA_CAM / np.sqrt(len(good)))
                self.stats['updates'] += 1
            alive = sum(self.kr[c].alive for c in gray)
            rot = abs(float(np.angle(np.exp(1j * (self.ekf.X[2] - self.ekf.X[3])))))
            if (fs.stamp_ns - self.kf_t) / 1e9 > self.KF_MAX_AGE_S or alive < 2 or rot > self.KF_MAX_ROT:
                self._new_keyframe(fs.stamp_ns, gray)
        if self.bounds:
            self._update_lane()
            if self.priors_active(fs.stamp_ns):
                self.ekf.constrain_to_lane(self.track, lane_name=self.lane)
        return self.ekf.pose

    def priors_active(self, t_ns):
        return True

    PIT_SPEED = 12.0        # m/s; the training run drives the pit lane at a limiter-like 8 m/s, the track at 20-40
    PIT_HOLD_S = 2.0

    def _update_lane(self):
        """Lane membership is a tracked state: track -> pit only near the pit entry AND after >= 2 s below the
        pit speed; pit -> track near the pit exit or after >= 2 s above it. A drifting estimate therefore
        cannot hop onto the parallel lane (EXP-0009)."""
        t = self.t
        slow = self.odo.v < self.PIT_SPEED
        if not hasattr(self, '_slow_since'):
            self._slow_since, self._fast_since = (t if slow else None), (None if slow else t)
        if slow:
            self._slow_since = self._slow_since or t
            self._fast_since = None
        else:
            self._fast_since = self._fast_since or t
            self._slow_since = None
        held_slow = self._slow_since is not None and (t - self._slow_since) / 1e9 > self.PIT_HOLD_S
        held_fast = self._fast_since is not None and (t - self._fast_since) / 1e9 > self.PIT_HOLD_S
        x, y, yaw = self.ekf.X[:3]
        tr, pit = self.track.track, self.track.pit
        if self.lane == 'track':
            s, _ = tr.project(x, y, yaw)[:2]
            near_entry = abs((s[0] - self.track.pit_entry_s_track + tr.length / 2) % tr.length - tr.length / 2) < 120
            if near_entry and held_slow:
                sp, np_ = pit.project(x, y, yaw)[:2]
                st, nt = tr.project(x, y, yaw)[:2]
                if sp[0] < 200 and pit.inside(sp, np_)[0] and not tr.inside(st, nt)[0]:
                    self.lane = 'pit'
                    self.stats['lane_switches'] += 1
        else:
            sp, np_ = pit.project(x, y, yaw)[:2]
            if held_fast:
                self.lane = 'track'
                self.stats['lane_switches'] += 1
            elif sp[0] > pit.length - 150:
                st, nt = tr.project(x, y, yaw)[:2]
                if tr.inside(st, nt)[0] and not pit.inside(sp, np_)[0]:
                    self.lane = 'track'
                    self.stats['lane_switches'] += 1

    def _new_keyframe(self, t, gray):
        list(self.pool.map(lambda c: self.kr[c].reset(gray[c]), list(gray)))
        self.ekf.clone_heading()
        self.kf_t = t
        self.stats['keyframes'] += 1


class VisualMapLoc(VisualHeading):
    """D5: VisualHeading + absolute pose fixes from the geo-referenced feature map (PnP in fl / fr / r)."""
    MAP_CAMS = ('fr', 'r', 'fl')
    VMAP = C.ASSETS / 'vmap.npz'
    SIGMA_XY, SIGMA_YAW = 0.6, np.radians(0.4)       # held-out lap 4: median 0.3-0.66 m, 0.17-0.29 deg
    MIN_INL_SINGLE = 18      # a lone camera fix needs this many PnP inliers (30 -> 18: held-out laps pooled 1.06 -> 0.81 m, EXP-0015)
    AGREE_M = 2.0

    def __init__(self, init, cameras, track):
        super().__init__(init, cameras, track)
        from .align import rot_adjust
        from .tools.make_masks import load_masks
        from .vmap import MapLocalizer, VisualMap
        adj = json.loads((C.ASSETS / 'camera_adjust.json').read_text())
        Tbc = {c: cameras[c].T_base_cam @ rot_adjust(*adj.get(c, (0, 0, 0))) for c in self.MAP_CAMS}
        masks = {c: m[::2, ::2][:364, :753] for c, m in load_masks(self.MAP_CAMS).items()}
        self.loc = MapLocalizer(VisualMap(self.VMAP), cameras, Tbc, masks)
        self.stats.update({'map_fixes': 0, 'map_rejected': 0})
        self.last_fix_t = None
        self.n_reject = 0
        self.t0_stress = None

    @property
    def DECODE_CAMS(self):
        return tuple(dict.fromkeys(self.VIS_CAMS + self.MAP_CAMS))

    def on_frames(self, fs):
        gray = self._gray(fs)
        cams = [c for c in self.MAP_CAMS if c in gray]
        feats = {c: self.pool.submit(self.loc.features, c, gray[c]) for c in cams}   # overlaps the heading step
        pose = super().on_frames(fs)
        x, y, yaw = self.ekf.X[:3]
        sd = float(np.sqrt(max(self.ekf.P[0, 0] + self.ekf.P[1, 1], 0)))
        radius = float(np.clip(3 * sd + 25, 40, 150))
        feats = {c: f.result() for c, f in feats.items()}
        if self._fix_dropped(fs.stamp_ns):
            cams = []
        res = self.pool.map(lambda c: self._seeded(self.loc.localize, c, gray[c], x, y, yaw, radius, feats[c]), cams)
        fixes = [r for r in res if r is not None and abs(np.angle(np.exp(1j * (r['yaw'] - yaw)))) < np.radians(10)]
        use = []
        if len(fixes) >= 2:                      # consensus between cameras
            P = np.array([[f['x'], f['y']] for f in fixes])
            med = np.median(P, axis=0)
            use = [f for f, p in zip(fixes, P) if np.hypot(*(p - med)) < self.AGREE_M]
            use = use if len(use) >= 2 else [f for f in fixes if f['n'] >= self.MIN_INL_SINGLE][:1]
        elif fixes and fixes[0]['n'] >= self.MIN_INL_SINGLE:
            use = fixes
        # a lone fix far outside the filter's uncertainty is more likely a wrong place match
        use = [f for f in use if len(use) >= 2 or np.hypot(f['x'] - x, f['y'] - y) < 3 * sd + 8]
        if use:
            mx, my = np.mean([f['x'] for f in use]), np.mean([f['y'] for f in use])
            myaw = float(np.angle(np.mean([np.exp(1j * f['yaw']) for f in use])))
            k = len(use)
            sxy, syaw = self.SIGMA_XY / np.sqrt(k), self.SIGMA_YAW / np.sqrt(k)
            # statistical consistency (NIS, 2-D position) once the filter is confident; repeated consistent
            # rejections mean the FILTER is wrong, so after REJECT_RESET in a row the fix is accepted anyway
            Sxy = self.ekf.P[:2, :2] + np.eye(2) * sxy ** 2
            r = np.array([mx - x, my - y])
            nis = float(r @ np.linalg.solve(Sxy, r))
            if nis > self.NIS_GATE and self.n_reject < self.REJECT_RESET:
                self.n_reject += 1
                use = []
            else:
                self.n_reject = 0
                self.ekf.update_pose(mx, my, myaw, sxy, syaw)
                self.stats['map_fixes'] += 1
                self.last_fix_t = fs.stamp_ns
        self.stats['map_rejected'] += len(fixes) - len(use)
        return self.ekf.pose

    # STRESS TEST ONLY (defaults off): simulate map-matching failure, e.g. a daylight test against a night map.
    # FIX_DROP: probability that a frame gets no map fix; FIX_GAP_S / FIX_PERIOD_S: no fixes for GAP s every PERIOD s.
    FIX_DROP, FIX_GAP_S, FIX_PERIOD_S = 0.0, 0.0, 0.0

    def _fix_dropped(self, t_ns):
        if self.t0_stress is None:
            self.t0_stress, self.rng_stress = t_ns, np.random.default_rng(0)
        if self.FIX_DROP > 0 and self.rng_stress.random() < self.FIX_DROP:
            return True
        return self.FIX_PERIOD_S > 0 and ((t_ns - self.t0_stress) / 1e9) % self.FIX_PERIOD_S >= self.FIX_PERIOD_S - self.FIX_GAP_S

    NIS_GATE = 13.8          # chi2(2 dof) 99.9 %
    REJECT_RESET = 15        # consecutive rejected fixes -> trust the map again

    PRIOR_AFTER_S = 2.0      # map priors only when map fixes have been missing this long (they bias lateral, EXP-0012)

    def priors_active(self, t_ns):
        return self.last_fix_t is None or (t_ns - self.last_fix_t) / 1e9 > self.PRIOR_AFTER_S


class VisualHeadingEst(VisualHeading):
    source = 'estimator'


ESTIMATORS = {'hold': HoldEstimator, 'odom': OdomEstimator, 'odom_bounds': OdomBounds,
              'odom_est': OdomEst, 'odom_est_bounds': OdomEstBounds, 'vis': VisualHeading, 'vis_est': VisualHeadingEst,
              'vismap': VisualMapLoc}
NEEDS_AMBIGUOUS = {'odom_est', 'odom_est_bounds', 'vis_est'}


def run(mcap, out, estimator='hold', state='allowed', start_s=0.0, duration_s=None, cams=C.CAMS, overrides=None,
        init=None):
    if estimator in NEEDS_AMBIGUOUS and state != 'ambiguous':
        raise SystemExit(f'{estimator} uses fused estimator fields: pass --state ambiguous (rules question Q1)')
    cameras, bounds = load_static(mcap, cams)
    track = TrackModel(bounds)
    if init is None:
        init = initial_pose(mcap)                       # first map -> base_link pose of the run (allowed)
    else:                                               # (x, y, yaw) given explicitly; stamp = first image
        init = (0, *init)
    est = ESTIMATORS[estimator](init, cameras, track)
    for k, v in (overrides or {}).items():            # tuning / ablation: estimator class parameters
        if not hasattr(est, k):
            raise SystemExit(f'{estimator} has no parameter {k}')
        setattr(est, k, type(getattr(est, k))(v))
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
            'late_drops': rp.late_drops, 'estimator_stats': getattr(est, 'stats', None), 'ms_per_frame_mean': float(proc.mean()) if len(proc) else None,
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
    ap.add_argument('--vmap', help='visual map (.npz); default camloc/assets/vmap.npz')
    ap.add_argument('--set', nargs='*', default=[], metavar='NAME=VALUE', help='override estimator parameters')
    ap.add_argument('--init', help='initial pose "x,y,yaw" if the test file has no /tf first pose')
    a = ap.parse_args(argv)
    if a.vmap:
        VisualMapLoc.VMAP = Path(a.vmap)
    ov = dict(kv.split('=', 1) for kv in a.set)
    init = tuple(float(v) for v in a.init.split(',')) if a.init else None
    print(json.dumps(run(a.mcap, a.out, a.estimator, a.state, a.start, a.duration, overrides=ov, init=init), indent=1))


if __name__ == '__main__':
    main()
