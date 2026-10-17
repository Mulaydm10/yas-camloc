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
        adj = json.loads((C.ROOT / 'camloc' / 'assets' / 'camera_adjust.json').read_text())
        masks = load_masks(self.VIS_CAMS)
        self.kr = {}
        for c in self.VIS_CAMS:
            Rbc = (cameras[c].T_base_cam @ rot_adjust(*adj.get(c, (0, 0, 0))))[:3, :3]
            self.kr[c] = KeyframeRotation(cameras[c], masks[c][::2, ::2][:364, :753], Rbc)
        scales = json.loads((C.ROOT / 'camloc' / 'assets' / 'vis_rot_scale.json').read_text())
        self.scale = {c: scales.get(c, 1.0) for c in self.VIS_CAMS}
        self.kf_t = None
        self.lane = str(track.locate(self.x, self.y, self.yaw)[0][0])
        self.stats = {'updates': 0, 'rejected': 0, 'keyframes': 0, 'lane_switches': 0}

    def on_frames(self, fs):
        self._advance(fs.stamp_ns)
        gray = {c: self.cv2.imdecode(np.frombuffer(fs.images[c], np.uint8), self.cv2.IMREAD_REDUCED_GRAYSCALE_2)
                for c in self.VIS_CAMS if c in fs.images}
        if self.kf_t is None:
            self._new_keyframe(fs.stamp_ns, gray)
        else:
            meas = [m[0] / self.scale[c] for c, g in gray.items() if (m := self.kr[c].track(g)) is not None]
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
            self.ekf.constrain_to_lane(self.track, lane_name=self.lane)
        return self.ekf.pose

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
        for c, g in gray.items():
            self.kr[c].reset(g)
        self.ekf.clone_heading()
        self.kf_t = t
        self.stats['keyframes'] += 1


class VisualHeadingEst(VisualHeading):
    source = 'estimator'


ESTIMATORS = {'hold': HoldEstimator, 'odom': OdomEstimator, 'odom_bounds': OdomBounds,
              'odom_est': OdomEst, 'odom_est_bounds': OdomEstBounds, 'vis': VisualHeading, 'vis_est': VisualHeadingEst}
NEEDS_AMBIGUOUS = {'odom_est', 'odom_est_bounds', 'vis_est'}


def run(mcap, out, estimator='hold', state='allowed', start_s=0.0, duration_s=None, cams=C.CAMS):
    if estimator in NEEDS_AMBIGUOUS and state != 'ambiguous':
        raise SystemExit(f'{estimator} uses fused estimator fields: pass --state ambiguous (rules question Q1)')
    cameras, bounds = load_static(mcap, cams)
    track = TrackModel(bounds)
    init = initial_pose(mcap)
    est = ESTIMATORS[estimator](init, cameras, track)
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
    a = ap.parse_args(argv)
    print(json.dumps(run(a.mcap, a.out, a.estimator, a.state, a.start, a.duration), indent=1))


if __name__ == '__main__':
    main()
