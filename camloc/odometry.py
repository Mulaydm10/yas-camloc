"""Car-state odometry: speed and yaw rate from the allowed signals (wheel speeds + steering), with
parameters fitted on the training run by camloc/tools/calibrate_odometry.py (camloc/assets/odometry.json).

Speed:   v = r * w, with w chosen per sample to dodge wheel slip: under braking the fronts lock first
         (use the faster axle), under traction the rears spin (use the front axle).
Yaw rate (kinematic bicycle with understeer and steering lag):
         wz = v * tan(delta(t - lag) - delta0) / (L * (1 + K * v^2))
`source='estimator'` instead uses the fused v_mps / wz_radps fields (only if the organisers allow them).
"""
import json
from collections import deque

import numpy as np

from . import config as C

PARAMS = C.ASSETS / 'odometry.json'


def wheel_rate(f, mode):
    fl, fr, rl, rr = f['omega_w_fl'], f['omega_w_fr'], f['omega_w_rl'], f['omega_w_rr']
    front, rear = 0.5 * (fl + fr), 0.5 * (rl + rr)
    if mode == 'front':
        return front
    if mode == 'rear':
        return rear
    if mode == 'mean4':
        return 0.25 * (fl + fr + rl + rr)
    if mode == 'antislip':           # braking: fronts lock -> faster axle; traction: rears spin -> front
        return np.where(front < rear, rear, front) if np.ndim(front) else (rear if front < rear else front)
    raise ValueError(mode)


class Odometry:
    """Streaming: feed every StateEvent; read (v, wz) at any time (values of the latest sample <= t)."""

    def __init__(self, params=None, source='wheels'):
        self.p = params or json.loads(PARAMS.read_text())
        self.source = source
        self.delta_hist = deque()        # (stamp_ns, delta) for the steering lag
        self.v = 0.0
        self.wz = 0.0
        self.t = None

    def update(self, stamp_ns, f):
        self.t = stamp_ns
        if self.source == 'estimator':
            self.v, self.wz = f['v_mps'], f['wz_radps']
            return
        p = self.p
        self.v = p['r'] * float(wheel_rate(f, p['wheel_mode']))
        self.delta_hist.append((stamp_ns, f['delta_wheel_rad']))
        lag_ns = p['lag_s'] * 1e9
        while len(self.delta_hist) > 1 and self.delta_hist[1][0] <= stamp_ns - lag_ns:
            self.delta_hist.popleft()
        delta = self.delta_hist[0][1]
        self.wz = self.v * np.tan(delta - p['delta0']) / (p['L'] * (1 + p['K'] * self.v ** 2))


def batch(t_ns, F, p, source='wheels'):
    """Vectorised version over arrays (for calibration and analysis). Same maths as Odometry."""
    if source == 'estimator':
        return F['v_mps'], F['wz_radps']
    v = p['r'] * wheel_rate(F, p['wheel_mode'])
    d = F['delta_wheel_rad']
    if p['lag_s'] > 0:
        d = np.interp(t_ns - p['lag_s'] * 1e9, t_ns, d, left=d[0])
    return v, v * np.tan(d - p['delta0']) / (p['L'] * (1 + p['K'] * v ** 2))


def integrate(t_ns, v, wz, x0, y0, yaw0):
    """Midpoint unicycle integration (vectorised cumulative form). Returns x, y, yaw at every sample."""
    dt = np.r_[0, np.diff(t_ns) / 1e9]
    wz_prev = np.r_[0, wz[:-1]]
    v_prev = np.r_[0, v[:-1]]
    yaw = yaw0 + np.cumsum(wz_prev * dt)
    ym = yaw - 0.5 * wz_prev * dt
    x = x0 + np.cumsum(v_prev * np.cos(ym) * dt)
    y = y0 + np.cumsum(v_prev * np.sin(ym) * dt)
    return x, y, yaw
