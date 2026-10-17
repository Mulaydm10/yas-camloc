"""TRAINING-TIME calibration of the car-state odometry (uses GT; never run at inference).

Fits the wheel rolling radius (per wheel-selection mode), and the bicycle yaw model
  wz * (L + L*K*v^2) + v*tan(delta0) = v*tan(delta(t-lag))      (linear in L, L*K, tan(delta0))
for a grid of steering lags. Models are compared on what matters for a filter: the drift over short
windows started from the GT pose (10 s, every 5 s, laps only). Also derives process-noise levels.

    python -m camloc.tools.calibrate_odometry      # writes camloc/assets/odometry.json
"""
import json

import numpy as np

from .. import config as C
from ..groundtruth import load_gt
from ..odometry import PARAMS, batch, integrate, wheel_rate
from ..track import wrap

WIN_S, STEP_S = 10.0, 5.0


def load_state():
    """All allowed + ambiguous fields at the state rate, read once (training only)."""
    path = C.CACHE / 'state_fields.npz'
    if path.exists():
        z = np.load(path)
        return z['t'], {k: z[k] for k in z.files if k != 't'}
    from ..data import Replay, StateEvent
    t, rows = [], []
    for ev in Replay(cams=(), state='ambiguous'):
        if isinstance(ev, StateEvent):
            t.append(ev.stamp_ns)
            rows.append(ev.fields)
    F = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    t = np.array(t, dtype=np.int64)
    C.CACHE.mkdir(exist_ok=True)
    np.savez(path, t=t, **F)
    return t, F


def gt_at(gt, t):
    yu = np.unwrap(gt['yaw'])
    return np.interp(t, gt['t'], gt['x']), np.interp(t, gt['t'], gt['y']), np.interp(t, gt['t'], yu)


def window_drift(t, v, wz, gt, mask_t):
    """Errors after WIN_S of integration from the GT pose, for windows starting every STEP_S."""
    gx, gy, gyaw = gt_at(gt, t)
    out = []
    for t0 in np.arange(t[0], t[-1] - WIN_S * 1e9, STEP_S * 1e9):
        i0, i1 = np.searchsorted(t, [t0, t0 + WIN_S * 1e9])
        if not mask_t[i0:i1].all():
            continue
        x, y, yaw = integrate(t[i0:i1], v[i0:i1], wz[i0:i1], gx[i0], gy[i0], gyaw[i0])
        dx, dy = x[-1] - gx[i1 - 1], y[-1] - gy[i1 - 1]
        h = gyaw[i1 - 1]
        out.append((np.cos(h) * dx + np.sin(h) * dy, -np.sin(h) * dx + np.cos(h) * dy, wrap(yaw[-1] - gyaw[i1 - 1])))
    e = np.array(out)
    return {'n_windows': len(e), 'lon_rmse_m': float(np.sqrt(np.mean(e[:, 0] ** 2))),
            'lat_rmse_m': float(np.sqrt(np.mean(e[:, 1] ** 2))),
            'yaw_rmse_deg': float(np.degrees(np.sqrt(np.mean(e[:, 2] ** 2)))),
            'pos_rmse_m': float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2)))}


def main():
    t, F = load_state()
    gt = load_gt()
    ts = t / 1e9
    gx, gy, gyaw = gt_at(gt, t)
    v_gt = np.hypot(np.gradient(gx, ts), np.gradient(gy, ts))
    wz_gt = np.gradient(gyaw, ts)
    # smooth the GT derivatives a little (106 Hz finite differences are noisy)
    k = np.ones(11) / 11
    v_gt, wz_gt = np.convolve(v_gt, k, 'same'), np.convolve(wz_gt, k, 'same')
    acc = np.interp(t, gt['t'], gt['pos_acc'])
    good = (acc <= 1.0) & (F['valid_wheelsspeeds_b'] > 0)
    moving = good & (v_gt > 5)

    report = {'speed': {}, 'yaw': {}, 'drift_10s': {}}
    radii = {}
    for mode in ('front', 'rear', 'mean4', 'antislip'):
        w = wheel_rate(F, mode)
        r = float(np.sum(v_gt[moving] * w[moving]) / np.sum(w[moving] ** 2))
        radii[mode] = r
        report['speed'][mode] = {'r': r, 'rmse_mps': float(np.sqrt(np.mean((r * w[moving] - v_gt[moving]) ** 2)))}
    best_mode = min(report['speed'], key=lambda m: report['speed'][m]['rmse_mps'])

    best = None
    v = radii[best_mode] * wheel_rate(F, best_mode)
    for lag in np.arange(0, 0.31, 0.02):
        d = np.interp(t - lag * 1e9, t, F['delta_wheel_rad'], left=F['delta_wheel_rad'][0])
        m = moving & (np.abs(d) < 0.3)
        A = np.c_[wz_gt[m], wz_gt[m] * v[m] ** 2, v[m]]
        b = v[m] * np.tan(d[m])
        (L, LK, td0), *_ = np.linalg.lstsq(A, b, rcond=None)
        p = {'r': radii[best_mode], 'wheel_mode': best_mode, 'L': float(L), 'K': float(LK / L),
             'delta0': float(np.arctan(td0)), 'lag_s': float(lag)}
        _, wz = batch(t, F, p)
        rm = float(np.sqrt(np.mean((wz[moving] - wz_gt[moving]) ** 2)))
        report['yaw'][f'lag {lag:.2f}'] = {'L': p['L'], 'K': p['K'], 'delta0_deg': float(np.degrees(p['delta0'])), 'wz_rmse': rm}
        if best is None or rm < best[0]:
            best = (rm, p)
    p = best[1]

    # compare on 10 s drift (the quantity a filter cares about); laps only (exclude the pit-lane start/end)
    lap_mask = (ts - ts[0] > 90) & (ts - ts[0] < 1160) & good
    for name, (vv, ww) in {
        'wheels_bicycle (chosen)': batch(t, F, p),
        'wheels_bicycle no lag/understeer': batch(t, F, {**p, 'lag_s': 0.0, 'K': 0.0}),
        'estimator v_mps+wz (ambiguous)': batch(t, F, p, 'estimator'),
        'wheels v + estimator wz (ambiguous)': (batch(t, F, p)[0], F['wz_radps']),
    }.items():
        report['drift_10s'][name] = window_drift(t, vv, ww, gt, lap_mask)

    # process noise (per sqrt(s)) from the chosen model's input errors, inflated so that a random walk
    # reproduces the empirical 10 s drift (errors are time-correlated, a white-noise fit underestimates)
    vv, ww = batch(t, F, p)
    dr = report['drift_10s']['wheels_bicycle (chosen)']
    # continuous white input noise q: integrated error std after T seconds = q*v*sqrt(T) (speed), q*sqrt(T) (yaw)
    p['q'] = {'sigma_v_frac': float(dr['lon_rmse_m'] / (np.mean(v_gt[lap_mask]) * np.sqrt(WIN_S))),
              'sigma_wz': float(np.radians(dr['yaw_rmse_deg']) / np.sqrt(WIN_S)),
              'sigma_v_abs': 0.2}
    p['fit'] = {'speed_rmse_mps': report['speed'][best_mode]['rmse_mps'], 'wz_rmse_radps': best[0],
                'drift_10s': dr, 'note': 'fitted on the training run with GT (training-time calibration)'}
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(p, indent=1))
    (C.CACHE / 'odometry_calibration.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({'speed': report['speed'], 'chosen': {k: v for k, v in p.items() if k != 'fit'},
                      'wz_rmse': best[0], 'drift_10s': report['drift_10s']}, indent=1))


if __name__ == '__main__':
    main()
