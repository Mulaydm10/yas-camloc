"""Causal dead-reckoning baselines (step 2.5). Initialised from the FIRST GT pose only (except 5c).

  (a) v_mps + wz_radps, unicycle integration
  (b) wheel speeds * fitted rolling radius + steering (delta_wheel_rad) through a kinematic bicycle
      model with fitted wheelbase. Radius and wheelbase are FIT ON GT (training-time calibration only).
  (c) = (a) re-initialised from GT every N s (N = 5, 10, 30) -- DIAGNOSTIC ONLY, uses GT at inference.

Integration runs at the state_estimation rate (~106 Hz) using only samples with stamp <= t; poses are
emitted at every camera_fl header stamp. Scored with tools/evaluate.py.

Writes analysis/out/sub_*.csv, analysis/out/deadreckon.json, analysis/deadreckon_err.png
"""
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, FIG, OUT, wrap

EVAL = DATA / 'tools' / 'evaluate.py'


def load():
    z = np.load(OUT / 'state_est.npz')
    keys = list(z['keys'])
    D = z['data']
    get = lambda k: D[:, keys.index(k)]
    t = z['t_ns']
    o = np.argsort(t)
    return t[o], {k: get(k)[o] for k in keys}


def integrate(t, v, wz, x0, y0, yaw0, resets=None):
    """Unicycle. resets: dict idx -> (x, y, yaw) applied before step idx (diagnostic only)."""
    n = len(t)
    X, Y, P = np.empty(n), np.empty(n), np.empty(n)
    x, y, p = x0, y0, yaw0
    for i in range(n):
        if resets is not None and i in resets:
            x, y, p = resets[i]
        if i > 0:
            dt = (t[i] - t[i - 1]) / 1e9
            pm = p + 0.5 * wz[i - 1] * dt          # midpoint heading
            x += v[i - 1] * np.cos(pm) * dt
            y += v[i - 1] * np.sin(pm) * dt
            p += wz[i - 1] * dt
        X[i], Y[i], P[i] = x, y, p
    return X, Y, wrap(P)


def sample_causal(t_src, arrs, t_out):
    j = np.searchsorted(t_src, t_out, side='right') - 1       # last sample with stamp <= t_out
    ok = j >= 0
    return ok, [a[np.clip(j, 0, None)] for a in arrs]


def write_sub(path, t, x, y, yaw):
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,yaw\n')
        for a, b, c, d in zip(t, x, y, yaw):
            f.write(f'{int(a)},{b:.4f},{c:.4f},{d:.6f}\n')


def score(path):
    out = subprocess.run([sys.executable, str(EVAL), str(OUT / 'gt.csv'), str(path)],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        p = line.split()
        if line.startswith('position RMSE'):
            res['pos_rmse'] = float(p[3])
        elif line.startswith('position median'):
            res['median'], res['p95'] = float(p[2]), float(p[4])
        elif line.startswith('lateral RMSE'):
            res['lat_rmse'] = float(p[3])
        elif line.startswith('longitudinal'):
            res['lon_rmse'] = float(p[3])
        elif line.startswith('yaw RMSE'):
            res['yaw_rmse_deg'] = float(p[3])
        elif line.startswith('within 1.00'):
            res['within_1m_pct'] = float(p[3])
        elif line.startswith('within 0.50'):
            res['within_0.5m_pct'] = float(p[3])
        elif line.startswith('coverage'):
            res['coverage_pct'] = float(p[4])
    return res, out


def err_vs_gt(t_out, x, y, gt):
    gx = np.interp(t_out, gt[:, 0], gt[:, 1])
    gy = np.interp(t_out, gt[:, 0], gt[:, 2])
    return np.hypot(x - gx, y - gy)


def first_exceed(tt, e, thr):
    i = np.where(e > thr)[0]
    return float(tt[i[0]]) if len(i) else None


def main():
    t, F = load()
    gt = np.load(OUT / 'gt.npy')
    cam_t = np.load(OUT / 'image_index.npz')['fl'][:, 0]
    x0, y0, yaw0 = gt[0, 1], gt[0, 2], gt[0, 4]
    tt_cam = (cam_t - cam_t[0]) / 1e9
    results, errs = {}, {}

    # (a)
    X, Y, P = integrate(t, F['v_mps'], F['wz_radps'], x0, y0, yaw0)
    runs = {'a_v_wz': (X, Y, P)}

    # (b) fit rolling radius r: v_gt ~ r * mean(omega); wheelbase Lwb: wz_gt ~ v tan(delta) / Lwb  (GT fit)
    om = 0.25 * (F['omega_w_fl'] + F['omega_w_fr'] + F['omega_w_rl'] + F['omega_w_rr'])
    om_f = 0.5 * (F['omega_w_fl'] + F['omega_w_fr'])
    gk = 10
    gvx = np.gradient(np.interp(t, gt[:, 0], gt[:, 1]), t / 1e9)
    gvy = np.gradient(np.interp(t, gt[:, 0], gt[:, 2]), t / 1e9)
    v_gt = np.hypot(gvx, gvy)
    m = (om_f > 5) & (F['valid_wheelsspeeds_b'] > 0)
    r_f = float(np.sum(v_gt[m] * om_f[m]) / np.sum(om_f[m] ** 2))
    r_all = float(np.sum(v_gt[m] * om[m]) / np.sum(om[m] ** 2))
    v_wh = r_f * om_f
    yaw_gt_u = np.unwrap(np.interp(t, gt[:, 0], np.unwrap(gt[:, 4])))
    wz_gt = np.gradient(yaw_gt_u, t / 1e9)
    td = np.tan(F['delta_wheel_rad'])
    mm = m & (np.abs(td) > 0.01) & (v_wh > 5)
    # least squares wz_gt ~ v*tan(delta)/L - v*tan(delta0)/L  ->  [v*td, v] @ [1/L, -tan(delta0)/L]
    A = np.c_[v_wh[mm] * td[mm], v_wh[mm]]
    coef = np.linalg.lstsq(A, wz_gt[mm], rcond=None)[0]
    Lwb = float(1 / coef[0])
    delta0 = float(np.arctan(-coef[1] * Lwb))
    inv_L_nobias = float(np.sum(wz_gt[mm] * v_wh[mm] * td[mm]) / np.sum((v_wh[mm] * td[mm]) ** 2))
    wz_b = v_wh * (td - np.tan(delta0)) / Lwb
    wz_b0 = v_wh * td / (1 / inv_L_nobias)
    runs['b0_wheels_bicycle_nobias'] = integrate(t, v_wh, wz_b0, x0, y0, yaw0)
    Xb, Yb, Pb = integrate(t, v_wh, wz_b, x0, y0, yaw0)
    runs['b_wheels_bicycle'] = (Xb, Yb, Pb)
    fit = {'rolling_radius_front_m': r_f, 'rolling_radius_all4_m': r_all, 'wheelbase_fit_m': Lwb, 'steer_offset_deg': float(np.degrees(delta0)), 'wheelbase_fit_nobias_m': 1 / inv_L_nobias,
           'wheel_speed_vs_gt_rmse_mps': float(np.sqrt(np.mean((v_wh[m] - v_gt[m]) ** 2))),
           'bicycle_wz_vs_gt_rmse_radps': float(np.sqrt(np.mean((wz_b[mm] - wz_gt[mm]) ** 2))),
           'wz_field_vs_gt_rmse_radps': float(np.sqrt(np.mean((F['wz_radps'] - wz_gt) ** 2))),
           'v_field_vs_gt_rmse_mps': float(np.sqrt(np.mean((F['v_mps'] - v_gt) ** 2))),
           'wz_field_bias_radps': float(np.mean(F['wz_radps'] - wz_gt))}

    # (b2) wheel speed + wz field (mix): isolates heading vs speed error source
    Xm, Ym, Pm = integrate(t, v_wh, F['wz_radps'], x0, y0, yaw0)
    runs['b2_wheels_v_plus_wz'] = (Xm, Ym, Pm)

    # (c) resets from GT every N s (diagnostic)
    for N in (5, 10, 30):
        resets = {}
        gi = 0
        for tr in np.arange(t[0], t[-1], N * 1e9):
            i = int(np.searchsorted(t, tr))
            g = int(np.searchsorted(gt[:, 0], t[i], side='right') - 1)
            resets[i] = (gt[g, 1], gt[g, 2], gt[g, 4])
        runs[f'c_reset_{N}s'] = integrate(t, F['v_mps'], F['wz_radps'], x0, y0, yaw0, resets)

    for name, (X_, Y_, P_) in runs.items():
        ok, (xs, ys, ps) = sample_causal(t, [X_, Y_, P_], cam_t)
        path = OUT / f'sub_{name}.csv'
        write_sub(path, cam_t[ok], xs[ok], ys[ok], ps[ok])
        r, raw = score(path)
        e = err_vs_gt(cam_t[ok], xs[ok], ys[ok], gt)
        r.update({f'first_exceed_{k}m_s': first_exceed(tt_cam[ok], e, k) for k in (1, 5, 20)})
        r['final_err_m'] = float(e[-1])
        results[name] = r
        errs[name] = (tt_cam[ok], e)
        print(f'--- {name}\n{raw}', flush=True)
    results['fit'] = fit
    (OUT / 'deadreckon.json').write_text(json.dumps(results, indent=1))
    print(json.dumps(fit, indent=1))

    fig, ax = plt.subplots(figsize=(13, 5))
    for name, (a, e) in errs.items():
        ax.plot(a, e, lw=0.8, label=name)
    ax.set_yscale('log'); ax.set_xlabel('t [s]'); ax.set_ylabel('position error [m]')
    for thr in (1, 5, 20):
        ax.axhline(thr, color='gray', ls=':', lw=0.6)
    ax.legend(); ax.set_title('Dead-reckoning position error vs GT (camera_fl stamps)')
    fig.tight_layout(); fig.savefig(FIG / 'deadreckon_err.png', dpi=100); plt.close(fig)

    b = np.load(OUT / 'track_bounds.npz')
    fig, ax = plt.subplots(figsize=(9, 10))
    for k in b:
        ax.plot(b[k][:, 0], b[k][:, 1], 'k', lw=0.4)
    ax.plot(gt[:, 1], gt[:, 2], 'g', lw=0.6, label='GT')
    for name in ('a_v_wz', 'b_wheels_bicycle'):
        ax.plot(runs[name][0], runs[name][1], lw=0.6, label=name)
    ax.set_aspect('equal'); ax.legend(); ax.set_title('Dead reckoning (no resets) vs GT')
    fig.tight_layout(); fig.savefig(FIG / 'deadreckon_traj.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
