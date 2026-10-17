"""Car-state field audit (step 2.4): range, update rate, constant/invalid, and evidence of being
derived from the GT pose. Writes analysis/out/fields.json and prints a table.
"""
import json

import numpy as np

from common import OUT, wrap


def main():
    z = np.load(OUT / 'state_est.npz')
    K = list(z['keys'])
    D = z['data']
    t = z['t_ns']
    dur = (t[-1] - t[0]) / 1e9
    F = {k: D[:, i] for i, k in enumerate(K)}
    gt = np.load(OUT / 'gt.npy')
    gx, gy = np.interp(t, gt[:, 0], gt[:, 1]), np.interp(t, gt[:, 0], gt[:, 2])
    gyaw = np.interp(t, gt[:, 0], np.unwrap(gt[:, 4]))
    rows = {}
    for k in K[2:]:
        v = F[k]
        ch = np.sum(np.diff(v) != 0)
        rows[k] = {'min': float(v.min()), 'max': float(v.max()), 'update_hz': float(ch / dur),
                   'constant': bool(ch == 0)}
    ev = {}
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    ev['x_m,y_m vs /tf GT (m)'] = rms(np.hypot(F['x_m'] - gx, F['y_m'] - gy))
    ev['yaw_rad vs /tf GT (deg)'] = float(np.degrees(rms(wrap(F['yaw_rad'] - gyaw))))
    ev['yaw_vel_rad vs yaw_rad+beta_rad (deg)'] = float(np.degrees(rms(wrap(F['yaw_vel_rad'] - F['yaw_rad'] - F['beta_rad']))))
    ev['yaw_vel_rad vs yaw_rad (deg)'] = float(np.degrees(rms(wrap(F['yaw_vel_rad'] - F['yaw_rad']))))
    mv = F['v_mps'] > 5
    ev['kappa vs wz/v (1/m, v>5)'] = rms(F['kappa_radpm'][mv] - F['wz_radps'][mv] / F['v_mps'][mv])
    ev['ax_vel vs ax (m/s2)'] = rms(F['ax_vel_mps2'] - F['ax_mps2'])
    ev['v_mps vs vx_mps (m/s)'] = rms(F['v_mps'] - F['vx_mps'])
    ev['v_mps vs v_raw_mps (m/s)'] = rms(F['v_mps'] - F['v_raw_mps'])
    ev['beta vs atan(vy/vx) (deg, v>5)'] = float(np.degrees(rms(F['beta_rad'][mv] - np.arctan2(F['vy_mps'][mv], F['vx_mps'][mv]))))
    ev['timestamp vs header stamp (ms)'] = rms((F['timestamp'] - t / 1e9) * 1e3)
    # GT yaw rate from /tf vs wz field
    k = 10
    wz_gt = np.gradient(gyaw, t / 1e9)
    ev['wz_radps vs d(yaw_GT)/dt (rad/s)'] = rms(F['wz_radps'] - wz_gt)
    ev['ay vs v*wz (m/s2, v>5)'] = rms(F['ay_mps2'][mv] - F['v_mps'][mv] * F['wz_radps'][mv])
    # wheel-speed-only velocity vs v_mps: if v_mps were pure wheel odometry these would match closely
    omr = 0.307 * 0.5 * (F['omega_w_fl'] + F['omega_w_fr'])
    ev['v_mps vs 0.307*front wheel omega (m/s)'] = rms(F['v_mps'] - omr)
    # lagged correlation of v_raw with GT speed (does v_raw lead/lag?)
    out = {'fields': rows, 'evidence': ev, 'duration_s': dur}
    (OUT / 'fields.json').write_text(json.dumps(out, indent=1))
    for k_, r in rows.items():
        print(f"{k_:46s} {r['min']:11.4g} {r['max']:11.4g} upd {r['update_hz']:7.1f} Hz {'CONST' if r['constant'] else ''}")
    print(json.dumps(ev, indent=1))


if __name__ == '__main__':
    main()
