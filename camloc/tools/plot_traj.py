"""EVALUATION: plot an estimated trajectory against the GT on the track map (deliverable "visualisation").

    python -m camloc.tools.plot_traj cache/sub.csv --laps 4 --out docs/traj_lap4.png
Panels: track map with GT and estimate coloured by position error; error vs time (lateral / longitudinal).
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..data import load_static
from ..evaluate import gt_context, lap_windows, load_sub, pose_errors
from ..groundtruth import load_gt
from ..track import TrackModel


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('sub')
    ap.add_argument('--laps', type=int, nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    a = ap.parse_args(argv)
    gt = load_gt()
    bounds = load_static(cams=())[1]
    tm = TrackModel(bounds)
    sub = load_sub(a.sub)
    if a.laps:
        lw = {n: (t0, t1) for n, t0, t1 in lap_windows(gt_context(gt, tm))}
        m = np.zeros(len(sub['t']), bool)
        for n in a.laps:
            m |= (sub['t'] >= lw[n][0]) & (sub['t'] <= lw[n][1])
        sub = {k: v[m] for k, v in sub.items()}
    e = pose_errors(gt, sub)
    t = (e['t'] - e['t'][0]) / 1e9
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.03, 0.05, 0.5, 0.9])
    for k, v in bounds.items():
        ax.plot(v[:, 0], v[:, 1], color='0.75' if 'pit' in k else '0.45', lw=0.6)
    ax.plot(e['gx'], e['gy'], color='0.1', lw=1.0, label='ground truth')
    sc = ax.scatter(sub['x'][:len(e['t'])] if len(sub['x']) == len(e['t']) else np.interp(e['t'], sub['t'], sub['x']),
                    np.interp(e['t'], sub['t'], sub['y']), c=np.minimum(e['pos'], 3), s=4, cmap='RdYlGn_r', vmin=0, vmax=3,
                    label='estimate')
    plt.colorbar(sc, ax=ax, shrink=0.5, label='position error [m] (clipped at 3)')
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    ax.set_title(a.title or f'{a.sub} laps {a.laps or "all"}')
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    b1 = fig.add_axes([0.6, 0.55, 0.38, 0.38])
    b1.plot(t, e['lat'], lw=0.7, label=f'lateral (RMSE {rms(e["lat"]):.2f} m)')
    b1.plot(t, e['lon'], lw=0.7, label=f'longitudinal (RMSE {rms(e["lon"]):.2f} m)')
    b1.axhline(0, color='k', lw=0.5)
    b1.set_ylabel('error [m]')
    b1.legend(fontsize=8)
    b1.set_title(f'position RMSE {rms(e["pos"]):.3f} m, within 1 m {np.mean(e["pos"] < 1) * 100:.1f} %')
    b2 = fig.add_axes([0.6, 0.08, 0.38, 0.38], sharex=b1)
    b2.plot(t, e['dyaw'], lw=0.7, color='tab:purple', label=f'heading (RMSE {rms(e["dyaw"]):.2f} deg)')
    b2.axhline(0, color='k', lw=0.5)
    b2.set_xlabel('t [s]')
    b2.set_ylabel('heading error [deg]')
    b2.legend(fontsize=8)
    fig.savefig(a.out, dpi=110)
    print(a.out)


if __name__ == '__main__':
    main()
