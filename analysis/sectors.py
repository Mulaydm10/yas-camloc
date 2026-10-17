"""Sector difficulty (step 2.7): split the lap into 10 equal-length sectors of the centreline and report
speed, curvature, visual-cue proxies (ORB keypoints, lower-half Canny edge density in the wide cameras)
and GT-quality flags. Only on-track samples (pit lane excluded) are used.

Writes analysis/out/sectors.json, analysis/sectors.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import FIG, OUT

NSEC = 10


def main():
    tm = np.load(OUT / 'track_model.npz')
    g = np.load(OUT / 'gt_track.npz')
    L = float(tm['length'])
    cs = np.load(OUT / 'cam_stats.npz')
    F = list(cs['fields'])
    se = np.load(OUT / 'state_est.npz')
    K = list(se['keys'])
    pa = np.hypot(se['data'][:, K.index('pos_accuracy[0]')], se['data'][:, K.index('pos_accuracy[1]')])

    st = cs['fl'][:, F.index('stamp_ns')].astype(np.int64)
    j = np.clip(np.searchsorted(g['t'], st), 0, len(g['t']) - 1)
    s, v, on = g['s'][j], g['v_se'][j], g['in_track'][j] & ~g['in_pit'][j]
    yr = np.abs(g['yr'][j])
    pa_c = np.interp(st, se['t_ns'], pa)
    sec = np.minimum((s / L * NSEC).astype(int), NSEC - 1)
    ks = np.minimum((tm['s'] / L * NSEC).astype(int), NSEC - 1)
    kap = np.abs(tm['kappa'])

    def cam(c, f):
        a = cs[c]
        # camera frames are index-aligned with fl to within 0.01 ms except r (5 drops) -> match by stamp
        tt = a[:, F.index('stamp_ns')].astype(np.int64)
        k = np.clip(np.searchsorted(tt, st), 0, len(tt) - 1)
        return a[k, F.index(f)]

    orb_w = 0.5 * (cam('cl', 'orb') + cam('cr', 'orb'))
    edge_w = 0.5 * (cam('cl', 'edge_low') + cam('cr', 'edge_low'))
    orb_f = cam('fl', 'orb')
    bright = cam('fr', 'mean')
    # where are the GT-accuracy blips? (pos_accuracy > 1 m on track)
    blip_s = s[on & (pa_c > 1)]
    rows = []
    for i in range(NSEC):
        m = on & (sec == i)
        mk = ks == i
        straight = float(np.mean(kap[mk] < 1 / 500))
        rows.append({
            'sector': i + 1, 's_from_m': round(i * L / NSEC), 's_to_m': round((i + 1) * L / NSEC),
            'n_frames': int(m.sum()), 'speed_mean': float(v[m].mean()), 'speed_max': float(v[m].max()),
            'yawrate_abs_mean': float(np.nanmean(yr[m])),
            'kappa_abs_max': float(kap[mk].max()), 'min_radius_m': float(1 / kap[mk].max()),
            'straight_share': straight,
            'orb_wide_median': float(np.median(orb_w[m])), 'edge_wide_median': float(np.median(edge_w[m])),
            'orb_fl_median': float(np.median(orb_f[m])), 'brightness_fr_median': float(np.median(bright[m])),
            'gt_blip_frames': int(np.sum((sec == i) & on & (pa_c > 1))),
        })
    # difficulty score: high speed, straight (few lateral cues to pin s), low cue counts (z-scores)
    arr = lambda k: np.array([r[k] for r in rows])
    zs = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    score = zs(arr('speed_mean')) + zs(arr('straight_share')) - zs(arr('orb_wide_median')) - zs(arr('edge_wide_median'))
    for r, sc in zip(rows, score):
        r['difficulty_z'] = float(sc)
    out = {'length_m': L, 'sectors': rows, 'gt_blip_s_positions_m': sorted(set(int(x) for x in np.round(blip_s, -1)))}
    (OUT / 'sectors.json').write_text(json.dumps(out, indent=1))
    print(f"{'sec':>3} {'s range':>11} {'n':>5} {'v_mean':>6} {'v_max':>5} {'Rmin':>6} {'strt%':>5} {'orbW':>5} {'edgeW':>6} {'orbFL':>5} {'blip':>4} {'diff':>5}")
    for r in rows:
        print(f"{r['sector']:3d} {r['s_from_m']:5d}-{r['s_to_m']:<5d} {r['n_frames']:5d} {r['speed_mean']:6.1f} {r['speed_max']:5.1f} "
              f"{r['min_radius_m']:6.0f} {r['straight_share'] * 100:5.0f} {r['orb_wide_median']:5.0f} {r['edge_wide_median']:6.3f} "
              f"{r['orb_fl_median']:5.0f} {r['gt_blip_frames']:4d} {r['difficulty_z']:5.2f}")
    print('GT blip s positions:', out['gt_blip_s_positions_m'])

    c = tm['c']
    b = np.load(OUT / 'track_bounds.npz')
    fig, ax = plt.subplots(figsize=(10, 11))
    for k in b:
        ax.plot(b[k][:, 0], b[k][:, 1], color='0.6', lw=0.5)
    cmap = plt.get_cmap('RdYlGn_r')
    d = (score - score.min()) / (np.ptp(score) + 1e-9)
    for i in range(NSEC):
        mk = ks == i
        ax.plot(c[mk, 0], c[mk, 1], color=cmap(d[i]), lw=5)
        mid = c[mk][len(c[mk]) // 2]
        ax.annotate(f'S{i + 1}\n{rows[i]["speed_mean"]:.0f} m/s', mid, fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.plot(*c[0], 'k*', ms=12)
    ax.set_aspect('equal'); ax.set_title('Sectors coloured by difficulty proxy (red = harder); star = s=0, driving direction = increasing s')
    fig.tight_layout(); fig.savefig(FIG / 'sectors.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
