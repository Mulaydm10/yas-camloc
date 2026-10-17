"""Track map + ground-truth analysis (steps 2.2 and 2.3).

Builds a 1 m centreline from the surveyed borders, measures width and curvature, projects the GT
trajectory onto it (s, n, heading error), finds laps and pit passes, and plots.

Writes analysis/out/track_model.npz, analysis/out/gt_track.npz, analysis/out/track_gt.json,
analysis/traj_speed.png, analysis/time_series.png, analysis/track_profile.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from common import FIG, OUT, wrap


def densify(p, step=0.1):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def resample_closed(p, step=1.0):
    p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    n = int(round(d[-1] / step))
    s = np.linspace(0, d[-1], n, endpoint=False)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], d[-1]


def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def build_centreline(L, R):
    Rd = densify(R)
    _, j = cKDTree(Rd).query(L)
    mid = 0.5 * (L + Rd[j])
    c, _ = resample_closed(mid, 1.0)
    # smooth (wrap-around) then resample again to exact 1 m spacing
    k = 15
    ext = np.vstack([c[-k:], c, c[:k]])
    ext = savgol_filter(ext, 21, 3, axis=0)
    c, length = resample_closed(ext[k:-k], 1.0)
    return c, length


def geometry(c):
    n = len(c)
    d = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    hdg = np.arctan2(d[:, 1], d[:, 0])
    ds = np.hypot(*(np.roll(c, -1, axis=0) - c).T)
    dh = wrap(np.roll(hdg, -1) - np.roll(hdg, 1))
    kappa = dh / (2 * ds.mean())
    ext = np.r_[kappa[-10:], kappa, kappa[:10]]
    kappa = savgol_filter(ext, 21, 2)[10:-10]
    return hdg, kappa


def main():
    b = np.load(OUT / 'track_bounds.npz')
    L, R = b['track_left'][:, :2], b['track_right'][:, :2]
    PL, PR = b['pitlane_left'][:, :2], b['pitlane_right'][:, :2]
    c, length = build_centreline(L, R)
    hdg, kappa = geometry(c)
    s_c = np.arange(len(c)) * (length / len(c))
    nrm = np.c_[-np.sin(hdg), np.cos(hdg)]          # left normal
    # width: distance from centreline to each border along the normal, approximated by nearest point
    dl, _ = cKDTree(densify(np.vstack([L, L[:1]]))).query(c)
    dr, _ = cKDTree(densify(np.vstack([R, R[:1]]))).query(c)
    # which border is on the left of the driving direction is decided below from the GT

    gt = np.load(OUT / 'gt.npy')
    t, x, y, yaw = gt[:, 0].astype(np.int64), gt[:, 1], gt[:, 2], gt[:, 4]
    tt = (t - t[0]) / 1e9
    dtn = np.diff(t)
    # project GT onto centreline
    tree = cKDTree(c)
    _, j = tree.query(np.c_[x, y])
    rel = np.c_[x, y] - c[j]
    tng = np.c_[np.cos(hdg[j]), np.sin(hdg[j])]
    along = np.sum(rel * tng, axis=1)
    s = (s_c[j] + along) % length
    n = np.sum(rel * nrm[j], axis=1)
    epsi = wrap(yaw - hdg[j])
    direction = np.sign(np.median(np.cos(epsi)))
    if direction < 0:                       # make the centreline follow the driving direction
        c, nrm = c[::-1], -nrm[::-1]
        dl, dr = dr[::-1], dl[::-1]
        hdg, kappa = wrap(hdg[::-1] + np.pi), -kappa[::-1]
        tree = cKDTree(c)
        _, j = tree.query(np.c_[x, y])
        rel = np.c_[x, y] - c[j]
        tng = np.c_[np.cos(hdg[j]), np.sin(hdg[j])]
        s = (s_c[j] + np.sum(rel * tng, axis=1)) % length
        n = np.sum(rel * nrm[j], axis=1)
        epsi = wrap(yaw - hdg[j])
    # left/right width in driving direction: decide which border lies on +normal side
    side_L = np.median(np.sum((cKDTree(L).query(c)[1][:, None] * 0 + L[cKDTree(L).query(c)[1]] - c) * nrm, axis=1))
    w_left, w_right = (dl, dr) if side_L > 0 else (dr, dl)

    # in-track / in-pit via polygons: the track is an annulus between the two border loops
    outer, inner = (L, R) if abs(area(L)) > abs(area(R)) else (R, L)
    pts = np.c_[x, y]
    in_track = Path(outer).contains_points(pts) & ~Path(inner).contains_points(pts)
    pit_poly = np.vstack([PL, PR[::-1]])
    in_pit_poly = Path(pit_poly).contains_points(pts)
    in_pit = in_pit_poly & ~in_track
    # also count pit when outside the track and near the pit-lane borders (polygon may be imperfect)
    dpit = np.minimum(cKDTree(densify(PL)).query(pts)[0], cKDTree(densify(PR)).query(pts)[0])
    off = ~in_track
    in_pit = in_pit | (off & (dpit < 15))

    # speed / yaw rate from GT finite differences over ~0.1 s
    k = 10
    vx = (x[k:] - x[:-k]) / ((t[k:] - t[:-k]) / 1e9)
    vy = (y[k:] - y[:-k]) / ((t[k:] - t[:-k]) / 1e9)
    v = np.r_[np.full(k // 2, np.nan), np.hypot(vx, vy), np.full(k - k // 2, np.nan)]
    yu = np.unwrap(yaw)
    yr = np.r_[np.full(k // 2, np.nan), (yu[k:] - yu[:-k]) / ((t[k:] - t[:-k]) / 1e9), np.full(k - k // 2, np.nan)]
    dist = np.sum(np.hypot(np.diff(x), np.diff(y)))

    # laps: s wraps from > 0.8 L to < 0.2 L while on track (or in pit, which runs parallel to the straight)
    su = np.unwrap(s / length * 2 * np.pi) / (2 * np.pi) * length
    s0 = su[0]
    lap_idx = np.floor((su - s0) / length).astype(int)
    # lap line = centreline s = 0; count a crossing only on track, moving forward (s wraps high -> low)
    wrapf = (s[:-1] > 0.9 * length) & (s[1:] < 0.1 * length) & in_track[:-1] & in_track[1:]
    cross = np.where(wrapf)[0] + 1
    cross = cross[np.r_[True, np.diff(tt[cross]) > 30]] if len(cross) else cross
    lap_times = np.diff(tt[cross])
    # pit segments (merge gaps < 5 s caused by the polygon edge)
    pit_edges = np.diff(np.r_[0, in_pit.astype(int), 0])
    p_start, p_end = np.where(pit_edges == 1)[0], np.where(pit_edges == -1)[0]
    segs = []
    for a, e in zip(p_start, p_end):
        if segs and tt[a] - segs[-1][1] < 5:
            segs[-1][1] = tt[e - 1]
        else:
            segs.append([tt[a], tt[e - 1]])
    pit_segs = [(a, b_) for a, b_ in segs if b_ - a > 2.0]
    kmax_i = int(np.argmax(np.abs(kappa)))

    se = np.load(OUT / 'state_est.npz')
    keys = list(se['keys'])
    se_t = se['t_ns']
    v_se = np.interp(t, se_t, se['data'][:, keys.index('v_mps')])
    moving = v_se > 1.0

    res = {
        'gt_duration_s': float(tt[-1]), 'gt_n': int(len(t)), 'gt_rate_hz': float((len(t) - 1) / tt[-1]),
        'gt_dt_ms_median': float(np.median(dtn) / 1e6), 'gt_dt_ms_max': float(dtn.max() / 1e6),
        'gt_gaps_over_50ms': int(np.sum(dtn > 50e6)),
        'gt_time_in_gaps_over_50ms_s': float(np.sum(dtn[dtn > 50e6]) / 1e9),
        'speed_max_mps': float(np.nanmax(v)), 'speed_mean_mps': float(np.nanmean(v)),
        'speed_mean_moving_mps': float(np.nanmean(v[moving])), 'speed_p95_mps': float(np.nanpercentile(v, 95)),
        'v_mps_field_max': float(v_se.max()),
        'speed_gt_vs_vmps_rmse': float(np.sqrt(np.nanmean((v - v_se) ** 2))),
        'yaw_rate_abs_max_radps': float(np.nanmax(np.abs(yr))),
        'yaw_rate_abs_p99_radps': float(np.nanpercentile(np.abs(yr), 99)),
        'distance_m': float(dist), 'time_stationary_s': float(np.sum(~moving) * np.median(dtn) / 1e9),
        'track_length_centreline_m': float(length), 'track_left_len_m': float(np.sum(np.hypot(*np.diff(L, axis=0).T))),
        'track_right_len_m': float(np.sum(np.hypot(*np.diff(R, axis=0).T))),
        'width_m': {q: float(np.percentile(w_left + w_right, q)) for q in (0, 5, 50, 95, 100)},
        'width_left_median_m': float(np.median(w_left)), 'width_right_median_m': float(np.median(w_right)),
        'kappa_abs_max_1pm': float(np.abs(kappa).max()), 'min_radius_m': float(1 / np.abs(kappa).max()),
        'min_radius_at_xy_s': [float(c[kmax_i, 0]), float(c[kmax_i, 1]), float(s_c[kmax_i])],
        'kappa_abs_p99_1pm': float(np.percentile(np.abs(kappa), 99)),
        'lap_crossing_times_s': [round(float(tt[i]), 1) for i in cross],
        'fastest_lap_s': float(lap_times.min()) if len(lap_times) else None,
        'share_track_with_radius_over_500m': float(np.mean(np.abs(kappa) < 1 / 500)),
        'n_lap_crossings': int(len(cross)), 'lap_times_s': [round(float(a), 2) for a in lap_times],
        'pit_share_time': float(np.mean(in_pit)), 'pit_segments_s': [[round(a, 1), round(b_, 1)] for a, b_ in pit_segs],
        'gt_off_track_not_pit_share': float(np.mean(~in_track & ~in_pit)),
        'lateral_offset_n_on_track_abs_p95_m': float(np.percentile(np.abs(n[in_track]), 95)),
        'heading_err_vs_centreline_abs_p95_deg': float(np.degrees(np.percentile(np.abs(epsi[in_track]), 95))),
        'first_pose': [float(x[0]), float(y[0]), float(yaw[0])],
    }
    (OUT / 'track_gt.json').write_text(json.dumps(res, indent=1))
    np.savez(OUT / 'track_model.npz', c=c, s=s_c, hdg=hdg, kappa=kappa, w_left=w_left, w_right=w_right, length=length)
    np.savez(OUT / 'gt_track.npz', t=t, s=s, su=su, n=n, epsi=epsi, in_pit=in_pit, in_track=in_track,
             v=v, v_se=v_se, yr=yr, lap_idx=lap_idx, cross=cross)
    print(json.dumps(res, indent=1))

    # ---- plots
    fig, ax = plt.subplots(figsize=(11, 11))
    for p, col in ((L, 'k'), (R, 'k'), (PL, 'tab:orange'), (PR, 'tab:orange')):
        ax.plot(p[:, 0], p[:, 1], col, lw=0.6)
    sc = ax.scatter(x[::5], y[::5], c=v_se[::5], s=1.5, cmap='turbo')
    plt.colorbar(sc, ax=ax, shrink=0.6, label='speed v_mps [m/s]')
    for i in range(0, len(c), 250):
        ax.annotate(f'{s_c[i]:.0f}', c[i], fontsize=7, color='tab:blue')
    ax.plot(*c[0], 'r*', ms=12)
    ax.set_aspect('equal'); ax.set_title('GT trajectory coloured by speed; black = track borders, orange = pit lane; blue = s [m]')
    fig.tight_layout(); fig.savefig(FIG / 'traj_speed.png', dpi=110); plt.close(fig)

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axs[0].plot(tt, v_se, lw=0.6); axs[0].set_ylabel('v [m/s]')
    axs[1].plot(tt, yr, lw=0.6); axs[1].set_ylabel('yaw rate [rad/s]')
    axs[2].plot(tt, in_pit, lw=0.8, color='tab:orange'); axs[2].set_ylabel('in pit')
    axs[3].plot(tt, n, lw=0.6); axs[3].set_ylabel('n (lateral, m)'); axs[3].set_xlabel('t [s]')
    for a in axs:
        for cidx in cross:
            a.axvline(tt[cidx], color='gray', lw=0.5, ls=':')
    fig.suptitle('GT time series (dotted = lap line crossings)')
    fig.tight_layout(); fig.savefig(FIG / 'time_series.png', dpi=100); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axs[0].plot(s_c, kappa); axs[0].set_ylabel('curvature [1/m]')
    axs[1].plot(s_c, w_left, label='left'); axs[1].plot(s_c, w_right, label='right')
    axs[1].set_ylabel('half-width [m]'); axs[1].set_xlabel('s [m]'); axs[1].legend()
    fig.tight_layout(); fig.savefig(FIG / 'track_profile.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
