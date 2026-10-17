"""Camera analysis (step 2.1): timing, sync vs fl, latency, brightness, sharpness, ego-body mask,
calibration-error budget. Reads analysis/out/{meta.json,cam_stats.npz,pixstats_*.npz}.

Writes analysis/out/cameras.json, analysis/body_mask.png, analysis/cam_stats_time.png
"""
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import CAMS, FIG, OUT


def main():
    meta = json.loads((OUT / 'meta.json').read_text())
    z = np.load(OUT / 'cam_stats.npz')
    F = list(z['fields'])
    col = {f: i for i, f in enumerate(F)}
    gtt = np.load(OUT / 'gt_track.npz')
    v_t, v = gtt['t'], gtt['v_se']
    ref = z['fl'][:, col['stamp_ns']].astype(np.int64)
    res = {}
    for c in CAMS:
        a = z[c]
        st = a[:, col['stamp_ns']].astype(np.int64)
        lg = a[:, col['log_ns']].astype(np.int64)
        d = np.diff(st) / 1e6
        j = np.clip(np.searchsorted(ref, st), 1, len(ref) - 1)
        near = np.where(np.abs(ref[j] - st) < np.abs(ref[j - 1] - st), ref[j], ref[j - 1])
        sync = (st - near) / 1e6
        lat = (lg - st) / 1e6
        k = meta['cameras'][c]['intrinsics']
        e = meta['cameras'][c]['extrinsics']
        spd = np.interp(st, v_t, v)
        lv = a[:, col['lapvar']]
        fast, slow = spd > 30, (spd > 3) & (spd < 15)
        res[c] = {
            'fx': k['K'][0], 'fy': k['K'][4], 'cx': k['K'][2], 'cy': k['K'][5], 'D': k['D'],
            'hfov_deg': float(np.degrees(2 * np.arctan(k['width'] / 2 / k['K'][0]))),
            'mount_xyz_m': e['t'], 'view_yaw_deg': e['view_yaw_deg'], 'view_pitch_down_deg': e['view_pitch_deg'],
            'n_frames': int(len(st)), 'period_ms_median': float(np.median(d)), 'period_ms_max': float(d.max()),
            'gaps_over_150ms': int(np.sum(d > 150)), 'missing_frames_est': int(np.sum(np.round(d / 100) - 1)),
            'non_monotonic_stamps': int(np.sum(d <= 0)),
            'sync_vs_fl_ms_median_abs': float(np.median(np.abs(sync))), 'sync_vs_fl_ms_max_abs': float(np.abs(sync).max()),
            'latency_log_minus_stamp_ms_median': float(np.median(lat)), 'latency_ms_p99': float(np.percentile(lat, 99)),
            'latency_ms_max': float(lat.max()),
            'brightness_mean_p5_p50_p95': [float(np.percentile(a[:, col['mean']], q)) for q in (5, 50, 95)],
            'p99_brightness_median': float(np.median(a[:, col['p99']])),
            'lapvar_p5_p50_p95': [float(np.percentile(lv, q)) for q in (5, 50, 95)],
            'lapvar_median_slow_vs_fast': [float(np.median(lv[slow])), float(np.median(lv[fast]))],
            'orb_median': float(np.median(a[:, col['orb']])), 'edge_low_median': float(np.median(a[:, col['edge_low']])),
            'jpeg_kb_median': float(np.median(a[:, col['jpeg_bytes']]) / 1e3),
        }
        # motion blur (pixels) at 40 m/s for a point 10 m to the side, 1 ms exposure, is unknown (no exposure
        # time in the data); we only report the speed dependence of sharpness above.
    # yaw error budget: lateral offset of a point at range r for 1 deg yaw error
    res['yaw_1deg_offset_m'] = {r: float(r * np.tan(np.radians(1))) for r in (10, 30, 60)}

    # ego-body mask: pixels with very low temporal std (car body, fixed) at 1/2 resolution
    fig, axs = plt.subplots(4, 4, figsize=(18, 10))
    for i, c in enumerate(CAMS):
        p = np.load(OUT / f'pixstats_{c}.npz')
        std, mean = p['std'], p['mean']
        mask = (cv2.GaussianBlur(std, (0, 0), 3) < 12).astype(np.uint8)
        n_, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = np.zeros_like(mask)
        h, w = mask.shape
        for l_ in range(1, n_):            # keep large components touching the image border = body
            x0, y0, ww, hh, area = stats[l_]
            if area > 0.01 * h * w and (y0 + hh >= h - 2 or x0 == 0 or x0 + ww >= w - 2):
                keep[lab == l_] = 1
        res[c]['body_mask_share'] = float(keep.mean())
        rows = np.where(keep.any(axis=1))[0]
        res[c]['body_mask_rows_full_res'] = [int(rows.min() * 2), int(rows.max() * 2)] if len(rows) else None
        cv2.imwrite(str(OUT / f'body_mask_{c}.png'), keep * 255)
        axs[i // 2, (i % 2) * 2].imshow(mean, cmap='gray'); axs[i // 2, (i % 2) * 2].set_title(f'{c} mean')
        ov = np.dstack([mean / 255] * 3); ov[keep > 0] = 0.5 * ov[keep > 0] + [0.5, 0, 0]
        axs[i // 2, (i % 2) * 2 + 1].imshow(np.clip(ov, 0, 1)); axs[i // 2, (i % 2) * 2 + 1].set_title(f'{c} static mask {keep.mean() * 100:.1f}%')
    for a in axs.ravel():
        a.axis('off')
    fig.tight_layout(); fig.savefig(FIG / 'body_mask.png', dpi=80); plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    t0 = ref[0]
    for c in CAMS:
        a = z[c]
        tt = (a[:, col['stamp_ns']] - t0) / 1e9
        axs[0].plot(tt, a[:, col['mean']], lw=0.5, label=c)
        axs[1].plot(tt, a[:, col['lapvar']], lw=0.5)
        axs[2].plot(tt, a[:, col['orb']], lw=0.5)
    axs[0].set_ylabel('mean brightness'); axs[0].legend(ncol=7, fontsize=8)
    axs[1].set_ylabel('Laplacian var'); axs[1].set_yscale('log'); axs[2].set_ylabel('ORB kp (max 500)')
    axs[2].set_xlabel('t [s]')
    fig.tight_layout(); fig.savefig(FIG / 'cam_stats_time.png', dpi=90); plt.close(fig)

    (OUT / 'cameras.json').write_text(json.dumps(res, indent=1))
    for c in CAMS:
        r = res[c]
        print(f"{c:2s} fx {r['fx']:6.0f} hfov {r['hfov_deg']:5.1f} yaw {r['view_yaw_deg']:7.1f} pitch {r['view_pitch_down_deg']:5.1f} "
              f"n {r['n_frames']} per {r['period_ms_median']:.1f}/{r['period_ms_max']:.0f} miss {r['missing_frames_est']} "
              f"sync {r['sync_vs_fl_ms_median_abs']:.2f}/{r['sync_vs_fl_ms_max_abs']:.1f} lat {r['latency_log_minus_stamp_ms_median']:.1f}/{r['latency_ms_max']:.0f} "
              f"bright {[round(q) for q in r['brightness_mean_p5_p50_p95']]} lap {[round(q) for q in r['lapvar_p5_p50_p95']]} "
              f"slow/fast {[round(q) for q in r['lapvar_median_slow_vs_fast']]} orb {r['orb_median']:.0f} edge {r['edge_low_median']:.3f} "
              f"mask {r['body_mask_share'] * 100:.1f}% rows {r['body_mask_rows_full_res']}")
    print(res['yaw_1deg_offset_m'])


if __name__ == '__main__':
    main()
