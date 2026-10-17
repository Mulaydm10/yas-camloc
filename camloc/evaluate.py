"""Evaluation harness: the official metric plus breakdowns. EVALUATION ONLY (uses ground truth).

    python -m camloc.evaluate sub.csv                    # whole run
    python -m camloc.evaluate sub.csv --laps 4           # validation laps (1-based full laps)
    python -m camloc.evaluate sub.csv --t0 600 --t1 700  # time window [s from first GT stamp]
    python -m camloc.evaluate sub.csv --official         # also run tools/evaluate.py on the same window

The error computation mirrors tools/evaluate.py exactly (linear GT interpolation at the submitted stamp,
yaw on the unwrapped GT angle, stamps with a GT gap > 50 ms skipped, lateral/longitudinal in the GT car
frame, coverage = share of 1 s bins from the first to the last GT stamp holding a pose).
Breakdowns: GT quality (estimator pos_accuracy <= 1 m vs blip), lane (track / pit), 10 track sectors,
GT speed bins, laps.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import config as C
from .groundtruth import load_gt, write_gt_csv

MAX_GT_GAP_NS = 50_000_000
BIN_NS = 1_000_000_000
BLIP_ACC_M = 1.0
N_SECTORS = 10
SPEED_BINS = (0, 10, 20, 30, 50)
OFFICIAL = C.ROOT / 'data' / 'c0_2026-10-13_22-53-42_camloc' / 'tools' / 'evaluate.py'


def load_sub(path):
    d = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    t = np.atleast_1d(np.asarray(d['timestamp_ns'], dtype=np.int64))
    o = np.argsort(t)
    return {'t': t[o], 'x': np.atleast_1d(d['x']).astype(float)[o], 'y': np.atleast_1d(d['y']).astype(float)[o],
            'yaw': np.atleast_1d(d['yaw']).astype(float)[o]}


def pose_errors(gt, sub):
    """Per submitted pose errors, identical to tools/evaluate.py. Returns dict of arrays (evaluated poses only)."""
    tg, ts = gt['t'], sub['t']
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    ts, i = ts[ok], i[ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yu = np.unwrap(gt['yaw'])
    lerp = lambda v: v[i - 1] + a * (v[i] - v[i - 1])
    gx, gy, gyaw = lerp(gt['x']), lerp(gt['y']), lerp(yu)
    dx, dy = sub['x'][ok] - gx, sub['y'][ok] - gy
    return {
        't': ts, 'n_submitted': len(sub['t']), 'gx': gx, 'gy': gy, 'gyaw': gyaw,
        'lon': np.cos(gyaw) * dx + np.sin(gyaw) * dy, 'lat': -np.sin(gyaw) * dx + np.cos(gyaw) * dy,
        'pos': np.hypot(dx, dy), 'dyaw': np.degrees(np.angle(np.exp(1j * (sub['yaw'][ok] - gyaw)))),
        'pos_acc': lerp(gt['pos_acc']),
    }


def coverage(t_gt0, t_gt1, ts):
    bins = np.arange(t_gt0, t_gt1, BIN_NS)
    covered = np.unique(np.searchsorted(bins, ts, side='right') - 1)
    covered = covered[(covered >= 0) & (covered < len(bins))]
    return len(covered) / max(len(bins), 1)


def metrics(e, m=None):
    m = np.ones(len(e['pos']), bool) if m is None else m
    n = int(m.sum())
    if n == 0:
        return {'n': 0}
    rms = lambda v: float(np.sqrt(np.mean(v[m] ** 2)))
    return {'n': n, 'pos_rmse': rms(e['pos']), 'lat_rmse': rms(e['lat']), 'lon_rmse': rms(e['lon']),
            'yaw_rmse_deg': rms(e['dyaw']), 'pos_median': float(np.median(e['pos'][m])),
            'pos_p95': float(np.percentile(e['pos'][m], 95)),
            'within_0.5m': float(np.mean(e['pos'][m] < 0.5)), 'within_1m': float(np.mean(e['pos'][m] < 1.0))}


def gt_context(gt, track):
    """Per-GT-sample context: lane, track s, speed, lap index (cached)."""
    path = C.CACHE / 'gt_context.npz'
    key = np.array([len(gt['t']), round(track.track.length, 3), round(track.pit.length, 3)])
    if path.exists():
        z = np.load(path)
        if 'key' in z.files and np.array_equal(z['key'], key):
            return {k: z[k] for k in z.files}
    lane, s, n = track.locate(gt['x'], gt['y'])
    st, _ = track.track.project(gt['x'], gt['y'])
    t = gt['t']
    k = 10
    v = np.full(len(t), np.nan)
    v[k // 2:len(t) - k + k // 2] = np.hypot(gt['x'][k:] - gt['x'][:-k], gt['y'][k:] - gt['y'][:-k]) / ((t[k:] - t[:-k]) / 1e9)
    v = np.where(np.isnan(v), np.nanmedian(v[:k]) if np.any(~np.isnan(v[:k])) else 0.0, v)
    on = lane == 'track'
    L = track.track.length
    wrap_f = (st[:-1] > 0.9 * L) & (st[1:] < 0.1 * L) & on[:-1] & on[1:]
    cross = np.where(wrap_f)[0] + 1
    if len(cross):
        cross = cross[np.r_[True, np.diff(t[cross]) > 30e9]]
    lap = np.searchsorted(cross, np.arange(len(t)), side='right')   # 0 = before first crossing
    out = {'t': t, 'lane_pit': lane == 'pit', 's_track': st, 'v': v, 'lap': lap, 'cross_t': t[cross], 'key': key}
    C.CACHE.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


def lap_windows(ctx):
    """Full laps as (lap number 1..N, t_start_ns, t_end_ns)."""
    ct = ctx['cross_t']
    return [(i + 1, int(ct[i]), int(ct[i + 1])) for i in range(len(ct) - 1)]


def evaluate(sub, gt, track, windows=None, run_official=False):
    """windows: list of (t0_ns, t1_ns) or None for the whole run."""
    ctx = gt_context(gt, track)
    if windows:
        mg = np.zeros(len(gt['t']), bool)
        ms = np.zeros(len(sub['t']), bool)
        for a, b in windows:
            mg |= (gt['t'] >= a) & (gt['t'] <= b)
            ms |= (sub['t'] >= a) & (sub['t'] <= b)
        gtw = {k: v[mg] for k, v in gt.items()}
        subw = {k: v[ms] for k, v in sub.items()}
    else:
        gtw, subw = gt, sub
    e = pose_errors(gtw, subw)
    if windows:
        cov = np.mean([coverage(a, b, e['t']) for a, b in windows])
    else:
        cov = coverage(gtw['t'][0], gtw['t'][-1], e['t'])
    j = np.clip(np.searchsorted(ctx['t'], e['t']), 0, len(ctx['t']) - 1)
    pit, s, v, lap = ctx['lane_pit'][j], ctx['s_track'][j], ctx['v'][j], ctx['lap'][j]
    good = e['pos_acc'] <= BLIP_ACC_M
    L = track.track.length
    sec = np.minimum((s / L * N_SECTORS).astype(int), N_SECTORS - 1)
    res = {
        'window': windows, 'n_submitted': e['n_submitted'], 'n_evaluated': len(e['t']), 'coverage': float(cov),
        'ranked': bool(cov >= 0.9),
        'all': metrics(e), 'gt_good': metrics(e, good), 'gt_blip': metrics(e, ~good),
        'lane': {'track': metrics(e, ~pit & good), 'pit': metrics(e, pit & good)},
        'sector': {f'S{k + 1}': metrics(e, ~pit & good & (sec == k)) for k in range(N_SECTORS)},
        'speed': {f'{a}-{b} m/s': metrics(e, good & (v >= a) & (v < b)) for a, b in zip(SPEED_BINS[:-1], SPEED_BINS[1:])},
        'lap': {str(int(k)): metrics(e, good & (lap == k)) for k in np.unique(lap)},
    }
    if run_official:
        res['official'] = official(gtw, subw)
    return res


def official(gt, sub):
    with tempfile.TemporaryDirectory() as d:
        g, s = Path(d) / 'gt.csv', Path(d) / 'sub.csv'
        write_gt_csv(gt, g)
        with open(s, 'w') as f:
            f.write('timestamp_ns,x,y,yaw\n')
            for a, b, c, y in zip(sub['t'], sub['x'], sub['y'], sub['yaw']):
                f.write(f'{int(a)},{b:.4f},{c:.4f},{y:.6f}\n')
        return subprocess.run([sys.executable, str(OFFICIAL), str(g), str(s)], capture_output=True, text=True).stdout


def fmt(res):
    def row(name, m):
        if not m.get('n'):
            return f'| {name} | 0 | | | | | | |'
        return (f"| {name} | {m['n']} | {m['pos_rmse']:.3f} | {m['lat_rmse']:.3f} | {m['lon_rmse']:.3f} | "
                f"{m['yaw_rmse_deg']:.2f} | {m['pos_p95']:.2f} | {m['within_1m'] * 100:.1f} |")
    out = [f"coverage {res['coverage'] * 100:.1f} % ({'ranked' if res['ranked'] else 'NOT RANKED'}), "
           f"evaluated {res['n_evaluated']} / {res['n_submitted']} poses",
           '', '| group | n | pos RMSE | lat RMSE | lon RMSE | yaw RMSE ° | p95 | ≤1 m % |', '|---|---|---|---|---|---|---|---|',
           row('**all** (official)', res['all']), row('GT good (acc ≤ 1 m)', res['gt_good']), row('GT blip', res['gt_blip'])]
    for grp in ('lane', 'sector', 'speed', 'lap'):
        for k, m in res[grp].items():
            out.append(row(f'{grp} {k}', m))
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('sub')
    ap.add_argument('--mcap', default=str(C.DEFAULT_MCAP))
    ap.add_argument('--laps', type=int, nargs='*')
    ap.add_argument('--t0', type=float)
    ap.add_argument('--t1', type=float)
    ap.add_argument('--official', action='store_true')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    from .data import load_static
    from .track import TrackModel
    gt = load_gt(a.mcap)
    _, bounds = load_static(a.mcap, cams=())
    track = TrackModel(bounds)
    windows = None
    if a.laps:
        lw = {n: (t0, t1) for n, t0, t1 in lap_windows(gt_context(gt, track))}
        windows = [lw[n] for n in a.laps]
    elif a.t0 is not None or a.t1 is not None:
        t0 = gt['t'][0] + int((a.t0 or 0) * 1e9)
        t1 = gt['t'][-1] if a.t1 is None else gt['t'][0] + int(a.t1 * 1e9)
        windows = [(t0, t1)]
    res = evaluate(load_sub(a.sub), gt, track, windows, a.official)
    print(fmt(res))
    if a.official:
        print('\n--- tools/evaluate.py on the same window ---\n' + res['official'])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
