"""Figures for the slides and the demo video, in the Atelier style (warm paper, ink, one blue accent).

GT is used for DISPLAY only (error colouring, reference line); nothing here feeds inference.
    .venv/bin/python -W ignore submissions/media/figs.py      -> submissions/build/fig_*.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, '.')
from camloc.data import load_static
from camloc.evaluate import gt_context, lap_windows, load_sub, pose_errors
from camloc.groundtruth import load_gt
from camloc.track import TrackModel

OUT = 'submissions/build'
for f in glob.glob(f'{OUT}/fonts/*.ttf'):
    fm.fontManager.addfont(f)
PAPER, SURF, GRID, INK, INK2, INK3, LINE = '#F1ECE1', '#F9F6EE', '#E3DCCB', '#211E18', '#4E4A3F', '#686457', '#CFC5AE'
BLUE, OK, WARN, CRIT = '#2B59D1', '#1B6B33', '#855504', '#B93232'
ERR = LinearSegmentedColormap.from_list('err', [OK, '#9C6A00', CRIT])
plt.rcParams.update({'font.family': 'JetBrains Mono', 'font.size': 15, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK3, 'ytick.color': INK3, 'axes.edgecolor': LINE, 'axes.facecolor': SURF,
                     'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 1, 'axes.spines.top': False, 'axes.spines.right': False})


def style(ax):
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(LINE)


gt = load_gt()
cams, bounds = load_static()
tm = TrackModel(bounds)
lw = {n: (a, b) for n, a, b in lap_windows(gt_context(gt, tm))}


def draw_track(ax, lw_=1.2):
    for k, v in bounds.items():
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[1] >= 2:
            ax.plot(v[:, 0], v[:, 1], color=LINE if 'pit' in str(k).lower() else INK3, lw=lw_, zorder=1)


def lap_sub(path, L):
    s = load_sub(path)
    m = (s['t'] >= lw[L][0]) & (s['t'] <= lw[L][1])
    return {k: v[m] for k, v in s.items()}


def colored(ax, x, y, c, vmax=1.5, lw_=3.2):
    pts = np.c_[x, y].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1), cmap=ERR, norm=plt.Normalize(0, vmax), lw=lw_, zorder=3)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    return lc


def fig_traj(L=4):
    """Held-out lap L: estimate coloured by position error on the track, plus error vs time."""
    s = lap_sub(f'cache/sub_ail_f{L}.csv', L)
    e = pose_errors(gt, s)
    t = (e['t'] - e['t'][0]) / 1e9
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_axes([0.02, 0.04, 0.50, 0.92])
    draw_track(ax)
    lc = colored(ax, s['x'], s['y'], e['pos'])
    ax.set_aspect('equal'); ax.axis('off')
    cax = fig.add_axes([0.06, 0.08, 0.16, 0.018])
    cb = fig.colorbar(lc, cax=cax, orientation='horizontal')
    cb.set_label('position error (m)'); cb.outline.set_visible(False)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    ax2 = fig.add_axes([0.58, 0.55, 0.39, 0.36]); style(ax2)
    ax2.fill_between(t, 0, e['pos'], color=BLUE, alpha=0.12, lw=0)
    ax2.plot(t, e['pos'], color=BLUE, lw=1.6)
    ax2.axhline(1, color=CRIT, lw=1, ls=(0, (4, 3)))
    ax2.set_ylabel('position error (m)'); ax2.set_xlabel('time in lap (s)'); ax2.set_ylim(0, max(2.5, e['pos'].max() * 1.05))
    ax3 = fig.add_axes([0.58, 0.10, 0.39, 0.32]); style(ax3)
    ax3.plot(t, e['lat'], color=INK, lw=1.3, label='lateral')
    ax3.plot(t, e['lon'], color=WARN, lw=1.0, alpha=0.8, label='longitudinal')
    ax3.axhline(0, color=INK3, lw=0.8)
    ax3.set_ylabel('signed error (m)'); ax3.set_xlabel('time in lap (s)'); ax3.legend(frameon=False, loc='upper right', ncol=2)
    fig.text(0.58, 0.95, f'LAP {L} · HELD OUT · RMSE {rms(e["pos"]):.3f} m · LATERAL {rms(e["lat"]):.3f} m · '
             f'{np.mean(e["pos"] < 1) * 100:.1f} % < 1 m', fontsize=14, color=INK2)
    fig.savefig(f'{OUT}/fig_traj_lap{L}.png', dpi=100)
    plt.close(fig)


def fig_deadreckon():
    """Why not odometry alone: wheel/steer dead reckoning from the TRUE start pose vs the camera localiser."""
    o = load_sub(f'{OUT}/odom_full.csv')
    v = load_sub('cache/sub_ail_f4.csv')
    eo, ev = pose_errors(gt, o), pose_errors(gt, v)
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_axes([0.02, 0.04, 0.46, 0.92])
    ax.plot(gt['x'], gt['y'], color=INK, lw=1.0, zorder=2, label='ground truth (display only)')
    ax.plot(o['x'], o['y'], color=WARN, lw=0.9, alpha=0.85, zorder=1, label='wheel + steering odometry alone')
    ax.set_aspect('equal'); ax.axis('off'); ax.legend(frameon=False, loc='lower right', fontsize=14)
    ax2 = fig.add_axes([0.55, 0.14, 0.42, 0.72]); style(ax2)
    to, tv = (eo['t'] - eo['t'][0]) / 1e9, (ev['t'] - ev['t'][0]) / 1e9
    ax2.semilogy(to, np.maximum(eo['pos'], 1e-2), color=WARN, lw=1.6, label='odometry alone')
    ax2.semilogy(tv, np.maximum(ev['pos'], 1e-2), color=BLUE, lw=0.6, label='camloc, no start pose')
    ax2.axhline(1, color=CRIT, lw=1, ls=(0, (4, 3)))
    ax2.set_xlabel('time (s)'); ax2.set_ylabel('position error (m, log)'); ax2.set_ylim(1e-2, 2e3)
    ax2.legend(frameon=False, loc='upper left', ncol=2)
    fig.savefig(f'{OUT}/fig_deadreckon.png', dpi=100)
    plt.close(fig)
    return float(np.sqrt(np.mean(eo['pos'] ** 2))), float(eo['pos'].max())


def fig_rig():
    """Camera rig from tf_static, top view (x forward). No camera images (dataset licence)."""
    fig, ax = plt.subplots(figsize=(10, 7)); style(ax)
    ax.add_patch(plt.Rectangle((-2.2, -0.95), 5.6, 1.9, fc=SURF, ec=INK, lw=2, zorder=1))
    names = {'fl': 'front left', 'fr': 'front right', 'r': 'rear', 'cl': 'side left', 'cr': 'side right',
             'rl': 'rear left', 'rr': 'rear right'}
    for c, cam in cams.items():
        T = cam.T_base_cam
        p, z = T[:3, 3], T[:3, 2]
        yaw = np.arctan2(z[1], z[0])
        fov = 2 * np.arctan(cam.K[0, 2] / cam.K[0, 0]) if hasattr(cam, 'K') else np.radians(60)
        used = c in ('fl', 'fr', 'r')
        col = BLUE if used else INK3
        r = 3.6
        a = np.linspace(yaw - fov / 2, yaw + fov / 2, 30)
        ax.fill(np.r_[p[0], p[0] + r * np.cos(a)], np.r_[p[1], p[1] + r * np.sin(a)], color=col, alpha=0.13 if used else 0.06, lw=0)
        ax.plot(p[0], p[1], 'o', color=col, ms=9, zorder=3)
        ax.text(p[0] + 4.5 * np.cos(yaw), p[1] + 4.5 * np.sin(yaw), f'{c}\n{names.get(c, c)}', ha='center', va='center',
                fontsize=12, color=col, fontweight='bold' if used else 'normal')
    ax.set_aspect('equal'); ax.set_xlim(-8, 9.5); ax.set_ylim(-6.2, 6.2); ax.axis('off')
    fig.savefig(f'{OUT}/fig_rig.png', dpi=110, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def fig_progression():
    """Pooled held-out RMSE (laps 2/4/6, map rebuilt without the scored lap) by development step (experiment log)."""
    steps = [('map fixes', 1.06, 'EXP-0015'), ('weak single-cam\nfixes', 0.81, 'EXP-0015'), ('CLAHE map', 0.673, 'EXP-0020'),
             ('quality-weighted\nfixes', 0.630, 'EXP-0022'), ('sideslip', 0.586, 'EXP-0025'),
             ('phase odometry\n+ strict calib.', 0.513, 'EXP-0028')]
    fig, ax = plt.subplots(figsize=(12, 6.2)); style(ax)
    x = np.arange(len(steps)); y = [s[1] for s in steps]
    ax.plot(x, y, color=BLUE, lw=2.5, marker='o', ms=11, mfc=SURF, mew=2.5)
    for i, (n, v, e) in enumerate(steps):
        ax.text(i, v + 0.045, f'{v:.2f}', ha='center', fontsize=17, color=INK, fontweight='bold')
    ax.set_xticks(x, [s[0] for s in steps], fontsize=12)
    ax.set_ylim(0.3, 1.2); ax.set_ylabel('pooled position RMSE (m)')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_progression.png', dpi=110, transparent=True)
    plt.close(fig)


def fig_midlap():
    """16 starts with no start pose: position error vs time since the first frame."""
    rows = {r['start']: r for r in json.load(open('cache/midlap_results.json'))}
    fig, ax = plt.subplots(figsize=(12, 6.2)); style(ax)
    for f in sorted(glob.glob('cache/sub_ml_*.csv')):
        tag = f.split('ml_')[1][:-4]
        e = pose_errors(gt, load_sub(f))
        t = (e['t'] - e['t'][0]) / 1e9
        hi = tag in ('lap6@0.7', 'lap2@0.9')
        ax.semilogy(t, np.maximum(e['pos'], 0.02), color=(CRIT if tag == 'lap6@0.7' else WARN) if hi else BLUE,
                    lw=2.2 if hi else 0.9, alpha=1 if hi else 0.35, zorder=3 if hi else 2,
                    label={'lap6@0.7': 'lap 6 @ 70 %: aliased lock 13.6 m off, switch after 0.3 s',
                           'lap2@0.9': 'lap 2 @ 90 %: lock 9.9 m off, converges by 11 s'}.get(tag))
    ax.plot([], [], color=BLUE, alpha=0.5, lw=1, label='the other 14 starts')
    ax.axhline(1, color=CRIT, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, 40); ax.set_ylim(0.02, 30)
    ax.set_xlabel('time since the first frame (s)'); ax.set_ylabel('position error (m, log)')
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_midlap.png', dpi=110, transparent=True)
    plt.close(fig)
    lt = [r['lock_t_s'] for r in rows.values()]
    return min(lt), max(lt)


if __name__ == '__main__' and 'anim' not in sys.argv[0]:
    fig_traj(4)
    fig_rig()
    fig_progression()
    print('midlap lock range', fig_midlap())
    if os.path.exists(f'{OUT}/odom_full.csv'):
        print('odometry alone: rmse, max', fig_deadreckon())
