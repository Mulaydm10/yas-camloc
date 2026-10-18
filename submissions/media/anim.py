"""Animations for the demo video (Atelier style, 1920x1080, 60 fps), piped to ffmpeg.
GT is drawn for DISPLAY only; the estimates come from causal runs that never read it.

    .venv/bin/python -W ignore submissions/media/anim.py lap4     -> build/out/traj_lap4.mp4
    .venv/bin/python -W ignore submissions/media/anim.py nostart  -> build/out/nostart.mp4
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, 'submissions/media')
sys.path.insert(0, '.')
import figs as F  # noqa: E402  (fonts, palette, GT, track)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

OUT = 'submissions/build/out'
W, H, FPS = 1920, 1080, 60


def writer(path):
    return subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}',
                             '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
                             '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)


def gt_at(t):
    return np.interp(t, F.gt['t'], F.gt['x']), np.interp(t, F.gt['t'], F.gt['y'])


def banner(fig, left, right):
    fig.patches.append(plt.Rectangle((0, 1 - 56 / H), 1, 56 / H, transform=fig.transFigure, color=F.INK, zorder=10))
    fig.text(64 / W, 1 - 28 / H, left, color='#F1ECE1', fontsize=15, fontweight='bold', va='center', zorder=11)
    fig.text(1 - 64 / W, 1 - 28 / H, right, color='#B9B19E', fontsize=15, va='center', ha='right', zorder=11)


def lap4(speed=4.0):
    s = F.lap_sub('submissions/build/lap4.csv', 4)
    e = F.pose_errors(F.gt, s)
    t = (s['t'] - s['t'][0]) / 1e9
    gx, gy = gt_at(s['t'])
    n_frames = int(t[-1] / speed * FPS) + FPS * 2
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=F.PAPER)
    banner(fig, '●  CAMLOC · HELD-OUT LAP 4 · REPLAY ×4', 'MAP + CALIBRATION WITHOUT LAP 4 · NO START POSE · GT FOR DISPLAY ONLY')
    ax = fig.add_axes([0.02, 0.05, 0.42, 0.86], facecolor=F.PAPER)
    F.draw_track(ax, 1.0)
    ax.set_aspect('equal'); ax.axis('off')
    trail = LineCollection([], cmap=F.ERR, norm=plt.Normalize(0, 1.5), lw=3.4, zorder=3)
    ax.add_collection(trail)
    car, = ax.plot([], [], 'o', color=F.BLUE, ms=13, mec=F.SURF, mew=2.5, zorder=5)
    az = fig.add_axes([0.47, 0.36, 0.25, 0.55], facecolor=F.SURF)
    F.style(az); az.set_xticks([]); az.set_yticks([])
    F.draw_track(az, 2.2)
    zt, = az.plot([], [], color=F.BLUE, lw=2.5, alpha=0.6, zorder=3)
    zg, = az.plot([], [], 'o', ms=26, mfc='none', mec=F.INK, mew=2, zorder=4, label='ground truth (display)')
    ze, = az.plot([], [], 'o', ms=12, color=F.BLUE, zorder=5, label='camloc estimate')
    az.set_aspect('equal'); az.legend(loc='lower left', frameon=False, fontsize=12)
    fig.text(0.47, 0.925, 'ZOOM · 40 m WINDOW', fontsize=13, color=F.INK3)
    ae = fig.add_axes([0.47, 0.08, 0.50, 0.21], facecolor=F.SURF)
    F.style(ae); ae.set_xlim(0, t[-1]); ae.set_ylim(0, 2.2); ae.axhline(1, color=F.CRIT, lw=1, ls=(0, (4, 3)))
    ae.set_xlabel('time in lap (s)', fontsize=13); ae.set_ylabel('error (m)', fontsize=13)
    el, = ae.plot([], [], color=F.BLUE, lw=1.4)
    stats = {k: fig.text(0.76, y, '', fontsize=s_, color=c, family=fam) for k, y, s_, c, fam in
             [('t', 0.86, 15, F.INK3, 'JetBrains Mono'), ('err', 0.74, 64, F.INK, 'Instrument Serif'),
              ('errl', 0.705, 14, F.INK3, 'JetBrains Mono'), ('rms', 0.58, 64, F.BLUE, 'Instrument Serif'),
              ('rmsl', 0.545, 14, F.INK3, 'JetBrains Mono'), ('in1', 0.42, 64, F.OK, 'Instrument Serif'),
              ('in1l', 0.385, 14, F.INK3, 'JetBrains Mono')]}
    pts = np.c_[s['x'], s['y']].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], 1)
    w = writer(f'{OUT}/traj_lap4.mp4')
    for k in range(n_frames):
        tc = min(k / FPS * speed, t[-1])
        i = int(np.searchsorted(t, tc, 'right')) - 1
        i = max(i, 1)
        trail.set_segments(segs[:i]); trail.set_array(e['pos'][:i])
        car.set_data([s['x'][i]], [s['y'][i]])
        az.set_xlim(gx[i] - 20 * 1.0, gx[i] + 20); az.set_ylim(gy[i] - 20 * 1.2, gy[i] + 20 * 1.2)
        j0 = max(0, i - 40)
        zt.set_data(s['x'][j0:i + 1], s['y'][j0:i + 1])
        zg.set_data([gx[i]], [gy[i]]); ze.set_data([s['x'][i]], [s['y'][i]])
        el.set_data(t[:i + 1], e['pos'][:i + 1])
        rms = float(np.sqrt(np.mean(e['pos'][:i + 1] ** 2)))
        stats['t'].set_text(f'LAP TIME {tc:5.1f} s')
        stats['err'].set_text(f'{e["pos"][i]:.2f} m'); stats['errl'].set_text('POSITION ERROR NOW')
        stats['rms'].set_text(f'{rms:.3f} m'); stats['rmsl'].set_text('RMSE SO FAR')
        stats['in1'].set_text(f'{np.mean(e["pos"][:i + 1] < 1) * 100:.1f} %'); stats['in1l'].set_text('WITHIN 1 m SO FAR')
        fig.canvas.draw()
        w.stdin.write(fig.canvas.buffer_rgba().tobytes())
    w.stdin.close(); w.wait()
    print('lap4 frames', n_frames, 'final rmse', rms)


def nostart(tag='lap6@0.7', t_show=8.0):
    s = F.load_sub(f'cache/sub_ml_{tag}.csv')
    e = F.pose_errors(F.gt, s)
    t = (s['t'] - s['t'][0]) / 1e9
    gx, gy = gt_at(s['t'])
    # switch: the first frame where the error drops below 3 m after the aliased lock
    i_sw = int(np.argmax(e['pos'] < 3.0))
    t_sw = t[i_sw]
    # time map: hold the lock 6 s, play 0-0.8 s at x0.1 (8 s), then 0.8-t_show at x1
    tcs = np.r_[np.zeros(6 * FPS), np.linspace(0, 0.8, 8 * FPS, endpoint=False), np.linspace(0.8, t_show, int((t_show - 0.8) * FPS))]
    n_frames = len(tcs)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=F.PAPER)
    banner(fig, '●  CAMLOC · NO START POSE · LAP 6 @ 70 %', 'HELD-OUT LAP 6 · GT FOR DISPLAY ONLY')
    az = fig.add_axes([0.04, 0.06, 0.52, 0.86], facecolor=F.SURF)
    F.style(az); az.set_xticks([]); az.set_yticks([])
    F.draw_track(az, 2.6)
    az.set_aspect('equal')
    gtr, = az.plot([], [], color=F.INK, lw=1.5, alpha=0.5, zorder=2)
    etr, = az.plot([], [], color=F.BLUE, lw=3, zorder=3)
    zg, = az.plot([], [], 'o', ms=34, mfc='none', mec=F.INK, mew=2.5, zorder=4, label='ground truth (display)')
    ze, = az.plot([], [], 'o', ms=15, color=F.BLUE, zorder=5, label='camloc estimate')
    link, = az.plot([], [], color=F.CRIT, lw=1.5, ls=(0, (3, 3)), zorder=3)
    az.legend(loc='lower left', frameon=False, fontsize=14)
    head = fig.text(0.60, 0.80, '', fontsize=50, family='Instrument Serif', color=F.INK)
    sub = fig.text(0.60, 0.72, '', fontsize=19, family='Source Serif 4', color=F.INK2, wrap=True)
    big = fig.text(0.60, 0.50, '', fontsize=96, family='Instrument Serif', color=F.INK)
    bigl = fig.text(0.60, 0.46, 'POSITION ERROR', fontsize=14, color=F.INK3)
    clk = fig.text(0.60, 0.34, '', fontsize=18, color=F.INK3)
    steps = fig.text(0.60, 0.12, '', fontsize=17, color=F.INK2, linespacing=1.9)
    cx, cy = gx[min(len(gx) - 1, i_sw + 30)], gy[min(len(gy) - 1, i_sw + 30)]
    w = writer(f'{OUT}/nostart.mp4')
    for k in range(n_frames):
        tc = tcs[k]
        i = int(np.searchsorted(t, tc, 'right')) - 1
        i = max(i, 0)
        az.set_xlim(gx[i] - 30, gx[i] + 30); az.set_ylim(gy[i] - 26, gy[i] + 26)
        gtr.set_data(gx[:i + 1], gy[:i + 1]); etr.set_data(s['x'][:i + 1], s['y'][:i + 1])
        zg.set_data([gx[i]], [gy[i]]); ze.set_data([s['x'][i]], [s['y'][i]])
        link.set_data([gx[i], s['x'][i]], [gy[i], s['y'][i]])
        if tc < t_sw:
            head.set_text('Provisional lock'); head.set_color(F.WARN)
            sub.set_text('Two cameras agree on a look-alike place\n13.6 m along the track. Uncertainty 3 m: not trusted yet.')
            big.set_color(F.CRIT)
        elif tc < t_sw + 2.5:
            head.set_text('Switch'); head.set_color(F.BLUE)
            sub.set_text('Three rejected fixes agree with each other,\nfar from the lock. The filter jumps to them.')
            big.set_color(F.BLUE)
        else:
            head.set_text('Confirmed'); head.set_color(F.OK)
            sub.set_text('Two accepted map fixes confirm the new lock.\nTracking continues at map-fix accuracy.')
            big.set_color(F.OK)
        big.set_text(f'{e["pos"][i]:.2f} m')
        clk.set_text(f't = {tc:4.2f} s after the first frame' + ('  · ×0.1' if 0 < tc < 0.8 else ''))
        steps.set_text(f'■  0.0 s  lock (2 cameras agree)\n'
                       f'{"■" if tc >= t_sw else "□"}  {t_sw:.1f} s  switch (3 consistent rejected fixes)\n'
                       f'{"■" if tc >= t_sw + 2.5 else "□"}  confirm (2 accepted fixes)')
        fig.canvas.draw()
        w.stdin.write(fig.canvas.buffer_rgba().tobytes())
    w.stdin.close(); w.wait()
    print('nostart frames', n_frames, 'switch at', t_sw, 'err before', e['pos'][0], 'after', e['pos'][i_sw])


if __name__ == '__main__':
    {'lap4': lap4, 'nostart': nostart}[sys.argv[1]]()
