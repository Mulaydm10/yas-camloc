"""7-camera montages at chosen sector midpoints on the fastest lap (visual inspection, step 2.6/2.7).

    python sector_frames.py            # sectors 2, 4, 7, 9, 10 by default
Writes analysis/montage_S<k>.png
"""
import json
import sys

import cv2
import numpy as np

from common import CAMS, FIG, IMG_TOPIC, MCAP, OUT, reader, stamp_ns


def main(sectors):
    tg = json.loads((OUT / 'track_gt.json').read_text())
    g = np.load(OUT / 'gt_track.npz')
    sj = json.loads((OUT / 'sectors.json').read_text())
    cross = tg['lap_crossing_times_s']
    lt = np.diff(cross)
    k = int(np.argmin(lt))
    t0 = g['t'][0]
    a, b = t0 + int(cross[k] * 1e9), t0 + int(cross[k + 1] * 1e9)
    m = (g['t'] >= a) & (g['t'] < b)
    for sec in sectors:
        r = sj['sectors'][sec - 1]
        smid = 0.5 * (r['s_from_m'] + r['s_to_m'])
        i = np.argmin(np.abs(np.where(m, g['s'], 1e9) - smid))
        tq = int(g['t'][i])
        imgs = {}
        with open(MCAP, 'rb') as f:
            for _, ch, _, msg in reader(f).iter_decoded_messages(
                    topics=[IMG_TOPIC.format(c) for c in CAMS], start_time=tq - 200_000_000, end_time=tq + 200_000_000):
                c = ch.topic.split('camera_')[1].split('/')[0]
                if c not in imgs and stamp_ns(msg.header) >= tq:
                    imgs[c] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        tiles = []
        for c in CAMS:
            im = cv2.resize(imgs[c], (753, 364))
            cv2.putText(im, f'camera_{c}', (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            tiles.append(im)
        info = np.zeros_like(tiles[0])
        for li, txt in enumerate([f'sector S{sec}', f's = {g["s"][i]:.0f} m', f'v = {g["v_se"][i]:.1f} m/s',
                                  f't = {(tq - t0) / 1e9:.1f} s (fastest lap)']):
            cv2.putText(info, txt, (40, 80 + 70 * li), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        tiles.append(info)
        grid = np.vstack([np.hstack(tiles[j:j + 2]) for j in range(0, 8, 2)])
        cv2.imwrite(str(FIG / f'montage_S{sec}.png'), grid)
        print(f'S{sec}: t={(tq - t0) / 1e9:.1f}s s={g["s"][i]:.0f} v={g["v_se"][i]:.1f}')


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [2, 4, 7, 9, 10])
