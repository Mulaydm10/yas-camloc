"""Per-frame image statistics for all 7 cameras (one process per camera, streaming).

Every frame is decoded at 1/2 resolution, grayscale (IMREAD_REDUCED_GRAYSCALE_2, 753x364), and we record:
  header stamp, log time, jpeg bytes, mean brightness, p99 brightness, Laplacian variance (sharpness),
  Canny edge density in the lower half (road), ORB keypoint count (max 500).
Also accumulates per-pixel mean/std over every 10th frame (to find the car's own body: low std).
Full-colour montages of all 7 cameras at t = 60, 330, 600, 900 s after the first image.

Writes analysis/out/cam_stats.npz, analysis/out/pixstats_<cam>.npz, analysis/montage_<t>s.png
"""
import time
from multiprocessing import Pool

import cv2
import numpy as np

from common import CAMS, FIG, IMG_TOPIC, MCAP, OUT, reader, stamp_ns

MONTAGE_T = [60, 330, 600, 900]
FIELDS = ['stamp_ns', 'log_ns', 'jpeg_bytes', 'mean', 'p99', 'lapvar', 'edge_low', 'orb']


def per_camera(cam):
    t0 = time.time()
    t_first = int(np.load(OUT / 'image_index.npz')[cam][0, 0])
    targets = {t_first + int(s * 1e9): s for s in MONTAGE_T}
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=20)
    rows, acc, acc2, n_acc = [], None, None, 0
    full = {}
    with open(MCAP, 'rb') as f:
        for k, (_, _, m, msg) in enumerate(reader(f).iter_decoded_messages(topics=[IMG_TOPIC.format(cam)])):
            st = stamp_ns(msg.header)
            buf = np.frombuffer(bytes(msg.data), np.uint8)
            g = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_2)
            lap = cv2.Laplacian(g, cv2.CV_32F).var()
            low = g[g.shape[0] // 2:]
            edges = cv2.Canny(low, 50, 150).mean() / 255
            kp = orb.detect(g, None)
            rows.append((st, m.log_time, len(buf), g.mean(), np.percentile(g, 99), lap, edges, len(kp)))
            if k % 10 == 0:
                gf = g.astype(np.float64)
                acc = gf if acc is None else acc + gf
                acc2 = gf ** 2 if acc2 is None else acc2 + gf ** 2
                n_acc += 1
            for tt, s in targets.items():
                if s not in full and st >= tt:
                    full[s] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    mean = acc / n_acc
    std = np.sqrt(np.maximum(acc2 / n_acc - mean ** 2, 0))
    np.savez(OUT / f'pixstats_{cam}.npz', mean=mean, std=std)
    for s, img in full.items():
        cv2.imwrite(str(OUT / f'frame_{cam}_{s}s.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'{cam}: {len(rows)} frames {time.time() - t0:.0f}s', flush=True)
    return cam, np.array(rows, dtype=np.float64)


def montage(s):
    tiles = []
    for cam in CAMS:
        img = cv2.imread(str(OUT / f'frame_{cam}_{s}s.jpg'))
        img = cv2.resize(img, (753, 364))
        cv2.putText(img, f'camera_{cam}', (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(img)
    tiles.append(np.zeros_like(tiles[0]))
    cv2.putText(tiles[-1], f't = {s} s', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
    cv2.imwrite(str(FIG / f'montage_{s}s.png'), grid)


def main():
    t0 = time.time()
    with Pool(len(CAMS)) as p:
        res = dict(p.map(per_camera, CAMS))
    np.savez(OUT / 'cam_stats.npz', fields=np.array(FIELDS), **res)
    for s in MONTAGE_T:
        montage(s)
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
