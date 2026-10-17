"""Ego-body masks (255 = usable scene, 0 = own car) from hand-drawn polygons.

Automatic extraction was tried and rejected: at night both temporal std and the median frame difference
are dominated by JPEG/sensor noise, and smooth asphalt / hazy sky read as "static" exactly like the
body. The body outline, however, is crisp in the per-camera MEAN image, so the polygons in
camloc/assets/body_polygons.json were drawn on it. This tool renders them to masks and a preview.

    python -m camloc.tools.make_masks            # writes camloc/assets/masks/<cam>.png, cache/masks_preview.png
"""
import json

import cv2
import numpy as np

from .. import config as C
from ..data import FrameSet, Replay

ASSETS = C.ASSETS
W, H = 1506, 728


def render(polys, w=W, h=H):
    m = np.full((h, w), 255, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 0)
    return m


def load_masks(cams=C.CAMS):
    """Masks for inference (bool, True = usable pixel), rendered from the polygons on the fly."""
    polys = json.loads((ASSETS / 'body_polygons.json').read_text())
    return {c: render(polys.get(c, [])) > 0 for c in cams}


def mean_images(n=200, step=50):
    acc, k = {}, 0
    for ev in Replay(state=None):
        if not isinstance(ev, FrameSet):
            continue
        k += 1
        if k % step:
            continue
        for c, b in ev.images.items():
            g = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float64)
            acc[c] = acc.get(c, 0) + g
        if k // step >= n:
            break
    return {c: (a / (k // step)).astype(np.uint8) for c, a in acc.items()}


def main():
    polys = json.loads((ASSETS / 'body_polygons.json').read_text())
    (ASSETS / 'masks').mkdir(parents=True, exist_ok=True)
    means = mean_images()
    tiles = []
    for c in C.CAMS:
        m = render(polys.get(c, []))
        cv2.imwrite(str(ASSETS / 'masks' / f'{c}.png'), m)
        v = cv2.cvtColor(cv2.equalizeHist(means[c]), cv2.COLOR_GRAY2BGR)
        v[m == 0] = (0.5 * v[m == 0] + [0, 0, 128]).astype(np.uint8)
        cv2.putText(v, f'{c}: masked {(m == 0).mean() * 100:.1f}%', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        tiles.append(cv2.resize(v, (753, 364)))
        print(f'{c}: masked {(m == 0).mean() * 100:.1f}%')
    tiles.append(np.zeros_like(tiles[0]))
    C.CACHE.mkdir(exist_ok=True)
    cv2.imwrite(str(C.CACHE / 'masks_preview.png'), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)]))


if __name__ == '__main__':
    main()
