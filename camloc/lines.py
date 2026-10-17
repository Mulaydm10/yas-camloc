"""Track-paint detector: white edge lines and the blue run-off paint, the cues that carry the surveyed
border geometry. Works on half-resolution images (753x364) for speed.

Output is an evidence map in [0, 1] and a binary line mask; `distance_map` turns it into a truncated
distance transform for chamfer matching of projected map borders.
"""
import cv2
import numpy as np

SCALE = 0.5


def evidence(bgr, valid):
    """bgr: half-res image; valid: bool mask (True = usable, i.e. not ego body). Returns (white, blue_edge)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # white paint: thin bright structure, brighter than its surroundings (top-hat), not a saturated light
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 7)))
    white = (th > 18) & (g < 235)
    # blue run-off paint: blue clearly above red, then keep its boundary (the edge we can localise)
    b, _, r = cv2.split(bgr.astype(np.int16))
    blue = ((b - r) > 18) & (b > 60)
    blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue_edge = cv2.morphologyEx(blue, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    white &= valid
    blue_edge &= valid
    return white, blue_edge


def detect(bgr_half, valid_half):
    """Line mask (half-res). `valid_half` should already exclude the ego body and non-ground pixels
    (see align.ground_roi). The blue channel was dropped: at night it is dominated by JPEG block noise."""
    white, _ = evidence(bgr_half, valid_half)
    m = white
    # drop specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 12
    return keep[lab]


def distance_map(line_mask, trunc=20.0):
    """Truncated L2 distance (px, half-res) to the nearest detected line pixel."""
    dt = cv2.distanceTransform((~line_mask).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(dt, trunc)
