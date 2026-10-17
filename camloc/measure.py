"""Per-point border measurement: project the NEAR-side surveyed border into a camera and search a short
vertical window around each projected point for the white-line peak (top-hat response).

Returns, per accepted point, the predicted pixel and the measured vertical offset dv (half-res px).
Only the near border is used: far borders, the pit lane and occluded stretches are what made a global
chamfer cost flat (EXP-0005).
"""
import cv2
import numpy as np

from .geometry import T_map_base, project
from .lines import SCALE

SIDE = {'cl': +1, 'rl': +1, 'cr': -1, 'rr': -1}          # which side of base_link the camera looks at
RANGE = {'cl': (3, 25), 'rl': (3, 25), 'cr': (3, 25), 'rr': (3, 25), 'fr': (8, 40), 'fl': (10, 60)}
SEARCH_PX = 14            # half-res vertical search half-window
MIN_RESP = 12             # minimum top-hat response (grey levels)


def tophat(img_half_bgr):
    g = cv2.cvtColor(img_half_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 7))).astype(np.float32)


def near_border_points(T_mb, border_pts, cam_name):
    """Border points (map, Nx3) on the camera's side and within its range, in base_link terms."""
    Tbm = np.linalg.inv(T_mb)
    P = (Tbm[:3, :3] @ border_pts.T + Tbm[:3, 3:4]).T
    lo, hi = RANGE[cam_name]
    d = np.hypot(P[:, 0], P[:, 1])
    keep = (d > lo) & (d < hi)
    if cam_name in SIDE:
        keep &= np.sign(P[:, 1]) == SIDE[cam_name]
        # nearest border on that side only: drop points much farther laterally than the nearest ones
        if keep.any():
            lat = np.abs(P[keep, 1])
            keep[np.where(keep)[0][lat > np.percentile(lat, 20) + 4.0]] = False
    else:
        keep &= P[:, 0] > lo
    return border_pts[keep]


def vertical_residuals(cam, th_half, valid_half, T_mb, pts, adjust=None):
    """For each projected point: search +-SEARCH_PX rows for the top-hat peak. Returns arrays
    (u, v_pred, dv, resp) in half-res pixels for accepted points."""
    uv, _, _ = project(cam, T_mb, pts, adjust, max_range=80.0)
    if not len(uv):
        return np.zeros((0, 4))
    uv = uv * SCALE
    h, w = th_half.shape
    u = np.round(uv[:, 0]).astype(int)
    v0 = uv[:, 1]
    ok = (u >= 0) & (u < w) & (v0 > SEARCH_PX + 1) & (v0 < h - SEARCH_PX - 2)
    u, v0 = u[ok], v0[ok]
    if not len(u):
        return np.zeros((0, 4))
    offs = np.arange(-SEARCH_PX, SEARCH_PX + 1)
    rows = np.round(v0).astype(int)[:, None] + offs[None, :]
    prof = th_half[rows, u[:, None]] * valid_half[rows, u[:, None]]
    k = np.argmax(prof, axis=1)
    resp = prof[np.arange(len(k)), k]
    # sub-pixel parabola
    km, kp = np.clip(k - 1, 0, len(offs) - 1), np.clip(k + 1, 0, len(offs) - 1)
    a, b, c = prof[np.arange(len(k)), km], resp, prof[np.arange(len(k)), kp]
    den = a - 2 * b + c
    sub = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den, 0.0)
    v_meas = rows[np.arange(len(k)), k] + np.clip(sub, -0.5, 0.5)
    good = (resp >= MIN_RESP) & (k > 0) & (k < len(offs) - 1)
    return np.c_[u[good], v0[good], v_meas[good] - v0[good], resp[good]]
