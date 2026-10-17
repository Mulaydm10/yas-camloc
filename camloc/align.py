"""Model-based alignment of the surveyed track borders to detected paint lines (image space).

For a candidate base_link pose (6-DoF), the 3D border points near the car are projected into each camera
and scored against a truncated distance transform of the detected line pixels (robust chamfer cost).
Used (1) at training time with the GT pose to self-calibrate camera rotations, and (2) at inference to
measure lateral offset and heading (D4).
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import densify3, project
from .lines import SCALE, detect, distance_map

ROAD_Z_IN_BASE = 0.066        # surveyed road surface is 6.6 cm above base_link (median over the training run)
LINE_CAMS = ('cl', 'cr', 'rl', 'rr', 'fr', 'fl')


def rot_adjust(rx, ry, rz):
    """Small rotation correction in the camera optical frame (radians), as a 4x4."""
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], float))
    T = np.eye(4)
    T[:3, :3] = R
    return T


class BorderMap:
    """3D border points (track + pit) densified to `step` m, with a spatial index."""

    def __init__(self, bounds3d, step=0.5):
        pts = [densify3(v, step) for k, v in bounds3d.items()]
        self.pts = np.vstack(pts)
        self.tree = cKDTree(self.pts[:, :2])

    def near(self, x, y, r=45.0):
        return self.pts[self.tree.query_ball_point([x, y], r)]

    def road_z(self, x, y, k=8):
        _, i = self.tree.query([x, y], k=k)
        return float(np.median(self.pts[i, 2]))


def ground_roi(cam, near=2.0, far=40.0):
    """Half-res pixel mask of where the flat road (z = ROAD_Z_IN_BASE) is seen between near and far."""
    h, w = int(cam.height * SCALE), int(cam.width * SCALE)
    u, v = np.meshgrid(np.arange(w) / SCALE, np.arange(h) / SCALE)
    pts = cv2.undistortPoints(np.c_[u.ravel(), v.ravel()].astype(np.float64).reshape(-1, 1, 2), cam.K, cam.D).reshape(-1, 2)
    rays_c = np.c_[pts, np.ones(len(pts))]
    R, t = cam.T_base_cam[:3, :3], cam.T_base_cam[:3, 3]
    rays_b = rays_c @ R.T
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = (ROAD_Z_IN_BASE - t[2]) / rays_b[:, 2]
    hit = (lam > 0) & np.isfinite(lam)
    d = np.where(hit, np.hypot(lam * rays_b[:, 0] + t[0], lam * rays_b[:, 1] + t[1]), np.inf)
    # plumb_bob undistortion is unreliable in the extreme corners of the wide lenses: drop |ray| too oblique
    ok = hit & (d > near) & (d < far) & (np.hypot(pts[:, 0], pts[:, 1]) < 2.0)
    return ok.reshape(h, w)


@dataclass
class CamObs:
    cam: object
    dist: np.ndarray        # truncated distance map (half-res, px)
    n_line_px: int


def observe(cameras, frameset, masks, rois, cams=LINE_CAMS):
    """Decode, detect lines, build distance maps for one FrameSet."""
    obs = {}
    for c in cams:
        if c not in frameset.images:
            continue
        img = cv2.imdecode(np.frombuffer(frameset.images[c], np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        valid = rois[c] & masks[c]
        m = detect(img, valid)
        obs[c] = CamObs(cameras[c], distance_map(m), int(m.sum()))
    return obs


def chamfer(cam_obs, T_map_base, pts, adjust=None, trunc=20.0, n_ref=None):
    """Mean squared truncated distance (half-res px) of projected border points to detected lines.
    n_ref: number of points expected in view (e.g. under the nominal pose); points that left the view
    count as `trunc`, so the cost cannot be lowered by rotating the borders out of the image."""
    uv, _, _ = project(cam_obs.cam, T_map_base, pts, adjust, max_range=40.0)
    n_ref = len(uv) if n_ref is None else max(n_ref, len(uv))
    if n_ref < 10:
        return None, 0
    if len(uv):
        uv = uv * SCALE
        d = cv2.remap(cam_obs.dist, uv[:, 0].astype(np.float32).reshape(-1, 1), uv[:, 1].astype(np.float32).reshape(-1, 1),
                      cv2.INTER_LINEAR, borderValue=trunc).ravel()
        ssum = float(np.sum(np.minimum(d, trunc) ** 2))
    else:
        ssum = 0.0
    return (ssum + (n_ref - len(uv)) * trunc ** 2) / n_ref, len(uv)
