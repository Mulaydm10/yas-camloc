"""3D road model from the surveyed borders (map-only, no GT): road height under the car and the car's
attitude implied by the road surface.

Fitted on the training run (cache/roadattitude.py): GT pitch = 0.99 * map pitch (residual 0.14 deg);
GT roll = 1.02 * map roll - 0.18 deg (residual 0.50 deg; lateral acceleration adds nothing).
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import densify3

ROAD_Z_IN_BASE = 0.066         # road surface above base_link (median, training run)
ROLL_GAIN, ROLL_OFFSET = 1.024, -0.0032
PITCH_GAIN, PITCH_OFFSET = 0.988, 0.0001


class Road3D:
    def __init__(self, bounds3d):
        self.L = densify3(bounds3d['track_left'], 0.25)
        self.R = densify3(bounds3d['track_right'], 0.25)
        self.PL = densify3(bounds3d['pitlane_left'], 0.25)
        self.PR = densify3(bounds3d['pitlane_right'], 0.25)
        self.trees = {k: cKDTree(v[:, :2]) for k, v in (('L', self.L), ('R', self.R), ('PL', self.PL), ('PR', self.PR))}
        self.pts = {'L': self.L, 'R': self.R, 'PL': self.PL, 'PR': self.PR}

    def _pair(self, x, y):
        """Nearest left/right border points of whichever lane (track or pit) is closer."""
        best = None
        for a, b in (('L', 'R'), ('PL', 'PR')):
            da, ia = self.trees[a].query([x, y])
            db, ib = self.trees[b].query([x, y])
            if best is None or da + db < best[0]:
                best = (da + db, self.pts[a][ia], self.pts[b][ib], da, db)
        return best[1:]

    def z(self, x, y):
        l, r, dl, dr = self._pair(x, y)
        return (l[2] * dr + r[2] * dl) / max(dl + dr, 1e-6)

    def attitude(self, x, y, yaw):
        """(z_base, roll, pitch) of base_link implied by the road at (x, y) heading yaw."""
        l, r, dl, dr = self._pair(x, y)
        zr = (l[2] * dr + r[2] * dl) / max(dl + dr, 1e-6)
        roll_map = np.arctan2(l[2] - r[2], np.hypot(*(l[:2] - r[:2])))
        f = 10.0 * np.array([np.cos(yaw), np.sin(yaw)])
        pitch_map = -np.arctan2(self.z(x + f[0], y + f[1]) - self.z(x - f[0], y - f[1]), 20.0)
        return (zr - ROAD_Z_IN_BASE, ROLL_GAIN * roll_map + ROLL_OFFSET, PITCH_GAIN * pitch_map + PITCH_OFFSET)
