"""Geo-referenced visual feature map and map-based localisation (PnP).

The map (built at training time by tools/build_map.py, from GT camera poses) stores, per keyframe, the
observing camera's pose and a set of triangulated 3D points (map frame) with ORB descriptors. At inference,
keyframes near the predicted pose are selected, the current image's ORB features are matched to their
points, and the camera pose is solved with PnP + RANSAC, giving an absolute (x, y, yaw) of base_link.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import T_map_base

ORB_PARAMS = dict(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=12)


def orb():
    return cv2.ORB_create(**ORB_PARAMS)


def detect(det, gray, mask):
    kp, des = det.detectAndCompute(gray, mask)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
    return np.array([k.pt for k in kp], np.float32), des


class VisualMap:
    def __init__(self, path):
        z = np.load(path)
        self.kf_cam = z['kf_cam']          # (K,) camera index into z['cams']
        self.kf_xy = z['kf_xy']            # (K, 2) base_link position at the keyframe
        self.kf_yaw = z['kf_yaw']
        self.kf_ptr = z['kf_ptr']          # (K+1,) slice into points
        self.P = z['P']                    # (N, 3) map points
        self.D = z['D']                    # (N, 32) ORB descriptors
        self.cams = [str(c) for c in z['cams']]
        self.tree = cKDTree(self.kf_xy)

    def candidates(self, cam, x, y, yaw, radius=40.0, max_dyaw=np.radians(35), max_kf=12):
        """Points of up to max_kf keyframes of `cam` within radius of (x, y) and similar heading."""
        ci = self.cams.index(cam)
        idx = self.tree.query_ball_point([x, y], radius)
        idx = [k for k in idx if self.kf_cam[k] == ci
               and abs(np.angle(np.exp(1j * (self.kf_yaw[k] - yaw)))) < max_dyaw]
        if not idx:
            return None, None
        idx = sorted(idx, key=lambda k: np.hypot(*(self.kf_xy[k] - [x, y])))[:max_kf]
        sl = np.concatenate([np.arange(self.kf_ptr[k], self.kf_ptr[k + 1]) for k in idx])
        return self.P[sl], self.D[sl]


class MapLocalizer:
    """Per-camera PnP against the map. Returns base_link (x, y, yaw) and the inlier count, or None."""
    MIN_INLIERS = 18

    def __init__(self, vmap, cameras, T_base_cam_adj, masks_half):
        self.vmap = vmap
        self.cams = cameras
        self.Tbc = T_base_cam_adj
        self.masks = {c: (m > 0).astype(np.uint8) * 255 for c, m in masks_half.items()}
        self.det = orb()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def localize(self, cam, gray_half, x, y, yaw, radius=40.0):
        P, D = self.vmap.candidates(cam, x, y, yaw, radius)
        if P is None or len(P) < self.MIN_INLIERS:
            return None
        uv, des = detect(self.det, gray_half, self.masks[cam])
        if len(uv) < self.MIN_INLIERS:
            return None
        m = self.bf.knnMatch(des, D, k=2)
        good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.8 * b.distance and a.distance < 64]
        if len(good) < self.MIN_INLIERS:
            return None
        img = uv[[g.queryIdx for g in good]] * 2.0                    # half-res -> full-res pixels
        obj = P[[g.trainIdx for g in good]].astype(np.float64)
        c = self.cams[cam]
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img.astype(np.float64), c.K, c.D, iterationsCount=200,
                                                 reprojectionError=4.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.MIN_INLIERS:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl].astype(np.float64), c.K, c.D, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_cam_map = np.eye(4)
        T_cam_map[:3, :3], T_cam_map[:3, 3] = R, tvec.ravel()
        T_map_base_ = np.linalg.inv(T_cam_map) @ np.linalg.inv(self.Tbc[cam])
        bx, by = T_map_base_[0, 3], T_map_base_[1, 3]
        byaw = np.arctan2(T_map_base_[1, 0], T_map_base_[0, 0])
        proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, c.K, c.D)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img[inl]) ** 2, axis=1))))
        depth = (R @ obj[inl].T + tvec).T[:, 2]
        return {'x': bx, 'y': by, 'yaw': byaw, 'n': int(len(inl)), 'rms_px': rms,
                'median_depth': float(np.median(depth))}
