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


def normalize(gray):
    """Local contrast normalisation (CLAHE): keeps ORB's absolute FAST threshold meaningful under lighting /
    contrast changes (EXP-0018). Applied identically at map build and query time (map flag `norm`)."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def detect(det, gray, mask, norm=False):
    kp, des = det.detectAndCompute(normalize(gray) if norm else gray, mask)
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
        self.norm = bool(z['norm']) if 'norm' in z.files else False
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
        # one detector / matcher per camera: OpenCV objects are not thread-safe, cameras run in parallel
        self.det = {c: orb() for c in self.masks}
        self.bf = {c: cv2.BFMatcher(cv2.NORM_HAMMING) for c in self.masks}

    def features(self, cam, gray_half):
        """ORB features (pose-independent, so they can be computed while the filter is still updating)."""
        return detect(self.det[cam], gray_half, self.masks[cam], self.vmap.norm)

    def localize(self, cam, gray_half, x, y, yaw, radius=40.0, feats=None):
        P, D = self.vmap.candidates(cam, x, y, yaw, radius)
        if P is None or len(P) < self.MIN_INLIERS:
            return None
        uv, des = feats if feats is not None else self.features(cam, gray_half)
        if len(uv) < self.MIN_INLIERS:
            return None
        m = self.bf[cam].knnMatch(des, D, k=2)
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
        return self._pose(cam, obj[inl], img[inl].astype(np.float64), rvec, tvec)

    def _pose(self, cam, obj, img, rvec, tvec):
        """LM-refine a PnP solution on its inliers (obj map points, img full-res px) -> base_link fix."""
        c = self.cams[cam]
        inl = np.arange(len(obj))
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, c.K, c.D, rvec, tvec)
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
                'median_depth': float(np.median(depth)),
                'uv_in': (img[inl] / 2.0).astype(np.float32), 'P_in': obj[inl]}      # for map-point tracking

    TRACK_FB_PX = 1.0           # LK forward-backward consistency (half-res px)

    def track(self, cam, prev_gray, gray, uv_prev, P):
        """TASK H.3: follow the last fix's inlier map points into `gray` with pyramidal LK and re-solve PnP with
        the SAME 3D points. Works when fresh ORB detection fails under blur (LK matches whole patches)."""
        if uv_prev is None or len(uv_prev) < self.MIN_INLIERS:
            return None
        p0 = uv_prev.reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=4)
        if p1 is None:
            return None
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(21, 21), maxLevel=4)
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < self.TRACK_FB_PX)
        if ok.sum() < self.MIN_INLIERS:
            return None
        img = p1.reshape(-1, 2)[ok].astype(np.float64) * 2.0
        obj = P[ok].astype(np.float64)
        c = self.cams[cam]
        good, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, c.K, c.D, iterationsCount=100, reprojectionError=4.0,
                                                   confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not good or inl is None or len(inl) < self.MIN_INLIERS:
            return None
        inl = inl.ravel()
        r = self._pose(cam, obj[inl], img[inl], rvec, tvec)
        r['tracked'] = True
        return r
