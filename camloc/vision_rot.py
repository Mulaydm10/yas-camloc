"""Visual heading increments: rotation of the car between a keyframe and now, per camera.

Features (Shi-Tomasi) are detected in the keyframe (outside the ego-body mask), tracked frame to frame with
pyramidal LK (forward-backward checked), and the keyframe -> current rotation is recovered from the
essential matrix of (keyframe position, current position). Errors therefore do not accumulate per frame,
only per keyframe (EXP-0007: chained frame-to-frame rotation is unbiased but outlier-prone).
"""
import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class KeyframeRotation:
    def __init__(self, cam, mask_half, R_base_cam, max_feat=300, min_feat=25):
        self.K = cam.K.copy()
        self.K[:2] *= 0.5                     # half-resolution intrinsics
        self.D = cam.D
        self.mask = (mask_half > 0).astype(np.uint8) * 255
        self.Rbc = R_base_cam
        self.max_feat, self.min_feat = max_feat, min_feat
        self.kf_pts = None                    # keyframe positions (N, 1, 2)
        self.cur_pts = None
        self.prev = None

    def reset(self, gray):
        p = cv2.goodFeaturesToTrack(gray, self.max_feat, 0.01, 8, mask=self.mask)
        self.kf_pts = p
        self.cur_pts = None if p is None else p.copy()
        self.prev = gray

    @property
    def alive(self):
        return self.cur_pts is not None and len(self.cur_pts) >= self.min_feat

    def track(self, gray):
        """Advance tracks to `gray`. Returns (dyaw_base_rad, n_inliers) for keyframe -> now, or None."""
        if not self.alive:
            self.prev = gray
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.cur_pts, None, **LK)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **LK)
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm((p0b - self.cur_pts).reshape(-1, 2), axis=1) < 1.0)
        self.kf_pts, self.cur_pts = self.kf_pts[ok], p1[ok]
        self.prev = gray
        if len(self.cur_pts) < self.min_feat:
            return None
        n0 = cv2.undistortPoints(self.kf_pts, self.K, self.D)
        n1 = cv2.undistortPoints(self.cur_pts, self.K, self.D)
        if np.median(np.linalg.norm((n1 - n0).reshape(-1, 2), axis=1)) < 1e-4:
            return 0.0, len(n0)               # no image motion (car stationary)
        E, inl = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1.0 / self.K[0, 0])
        if E is None or E.shape != (3, 3):
            return None
        n_in, R, _, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=inl)
        if n_in < self.min_feat:
            return None
        Rb = self.Rbc @ R.T @ self.Rbc.T      # camera motion expressed in base_link
        return float(np.arctan2(Rb[1, 0], Rb[0, 0])), int(n_in)
