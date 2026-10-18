"""Causal EKF on the car pose (x, y, yaw) in `map`.

The state lives in Cartesian map coordinates; measurements are expressed in track coordinates
(lateral offset n, heading error, progress s) through the TrackModel, which keeps the pit lane / track
topology out of the state. Prediction: unicycle driven by odometry (v, wz) with continuous white input
noise calibrated on the training run. D2 adds one map-only measurement: the track-bounds constraint
(the car centre stays within the lane borders plus a margin).
"""
import numpy as np

from .track import wrap


class PoseEKF:
    """State X = [x, y, yaw, yaw_kf(, slip)]. yaw_kf is a stochastic clone of the heading at the last visual
    keyframe (static in prediction), so relative visual rotation "yaw - yaw_kf" can be fused consistently.
    Optional `slip` (dict sigma0, q, tau[, c_sigma0, c_q, clip, c_clip]): sideslip states added to the travel
    direction, first-order Gauss-Markov processes learned online from the absolute fixes (TASK D.2, EXP-0026):
    b (rad), and with c_* also c (rad per m/s^2 of lateral acceleration a_y = v * wz), so the travel direction
    is heading + beta + b + c * a_y. q_lat (m per sqrt(m)) adds lateral process noise scaled with distance."""
    N = 4

    def __init__(self, x, y, yaw, q, p0=(0.05, 0.05, np.radians(0.5)), slip=None):
        self.slip = slip
        self.nc = 1 if slip and slip.get('c_sigma0') else 0
        self.N = 4 + (1 + self.nc if slip else 0)
        extra = ([slip['sigma0']] + ([slip['c_sigma0']] if self.nc else [])) if slip else []
        self.X = np.array([x, y, yaw, yaw] + [0.0] * len(extra), float)
        self.P = np.diag(np.square([*p0, p0[2]] + extra))
        self.q_lat = 0.0
        self.P[2, 3] = self.P[3, 2] = self.P[2, 2]
        self.q = q

    def _row(self, *vals):
        h = np.zeros((1, self.N))
        h[0, :len(vals)] = vals
        return h

    def clone_heading(self):
        """Start a new keyframe: yaw_kf := yaw, with the matching covariance."""
        self.X[3] = self.X[2]
        self.P[3, :] = self.P[2, :]
        self.P[:, 3] = self.P[:, 2]
        self.P[3, 3] = self.P[2, 2]

    # --------------------------------------------------------------- prediction
    def predict(self, dt, v, wz, beta=0.0):
        """beta: sideslip, the travel direction is heading + beta."""
        if dt <= 0:
            return
        x, y, h = self.X[:3]
        ay = v * wz
        hm = h + 0.5 * wz * dt + beta
        if self.slip:
            hm += self.X[4] + (self.X[5] * ay if self.nc else 0.0)
        c, s = np.cos(hm), np.sin(hm)
        self.X[:3] = [x + v * c * dt, y + v * s * dt, wrap(h + wz * dt)]
        F = np.eye(self.N)
        F[0, 2], F[1, 2] = -v * s * dt, v * c * dt
        G = np.zeros((self.N, 2))
        G[0, 0], G[1, 0], G[2, 1] = c * dt, s * dt, dt
        sv = self.q['sigma_v_frac'] * abs(v) + self.q['sigma_v_abs'] * (abs(v) > 0.5)
        M = np.diag([sv ** 2, self.q['sigma_wz'] ** 2]) / dt
        Q = G @ M @ G.T
        if self.slip:
            a = np.exp(-dt / self.slip['tau'])
            F[0, 4], F[1, 4], F[4, 4] = -v * s * dt, v * c * dt, a
            self.X[4] *= a
            Q[4, 4] += self.slip['q'] ** 2 * dt
            if self.nc:
                F[0, 5], F[1, 5], F[5, 5] = -v * s * dt * ay, v * c * dt * ay, a
                self.X[5] *= a
                Q[5, 5] += self.slip['c_q'] ** 2 * dt
        if self.q_lat:
            n = np.array([-s, c])
            Q[:2, :2] += np.outer(n, n) * self.q_lat ** 2 * abs(v) * dt
        self.P = F @ self.P @ F.T + Q

    # --------------------------------------------------------------- updates
    def update(self, z, h, H, R, angle=False):
        """Generic scalar/vector update. z, h: measurement and prediction; H: Jacobian; R: covariance."""
        z, h, H, R = np.atleast_1d(z), np.atleast_1d(h), np.atleast_2d(H), np.atleast_2d(R)
        r = z - h
        if angle:
            r = wrap(r)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ r
        self.X[2] = wrap(self.X[2])
        self.X[3] = wrap(self.X[3])
        I = np.eye(self.N)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T      # Joseph form
        self._clip_slip()
        return float(r @ np.linalg.solve(S, r))                          # NIS

    def update_lateral(self, lane, n_meas, sigma):
        """Measurement of the signed lateral offset n on `lane` (n > 0 = left of the driving direction)."""
        s, n = lane.project(self.X[0], self.X[1], self.X[2])[:2]
        hdg = lane.heading_at(s)[0]
        H = self._row(-np.sin(hdg), np.cos(hdg))
        return self.update(n_meas, n[0], H, sigma ** 2)

    def update_heading(self, lane, epsi_meas, sigma):
        """Measurement of the heading relative to the lane centreline."""
        s, _ = lane.project(self.X[0], self.X[1], self.X[2])[:2]
        hdg = lane.heading_at(s)[0]
        return self.update(epsi_meas, wrap(self.X[2] - hdg), self._row(0, 0, 1.0), sigma ** 2, angle=True)

    def update_pose(self, x, y, yaw, sigma_xy, sigma_yaw):
        """Absolute pose fix (map-based PnP): z = (x, y, yaw)."""
        H = np.zeros((3, self.N))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        z = np.array([x, y, yaw])
        h = self.X[:3].copy()
        r = z - h
        r[2] = wrap(r[2])
        R = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ r
        self.X[2], self.X[3] = wrap(self.X[2]), wrap(self.X[3])
        I = np.eye(self.N)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self._clip_slip()
        return float(r @ np.linalg.solve(S, r))

    def _clip_slip(self):
        if self.slip:
            if self.slip.get('clip'):
                self.X[4] = np.clip(self.X[4], -self.slip['clip'], self.slip['clip'])
            if self.nc and self.slip.get('c_clip'):
                self.X[5] = np.clip(self.X[5], -self.slip['c_clip'], self.slip['c_clip'])

    def update_relative_yaw(self, dyaw_meas, sigma):
        """Visual rotation since the keyframe: z = yaw - yaw_kf."""
        return self.update(dyaw_meas, wrap(self.X[2] - self.X[3]), self._row(0, 0, 1.0, -1.0), sigma ** 2, angle=True)

    def constrain_to_lane(self, track, margin=0.75, sigma=0.3, sigma_h=np.radians(2), gate_h=np.radians(10), lane_name=None):
        """Map priors on STRAIGHTS only (Lane.straight: |curvature| < 1/500 within +-100 m), where they do
        not depend on the along-track position s, so an s error cannot turn them into wrong corrections:
          heading: epsi ~ N(0, 2 deg)   (training GT on straights: |epsi| p95 1.6, max 5.3 deg); gated at 10 deg
          lateral: car centre within the lane borders + 0.75 m (training GT: max excess 0.5 m)
        In corners nothing is applied. Returns (lane name, applied?)."""
        if lane_name is None:
            lane_name = str(track.locate(self.X[0], self.X[1], self.X[2])[0][0])
        name = lane_name
        lane = track.lanes[name]
        s, n, e = lane.project(self.X[0], self.X[1], self.X[2])
        if not lane.straight_at(s)[0]:
            return name, False
        if abs(e[0]) < gate_h:
            self.update_heading(lane, 0.0, sigma_h)
        s, n = lane.project(self.X[0], self.X[1], self.X[2])[:2]
        wl, wr = lane.width_at(s)
        lo, hi = -wr[0] - margin, wl[0] + margin
        if n[0] > hi:
            self.update_lateral(lane, hi, sigma)
        elif n[0] < lo:
            self.update_lateral(lane, lo, sigma)
        return name, True

    @property
    def pose(self):
        return float(self.X[0]), float(self.X[1]), float(self.X[2])
