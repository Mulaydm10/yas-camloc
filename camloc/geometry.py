"""Camera geometry: map points -> pixels for a base_link pose, and pixel rays.

Frames: map (x east-ish, y, z up); base_link (x forward, y left, z up, REP 103); camera optical frame
(z forward, x right, y down). Camera.T_base_cam maps optical-frame points into base_link.
"""
import cv2
import numpy as np


def rpy_to_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def T_map_base(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def project(cam, T_map_base_, pts_map, T_cam_adjust=None, max_range=80.0):
    """Project (N,3) map points into `cam`. Returns (uv (M,2), idx (M,), depth (M,)) for the points in
    front of the camera, within max_range and inside the image. T_cam_adjust: optional 4x4 correction
    applied in the camera frame (for extrinsic self-calibration)."""
    T_base_cam = cam.T_base_cam if T_cam_adjust is None else cam.T_base_cam @ T_cam_adjust
    T_cam_map = np.linalg.inv(T_map_base_ @ T_base_cam)
    P = (T_cam_map[:3, :3] @ pts_map.T + T_cam_map[:3, 3:4]).T
    keep = (P[:, 2] > 0.5) & (np.linalg.norm(P, axis=1) < max_range)
    idx = np.where(keep)[0]
    if not len(idx):
        return np.zeros((0, 2)), idx, np.zeros(0)
    uv, _ = cv2.projectPoints(P[idx], np.zeros(3), np.zeros(3), cam.K, cam.D)
    uv = uv.reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    # plumb_bob blows up far outside the calibrated field: reject points whose undistorted ray is too oblique
    xn = P[idx, :2] / P[idx, 2:3]
    inside &= np.hypot(xn[:, 0], xn[:, 1]) < 1.2 * max(cam.width / cam.K[0, 0], cam.height / cam.K[1, 1])
    return uv[inside], idx[inside], P[idx[inside], 2]


def densify3(p, step=0.25):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    return np.c_[[np.interp(s, d, p[:, k]) for k in range(p.shape[1])]].T
