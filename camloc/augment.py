"""Image augmentations for robustness STRESS TESTS only (never used by default): applied to the half-resolution
grey query images by tools/vmap_check.py (--aug) and by the localiser through run.py --set AUG=<name>.

global: gammaG, contrastC, blurK, noiseS, day;  spatial (TASK G): flood, shadow, sky, speedV (blur for V m/s)
"""
import cv2
import numpy as np


def augment(g, aug):
    f = g.astype(np.float32)
    if aug.startswith('gamma'):
        f = 255 * (f / 255) ** float(aug[5:])
    elif aug.startswith('contrast'):
        f = 128 + float(aug[8:]) * (f - 128)
    elif aug.startswith('blur'):
        k = int(aug[4:])
        f = cv2.filter2D(f, -1, np.ones((1, k), np.float32) / k)
    elif aug.startswith('noise'):
        f = f + np.random.default_rng(int(f.sum()) % 2**32).normal(0, float(aug[5:]), f.shape)
    elif aug == 'day':
        f = 128 + 0.6 * (255 * (f / 255) ** 0.4 - 128)
    # spatial augmentations (Devin TASK G): local effects that a global CLAHE cannot undo
    elif aug == 'flood':          # floodlight glare: three bright Gaussian blobs, positions fixed per image size
        h, w = f.shape
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(h * 7 + w)
        for _ in range(3):
            cy, cx, r = rng.uniform(0.1, 0.6) * h, rng.uniform(0.1, 0.9) * w, rng.uniform(0.06, 0.12) * w
            f = f + 220 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    elif aug == 'shadow':         # cast shadow: a dark diagonal band over the lower half (55 % darker)
        h, w = f.shape
        yy, xx = np.mgrid[0:h, 0:w]
        band = (yy > 0.45 * h) & ((xx + 0.8 * yy) % (0.9 * w) < 0.45 * w)
        f = np.where(band, 0.45 * f, f)
    elif aug == 'sky':            # saturated sky / sun: the top 25 % clipped to white
        h = f.shape[0]
        f[: h // 4] = 255
    elif aug.startswith('speed'):  # motion blur scaled to speed: speedV = blur for V m/s (half-res px ~ V / 3)
        k = max(3, int(round(float(aug[5:]) / 3)))
        f = cv2.filter2D(f, -1, np.ones((1, k), np.float32) / k)
    elif aug != 'none':
        raise SystemExit(f'unknown --aug {aug}')
    return np.clip(f, 0, 255).astype(np.uint8)


def flow_field(cam, v, wz, h_ground, shape, step=16, z_far=60.0):
    """Image motion field (half-res px/s) of a camera on a car moving at v (m/s, base x) and yaw rate wz (rad/s),
    for scene points on the ground plane (h_ground metres below the camera) or at z_far when the ray does not
    hit the ground. Returns grid centres (us, vs) and flow (du, dv), on a step-px grid."""
    K = cam.K.copy()
    K[:2] *= 0.5
    Rbc, tbc = cam.T_base_cam[:3, :3], cam.T_base_cam[:3, 3]
    Rcb = Rbc.T
    vb = np.array([v, 0.0, 0.0]) + np.cross([0.0, 0.0, wz], tbc)     # camera velocity in base
    Vc, Wc = Rcb @ vb, Rcb @ np.array([0.0, 0.0, wz])
    h, w = shape
    us, vs = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    d_b = np.einsum('ij,jkl->ikl', Rbc, np.stack([x, y, np.ones_like(x)]))     # ray direction in base (per unit cam z)
    Z = np.where(d_b[2] < -1e-3, h_ground / np.maximum(-d_b[2], 1e-3), z_far)
    Z = np.minimum(Z, z_far)
    vx, vy, vz = Vc
    wx, wy, wzc = Wc
    xd = (-vx + x * vz) / Z + x * y * wx - (1 + x * x) * wy + y * wzc       # standard motion field (normalised)
    yd = (-vy + y * vz) / Z + (1 + y * y) * wx - x * y * wy - x * wzc
    return us, vs, xd * K[0, 0], yd * K[1, 1]


def flow_blur(g, cam, v, wz, exposure_s, h_ground, tiles=(6, 8)):
    """Physical motion blur (TASK H): per image tile, a line kernel of length exposure * |flow| along the flow."""
    f = g.astype(np.float32)
    h, w = f.shape
    us, vs, du, dv = flow_field(cam, v, wz, h_ground, (h, w))
    out = f.copy()
    th, tw = h // tiles[0], w // tiles[1]
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            y0, y1, x0, x1 = i * th, (i + 1) * th if i < tiles[0] - 1 else h, j * tw, (j + 1) * tw if j < tiles[1] - 1 else w
            m = (vs >= y0) & (vs < y1) & (us >= x0) & (us < x1)
            if not m.any():
                continue
            bx, by = exposure_s * du[m].mean(), exposure_s * dv[m].mean()
            L = float(np.hypot(bx, by))
            if L < 1.0:
                continue
            n = int(np.ceil(L)) | 1
            k = np.zeros((n, n), np.float32)
            c = n // 2
            cv2.line(k, (int(round(c - bx / 2)), int(round(c - by / 2))), (int(round(c + bx / 2)), int(round(c + by / 2))), 1.0, 1)
            k /= max(k.sum(), 1e-6)
            pad = n
            ya, yb, xa, xb = max(0, y0 - pad), min(h, y1 + pad), max(0, x0 - pad), min(w, x1 + pad)
            blk = cv2.filter2D(f[ya:yb, xa:xb], -1, k)
            out[y0:y1, x0:x1] = blk[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return np.clip(out, 0, 255).astype(np.uint8)
