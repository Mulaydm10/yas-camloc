"""Track model: the race track (closed lane) and the pit lane (open lane) as 1 m centrelines with
curvature and half-widths, (x, y) <-> (s, n) conversion, and the pit branch points on the track.

Conventions: s increases in the driving direction; n is positive to the LEFT of the driving direction;
epsi = yaw - centreline heading, wrapped to (-pi, pi]. Track s = 0 is an arbitrary fixed point of the
centreline (index 0 after construction); it is deterministic for a given bounds survey.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from . import config as C


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _arclen(p):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def _resample(p, step, closed):
    if closed:
        p = np.vstack([p, p[:1]])
    d = _arclen(p)
    n = int(round(d[-1] / step))
    s = np.linspace(0, d[-1], n, endpoint=not closed) if closed else np.linspace(0, d[-1], n + 1)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def _densify(p, step=0.1):
    d = _arclen(p)
    s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _centreline(left, right, closed, step=1.0):
    rd = _densify(np.vstack([right, right[:1]]) if closed else right)
    _, j = cKDTree(rd).query(left)
    mid = 0.5 * (left + rd[j])
    c = _resample(mid, step, closed)
    k = 15
    if closed:
        ext = savgol_filter(np.vstack([c[-k:], c, c[:k]]), 21, 3, axis=0)[k:-k]
    else:
        ext = savgol_filter(c, 21, 3, axis=0)
    return _resample(ext, step, closed)


@dataclass
class Lane:
    name: str
    xy: np.ndarray          # (N, 2) centreline points, ~1 m apart
    closed: bool
    left: np.ndarray        # border polylines (for widths / plotting)
    right: np.ndarray

    def __post_init__(self):
        self.seg = (np.roll(self.xy, -1, axis=0) if self.closed else np.vstack([self.xy[1:], self.xy[-1:]])) - self.xy
        if not self.closed:
            self.seg[-1] = self.seg[-2]
        self.seg_len = np.hypot(*self.seg.T)
        self.s = np.r_[0, np.cumsum(self.seg_len[:-1])]
        self.length = float(self.s[-1] + (self.seg_len[-1] if self.closed else 0.0))
        self.hdg = np.arctan2(self.seg[:, 1], self.seg[:, 0])            # per segment
        nxt = np.roll(self.xy, -1, axis=0) if self.closed else np.vstack([self.xy[1:], self.xy[-1:]])
        prv = np.roll(self.xy, 1, axis=0) if self.closed else np.vstack([self.xy[:1], self.xy[:-1]])
        self.hdg_v = np.arctan2(*(nxt - prv)[:, ::-1].T)                # per vertex (smooth)
        dh = wrap(np.roll(self.hdg, -1) - self.hdg) if self.closed else np.r_[wrap(np.diff(self.hdg)), 0]
        kap = dh / np.maximum(self.seg_len, 1e-6)
        if self.closed:
            kap = savgol_filter(np.r_[kap[-10:], kap, kap[:10]], 21, 2)[10:-10]
        else:
            kap = savgol_filter(kap, 21, 2)
        self.kappa = kap
        nrm = np.c_[-np.sin(self.hdg), np.cos(self.hdg)]
        lt, rt = cKDTree(_densify(self.left)), cKDTree(_densify(self.right))
        # half-widths measured to the nearest border point on each side of the driving direction
        self.w_left = lt.query(self.xy)[0]
        self.w_right = rt.query(self.xy)[0]
        il = cKDTree(self.left).query(self.xy)[1]
        side = np.median(np.sum((self.left[il] - self.xy) * nrm, axis=1))   # > 0: `left` border is on the left
        if side < 0:
            self.w_left, self.w_right = self.w_right, self.w_left
        self._tree = cKDTree(self.xy)
        # "straight" = |curvature| < 1/500 m everywhere within +-100 m. There, the lane heading and width do
        # not depend on s, so map priors stay valid even with a large along-track error.
        from scipy.ndimage import maximum_filter1d
        win = int(round(100 / max(np.median(self.seg_len), 1e-3)))
        kmax = maximum_filter1d(np.abs(self.kappa), size=2 * win + 1, mode='wrap' if self.closed else 'nearest')
        self.straight = kmax < 1 / 500

    # ------------------------------------------------------------------ geometry
    def project(self, x, y, yaw=None, max_heading_diff=np.radians(60), k=40):
        """(x, y[, yaw]) -> s, n[, epsi]. Vectorised. With yaw, only centreline vertices whose direction is
        within `max_heading_diff` of yaw are candidates, so parallel legs driven the other way (hairpins,
        pit straight) cannot capture the point; falls back to the plain nearest vertex if none qualifies."""
        p = np.c_[np.atleast_1d(x), np.atleast_1d(y)]
        if yaw is None:
            _, j = self._tree.query(p)
        else:
            kk = min(k, len(self.xy))
            d, jj = self._tree.query(p, k=kk)
            dh = np.abs(wrap(np.atleast_1d(yaw)[:, None] - self.hdg_v[jj]))
            cost = np.where(dh < max_heading_diff, d, d + 1e6)
            j = jj[np.arange(len(p)), np.argmin(cost, axis=1)]
        best = None
        for jj in (j - 1, j):                      # try the segment before and after the nearest vertex
            if self.closed:
                jj = jj % len(self.xy)
            else:
                jj = np.clip(jj, 0, len(self.xy) - 2)
            a = self.xy[jj]
            d = self.seg[jj]
            L2 = np.maximum(np.sum(d * d, axis=1), 1e-9)
            u = np.clip(np.sum((p - a) * d, axis=1) / L2, 0, 1)
            q = a + u[:, None] * d
            dist = np.hypot(*(p - q).T)
            if best is None:
                best = [jj, u, q, dist]
            else:
                m = dist < best[3]
                for k, v in enumerate((jj, u, q, dist)):
                    best[k] = np.where(m[:, None], v, best[k]) if np.ndim(v) == 2 else np.where(m, v, best[k])
        jj, u, q, dist = best
        jj = jj.astype(int)
        s = self.s[jj] + u * self.seg_len[jj]
        if self.closed:
            s = s % self.length
        t = self.seg[jj] / np.maximum(self.seg_len[jj], 1e-9)[:, None]
        r = p - q
        n = t[:, 0] * r[:, 1] - t[:, 1] * r[:, 0]
        if yaw is None:
            return s, n
        return s, n, wrap(np.atleast_1d(yaw) - self.heading_at(s))

    def _interp(self, s, arr, angle=False):
        s = np.atleast_1d(s).astype(float)
        if self.closed:
            s = s % self.length
            xs = np.r_[self.s, self.length]
            ys = np.r_[arr, arr[:1]]
        else:
            xs, ys = self.s, arr
        if angle:
            yu = np.unwrap(ys)
            return wrap(np.interp(s, xs, yu))
        return np.interp(s, xs, ys)

    def heading_at(self, s):
        """Smooth centreline heading (central differences at the vertices, interpolated)."""
        return self._interp(s, self.hdg_v, angle=True)

    def kappa_at(self, s):
        return self._interp(s, self.kappa)

    def width_at(self, s):
        return self._interp(s, self.w_left), self._interp(s, self.w_right)

    def to_xy(self, s, n=0.0, epsi=0.0):
        """(s, n[, epsi]) -> x, y, yaw."""
        x = self._interp(s, self.xy[:, 0])
        y = self._interp(s, self.xy[:, 1])
        h = self.heading_at(s)
        n = np.broadcast_to(n, x.shape)
        return x - np.sin(h) * n, y + np.cos(h) * n, wrap(h + epsi)

    def straight_at(self, s):
        return self._interp(s, self.straight.astype(float)) > 0.5

    def inside(self, s, n, margin=0.0):
        wl, wr = self.width_at(s)
        return (n <= wl + margin) & (n >= -wr - margin)


class TrackModel:
    """Track (closed) + pit lane (open), oriented in the driving direction, with branch points."""

    def __init__(self, bounds):
        L, R = bounds['track_left'][:, :2], bounds['track_right'][:, :2]
        c = _centreline(L, R, closed=True)
        if (_area(c) > 0) != C.TRACK_DRIVE_CCW:
            c = c[::-1].copy()
        self.track = Lane('track', c, True, L, R)
        PL, PR = bounds['pitlane_left'][:, :2], bounds['pitlane_right'][:, :2]
        pc = _centreline(PL, PR, closed=False)
        # orient the pit lane like the track where they run side by side (map-only rule, no GT)
        s_, n_ = self.track.project(pc[:, 0], pc[:, 1])
        near = np.abs(n_) < 40
        dp = np.gradient(pc, axis=0)
        cos = np.cos(np.arctan2(dp[:, 1], dp[:, 0]) - self.track.heading_at(s_))
        if np.sum(cos[near]) < 0:
            pc = pc[::-1].copy()
        self.pit = Lane('pit', pc, False, PL, PR)
        self.lanes = {'track': self.track, 'pit': self.pit}
        # branch points: track s nearest to the pit-lane entry (pit s = 0) and exit (pit s = end)
        ent = self.track.project(*pc[0])
        ext = self.track.project(*pc[-1])
        self.pit_entry_s_track, self.pit_entry_offset = float(ent[0][0]), float(ent[1][0])
        self.pit_exit_s_track, self.pit_exit_offset = float(ext[0][0]), float(ext[1][0])

    def locate(self, x, y, yaw=None, margin=0.5):
        """Which lane each point is in. Track wins where the two overlap. Returns lane name array plus
        (s, n[, epsi]) on that lane."""
        st = self.track.project(x, y, yaw)
        sp = self.pit.project(x, y, yaw)
        on_track = self.track.inside(st[0], st[1], margin)
        on_pit = self.pit.inside(sp[0], sp[1], margin) & ~on_track
        # outside both: assign the lane with the smaller excess over its half-width
        ex_t = np.maximum(np.abs(st[1]) - np.where(st[1] > 0, *self.track.width_at(st[0])), 0)
        ex_p = np.maximum(np.abs(sp[1]) - np.where(sp[1] > 0, *self.pit.width_at(sp[0])), 0)
        pit = on_pit | (~on_track & ~on_pit & (ex_p < ex_t))
        lane = np.where(pit, 'pit', 'track')
        out = [np.where(pit, a, b) for a, b in zip(sp, st)]
        return (lane, *out)

    def summary(self):
        return {
            'track_length_m': self.track.length, 'pit_length_m': self.pit.length,
            'track_width_median_m': float(np.median(self.track.w_left + self.track.w_right)),
            'pit_width_median_m': float(np.median(self.pit.w_left + self.pit.w_right)),
            'pit_entry_s_track': self.pit_entry_s_track, 'pit_exit_s_track': self.pit_exit_s_track,
            'pit_entry_offset_n': self.pit_entry_offset, 'pit_exit_offset_n': self.pit_exit_offset,
        }
