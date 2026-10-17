"""Causal replay of the MCAP plus static inputs (calibration, track bounds, initial pose).

`Replay` streams the bag and yields events strictly in header-stamp order:
  StateEvent(stamp_ns, fields)   only the whitelisted car-state fields (config.STATE_ALLOWED [+ AMBIGUOUS])
  FrameSet(stamp_ns, images)     all camera images of one exposure (JPEG bytes, decode lazily)
A FrameSet at stamp T is yielded only after every state sample with stamp <= T, and before any with
stamp > T. The bag is in log-time order and log time >= stamp (median transport latency 42 ms, max
99 ms on the training run), so a heap released at `log_time - lag` restores stamp order exactly as long
as latency < lag. Anything arriving later than that is dropped and counted (`late_drops`), never
delivered out of order.
"""
import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C


def _reader(f):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    return make_reader(f, decoder_factories=[DecoderFactory()])


def _stamp(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


@dataclass
class StateEvent:
    stamp_ns: int
    fields: dict


@dataclass
class FrameSet:
    stamp_ns: int                       # the LATEST camera stamp in the set (strictly causal pose stamp)
    images: dict                        # cam -> JPEG bytes
    stamps: dict = field(default_factory=dict)  # cam -> header stamp

    def decode(self, cam, flags=None):
        import cv2
        buf = np.frombuffer(self.images[cam], np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR if flags is None else flags)


class CausalMerger:
    """Re-orders (log_time, stamp, item) arrivals into stamp order with a bounded lag."""

    def __init__(self, lag_ns):
        self.lag_ns = lag_ns
        self.heap = []
        self.seq = itertools.count()
        self.released_until = -math.inf
        self.late_drops = 0

    def push(self, log_ns, stamp_ns, item):
        if stamp_ns < self.released_until:
            self.late_drops += 1
            return []
        heapq.heappush(self.heap, (stamp_ns, next(self.seq), item))
        return self.pop_until(log_ns - self.lag_ns)

    def pop_until(self, t_ns):
        out = []
        while self.heap and self.heap[0][0] <= t_ns:
            st, _, it = heapq.heappop(self.heap)
            self.released_until = st
            out.append((st, it))
        return out

    def flush(self):
        return self.pop_until(math.inf)


class Replay:
    """Iterate StateEvent / FrameSet in stamp order. `start_s`/`duration_s` are relative to the first image."""

    def __init__(self, mcap_path=C.DEFAULT_MCAP, cams=C.CAMS, state='allowed', start_s=0.0, duration_s=None,
                 lag_ms=250.0, set_tol_ms=5.0):
        assert state in ('allowed', 'ambiguous', None)
        self.path = str(mcap_path)
        self.cams = tuple(cams)
        self.fields = () if state is None else C.STATE_ALLOWED + (C.STATE_AMBIGUOUS if state == 'ambiguous' else ())
        self.start_s, self.duration_s = start_s, duration_s
        self.lag_ns = int(lag_ms * 1e6)
        self.tol_ns = int(set_tol_ms * 1e6)
        self.late_drops = 0
        self.has_state = None

    def __iter__(self):
        topics = [C.IMG_TOPIC.format(c) for c in self.cams]
        cam_of = {C.IMG_TOPIC.format(c): c for c in self.cams}
        merger = CausalMerger(self.lag_ns)
        self._held = []
        with open(self.path, 'rb') as f:
            r = _reader(f)
            summ = r.get_summary()
            all_topics = {ch.topic for ch in summ.channels.values()}
            self.has_state = C.STATE_TOPIC in all_topics and bool(self.fields)
            if self.has_state:
                topics.append(C.STATE_TOPIC)
            t_first = summ.statistics.message_start_time
            start = t_first + int(self.start_s * 1e9)
            end = None if self.duration_s is None else start + int(self.duration_s * 1e9)
            pending = []                                   # image events waiting to be grouped
            for _, ch, m, msg in r.iter_decoded_messages(topics=topics, start_time=start, end_time=end,
                                                         log_time_order=True):
                st = _stamp(msg.header)
                if ch.topic == C.STATE_TOPIC:
                    item = ('state', {k: float(getattr(msg, k)) for k in self.fields})
                else:
                    item = ('img', cam_of[ch.topic], bytes(msg.data))
                for ev in self._emit(merger.push(m.log_time, st, item), pending):
                    yield ev
            for ev in self._emit(merger.flush(), pending, final=True):
                yield ev
        self.late_drops = merger.late_drops

    def _emit(self, released, pending, final=False):
        """Group released images within `tol` of the first into one FrameSet. State samples released while a
        set is open are held and emitted on the correct side of the set's stamp (<= before, > after)."""
        for st, item in released:
            if pending and st - pending[0][0] > self.tol_ns:
                yield from self._close(pending)
            if item[0] == 'img':
                pending.append((st, item[1], item[2]))
            elif pending:
                self._held.append(StateEvent(st, item[1]))
            else:
                yield StateEvent(st, item[1])
        if final:
            if pending:
                yield from self._close(pending)
            yield from self._held
            self._held = []

    def _close(self, pending):
        fs = self._frameset(pending)
        pending.clear()
        yield from (e for e in self._held if e.stamp_ns <= fs.stamp_ns)
        yield fs
        yield from (e for e in self._held if e.stamp_ns > fs.stamp_ns)
        self._held = []

    @staticmethod
    def _frameset(pending):
        images = {c: b for _, c, b in pending}
        stamps = {c: st for st, c, _ in pending}
        return FrameSet(max(stamps.values()), images, stamps)


# ---------------------------------------------------------------- static inputs

def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class Camera:
    name: str
    K: np.ndarray            # 3x3
    D: np.ndarray            # plumb_bob k1 k2 p1 p2 k3
    width: int
    height: int
    T_base_cam: np.ndarray   # 4x4, maps camera-optical-frame points into base_link


def load_static(mcap_path=C.DEFAULT_MCAP, cams=C.CAMS):
    """Camera calibration and track bounds from the latched topics (first copy of each)."""
    cameras, bounds, extr = {}, {}, {}
    want_info = {C.INFO_TOPIC.format(c): c for c in cams}
    with open(mcap_path, 'rb') as f:
        for _, ch, _, msg in _reader(f).iter_decoded_messages(
                topics=list(want_info) + [C.TF_STATIC_TOPIC, C.BOUNDS_TOPIC]):
            if ch.topic == C.TF_STATIC_TOPIC and not extr:
                for t in msg.transforms:
                    if t.header.frame_id == C.BASE_FRAME:
                        T = np.eye(4)
                        q, p = t.transform.rotation, t.transform.translation
                        T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
                        T[:3, 3] = (p.x, p.y, p.z)
                        extr[t.child_frame_id] = T
            elif ch.topic == C.BOUNDS_TOPIC and not bounds:
                bounds = {mk.ns: np.array([[p.x, p.y] for p in mk.points]) for mk in msg.markers}
            elif ch.topic in want_info and want_info[ch.topic] not in cameras:
                c = want_info[ch.topic]
                cameras[c] = (np.array(msg.k).reshape(3, 3), np.array(msg.d), msg.width, msg.height)
            if extr and bounds and len(cameras) == len(cams):
                break
    cams_out = {c: Camera(c, K, D, w, h, extr[f'constructor0/camera_{c}']) for c, (K, D, w, h) in cameras.items()}
    return cams_out, bounds


def load_bounds3d(mcap_path=C.DEFAULT_MCAP):
    """Track and pit-lane borders as (N, 3) map points (x, y, z), from the latched track_bounds topic."""
    with open(mcap_path, 'rb') as f:
        for _, _, _, msg in _reader(f).iter_decoded_messages(topics=[C.BOUNDS_TOPIC]):
            return {mk.ns: np.array([[p.x, p.y, p.z] for p in mk.points]) for mk in msg.markers}
    raise RuntimeError('no track_bounds message')


def initial_pose(mcap_path=C.DEFAULT_MCAP):
    """The first map -> base_link pose of the run: the ONLY ground truth allowed at inference."""
    with open(mcap_path, 'rb') as f:
        for _, _, _, msg in _reader(f).iter_decoded_messages(topics=[C.TF_TOPIC]):
            for t in msg.transforms:
                if t.header.frame_id == 'map' and t.child_frame_id == C.BASE_FRAME:
                    q, p = t.transform.rotation, t.transform.translation
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    return _stamp(t.header), p.x, p.y, yaw
    raise RuntimeError('no map -> base_link transform found')
