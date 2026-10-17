"""Ground truth for EVALUATION ONLY. Never import this from inference code (enforced by a test).

`load_gt(mcap)` extracts /tf map -> base_link (same as tools/extract_gt.py) plus the estimator's own
horizontal position accuracy from /constructor0/state_estimation, and caches them in cache/gt_eval.npz.
"""
import math

import numpy as np

from . import config as C


def load_gt(mcap_path=C.DEFAULT_MCAP, cache=True):
    path = C.CACHE / 'gt_eval.npz'
    if cache and path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from .data import _reader, _stamp
    tf, acc = [], []
    with open(mcap_path, 'rb') as f:
        for _, ch, _, msg in _reader(f).iter_decoded_messages(topics=[C.TF_TOPIC, C.STATE_TOPIC]):
            if ch.topic == C.TF_TOPIC:
                for t in msg.transforms:
                    if t.header.frame_id == 'map' and t.child_frame_id == C.BASE_FRAME:
                        q, p = t.transform.rotation, t.transform.translation
                        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                        tf.append((_stamp(t.header), p.x, p.y, yaw))
            else:
                acc.append((_stamp(msg.header), math.hypot(msg.pos_accuracy[0], msg.pos_accuracy[1])))
    tf = np.array(tf, dtype=np.float64)
    acc = np.array(acc, dtype=np.float64)
    o = np.argsort(tf[:, 0])
    t = tf[o, 0].astype(np.int64)
    oa = np.argsort(acc[:, 0])
    out = {'t': t, 'x': tf[o, 1], 'y': tf[o, 2], 'yaw': tf[o, 3],
           'pos_acc': np.interp(t, acc[oa, 0], acc[oa, 1])}
    if cache:
        C.CACHE.mkdir(exist_ok=True)
        np.savez(path, **out)
    return out


def write_gt_csv(gt, path):
    with open(path, 'w') as f:
        f.write('timestamp_ns,x,y,z,yaw\n')
        for a, b, c, d in zip(gt['t'], gt['x'], gt['y'], gt['yaw']):
            f.write(f'{int(a)},{b:.4f},{c:.4f},0.0,{d:.6f}\n')
