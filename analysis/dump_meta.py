"""One streaming pass over all non-image topics of the MCAP.

Writes to analysis/out/:
  meta.json        topics (count, rate, type), camera intrinsics + extrinsics, track-bound stats
  gt.npy / gt.csv  /tf map -> constructor0/base_link: t_ns, x, y, z, yaw
  state_est.npz    every numeric field of /constructor0/state_estimation (keys, t_ns, matrix)
  state_est_sample.txt  one full message as text
  track_bounds.npz polylines track_left/right, pitlane_left/right (N x 3)
  image_index.npz  per camera: header stamp + log time of every image (cheap, no decode)
"""
import csv
import json
import time
from collections import defaultdict

import numpy as np

from common import CAMS, INFO_TOPIC, MCAP, OUT, quat_to_R, quat_yaw, reader, stamp_ns

SE = '/constructor0/state_estimation'


def flatten(obj, prefix=''):
    """Yield (name, float) for every numeric/bool leaf of a decoded ROS message."""
    if isinstance(obj, bool):
        yield prefix, float(obj)
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten(v, f'{prefix}[{i}]')
    elif isinstance(obj, (str, bytes)):
        return
    else:
        slots = getattr(obj, '__slots__', None) or list(vars(obj).keys())
        for k in slots:
            yield from flatten(getattr(obj, k), f'{prefix}.{k}' if prefix else k)


def main():
    t0 = time.time()
    meta = {'topics': {}}
    counts, first_log, last_log, types = defaultdict(int), {}, {}, {}
    gt, se_rows, se_keys = [], [], None
    extr, intr, bounds = {}, {}, {}
    intr_changes = defaultdict(int)
    se_sample = None
    img_idx = defaultdict(list)

    with open(MCAP, 'rb') as f:
        r = reader(f)
        summ = r.get_summary()
        for ch in summ.channels.values():
            types[ch.topic] = summ.schemas[ch.schema_id].name
        # images: index only (header stamp needs decode of a tiny header -> use raw iter + cheap decode)
        for sch, ch, m, msg in r.iter_decoded_messages(
                topics=[t for t in types if not t.endswith('compressed_image')]):
            tp = ch.topic
            counts[tp] += 1
            first_log.setdefault(tp, m.log_time)
            last_log[tp] = m.log_time
            if tp == '/tf':
                for t in msg.transforms:
                    if t.header.frame_id == 'map' and t.child_frame_id == 'constructor0/base_link':
                        p = t.transform.translation
                        gt.append((stamp_ns(t.header), p.x, p.y, p.z, quat_yaw(t.transform.rotation)))
            elif tp == SE:
                kv = list(flatten(msg))
                if se_keys is None:
                    se_keys = ['header_stamp_ns', 'log_time_ns'] + [k for k, _ in kv]
                    se_sample = repr(msg)
                se_rows.append([stamp_ns(msg.header), m.log_time] + [v for _, v in kv])
            elif tp == '/tf_static' and not extr:
                for t in msg.transforms:
                    p, q = t.transform.translation, t.transform.rotation
                    extr[t.child_frame_id] = {
                        'parent': t.header.frame_id, 't': [p.x, p.y, p.z], 'q_xyzw': [q.x, q.y, q.z, q.w]}
            elif tp == '/constructor0/track_bounds' and not bounds:
                for mk in msg.markers:
                    bounds[mk.ns] = np.array([[p.x, p.y, p.z] for p in mk.points])
            elif tp.endswith('camera_info'):
                cam = tp.split('camera_')[1].split('/')[0]
                cur = {'K': list(msg.k), 'D': list(msg.d), 'P': list(msg.p), 'width': msg.width,
                       'height': msg.height, 'model': msg.distortion_model, 'frame': msg.header.frame_id}
                if cam in intr and (intr[cam]['K'] != cur['K'] or intr[cam]['D'] != cur['D']):
                    intr_changes[cam] += 1
                intr.setdefault(cam, cur)
        print(f'meta pass {time.time() - t0:.0f}s')

        # image stamps: separate pass with raw messages, decoding only the header (first bytes of CDR)
        t1 = time.time()
        for sch, ch, m in r.iter_messages(topics=[t for t in types if t.endswith('compressed_image')]):
            d = m.data  # CDR: 4-byte encapsulation, then int32 sec, uint32 nanosec
            sec = int.from_bytes(d[4:8], 'little', signed=True)
            nsec = int.from_bytes(d[8:12], 'little')
            cam = ch.topic.split('camera_')[1].split('/')[0]
            img_idx[cam].append((sec * 10**9 + nsec, m.log_time, len(d)))
            tp = ch.topic
            counts[tp] += 1
            first_log.setdefault(tp, m.log_time)
            last_log[tp] = m.log_time
        print(f'image index pass {time.time() - t1:.0f}s')

    gt = np.array(gt)
    np.save(OUT / 'gt.npy', gt)
    with open(OUT / 'gt.csv', 'w', newline='') as o:
        w = csv.writer(o)
        w.writerow(['timestamp_ns', 'x', 'y', 'z', 'yaw'])
        for row in gt:
            w.writerow([int(row[0]), f'{row[1]:.4f}', f'{row[2]:.4f}', f'{row[3]:.4f}', f'{row[4]:.6f}'])
    se = np.array(se_rows, dtype=np.float64)
    se_t = np.array([r_[0] for r_ in se_rows], dtype=np.int64)
    np.savez(OUT / 'state_est.npz', keys=np.array(se_keys), t_ns=se_t, data=se)
    (OUT / 'state_est_sample.txt').write_text(se_sample)
    np.savez(OUT / 'track_bounds.npz', **bounds)
    np.savez(OUT / 'image_index.npz', **{c: np.array(v, dtype=np.int64) for c, v in img_idx.items()})

    for tp in sorted(types):
        dur = (last_log[tp] - first_log[tp]) / 1e9 if tp in last_log else 0
        meta['topics'][tp] = {'type': types[tp], 'count': counts[tp],
                              'rate_hz': round((counts[tp] - 1) / dur, 3) if dur else None}
    cams = {}
    for c in CAMS:
        e = extr.get(f'constructor0/camera_{c}')
        R = quat_to_R(*e['q_xyzw'])
        # optical z axis (forward) expressed in base_link -> viewing direction
        z = R[:, 2]
        e['view_yaw_deg'] = float(np.degrees(np.arctan2(z[1], z[0])))
        e['view_pitch_deg'] = float(np.degrees(np.arcsin(-z[2])))  # + = looking down
        cams[c] = {'intrinsics': intr[c], 'extrinsics': e, 'camera_info_changes': intr_changes[c]}
    meta['cameras'] = cams
    meta['tf_static_frames'] = {k: v['parent'] for k, v in extr.items()}
    meta['track_bounds'] = {k: {'n_points': len(v)} for k, v in bounds.items()}
    meta['gt'] = {'n': len(gt), 't0_ns': int(gt[0, 0]), 't1_ns': int(gt[-1, 0])}
    meta['state_est'] = {'n': len(se), 'n_fields': len(se_keys) - 2}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(f'done {time.time() - t0:.0f}s: gt {len(gt)}, state_est {se.shape}, bounds {list(bounds)}')


if __name__ == '__main__':
    main()
