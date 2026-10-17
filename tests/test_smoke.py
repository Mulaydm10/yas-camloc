"""Smoke tests for the D1 pipeline. Synthetic tests always run; dataset tests skip without the MCAP."""
import re
from pathlib import Path

import numpy as np
import pytest

from camloc import config as C
from camloc.data import CausalMerger
from camloc.track import Lane, wrap

HAVE_DATA = C.DEFAULT_MCAP.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason='dataset not downloaded (see analysis/README.md)')


# ------------------------------------------------------------------ synthetic

def test_causal_merger_restores_stamp_order():
    rng = np.random.default_rng(0)
    stamps = np.sort(rng.integers(0, 10**10, 2000))
    log = stamps + rng.integers(0, 100_000_000, len(stamps))       # latency up to 100 ms
    order = np.argsort(log, kind='stable')
    m = CausalMerger(lag_ns=150_000_000)
    out = []
    for i in order:
        out += [s for s, _ in m.push(int(log[i]), int(stamps[i]), i)]
    out += [s for s, _ in m.flush()]
    assert m.late_drops == 0
    assert out == sorted(stamps.tolist())


def test_causal_merger_drops_too_late_instead_of_reordering():
    m = CausalMerger(lag_ns=10)
    m.push(100, 50, 'a')                 # released (50 <= 90)
    assert m.push(200, 40, 'late') == []  # stamp before already-released time: dropped
    assert m.late_drops == 1


def _circle_lane(r=100.0, n=628):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[r * np.cos(a), r * np.sin(a)]
    return Lane('c', xy, True, xy * (1 + 6 / r), xy * (1 - 6 / r))


def test_lane_roundtrip_on_circle():
    lane = _circle_lane()
    s = np.linspace(0, lane.length - 1, 50)
    n = np.linspace(-5, 5, 50)
    x, y, h = lane.to_xy(s, n, 0.1)
    s2, n2, e2 = lane.project(x, y, h)
    assert np.max(np.abs(wrap((s2 - s) / lane.length * 2 * np.pi))) * lane.length / (2 * np.pi) < 0.1
    assert np.max(np.abs(n2 - n)) < 0.02
    assert np.max(np.abs(e2 - 0.1)) < 0.01
    # CCW circle: left border (outer ring here) is on the left of travel -> left half-width ~ 6 m
    assert abs(np.median(lane.w_left) - 6) < 0.1 and abs(np.median(lane.w_right) - 6) < 0.1


# ------------------------------------------------------------------ GT isolation

def test_inference_modules_do_not_import_groundtruth():
    pkg = Path(__file__).resolve().parent.parent / 'camloc'
    eval_only = {'groundtruth.py', 'evaluate.py', 'calibrate_odometry.py'}   # evaluation / training-time only
    for f in pkg.rglob('*.py'):
        if f.name in eval_only:
            continue
        src = f.read_text()
        assert not re.search(r'\bgroundtruth\b|\bevaluate\b', re.sub(r'""".*?"""|#.*', '', src, flags=re.S)), f.name


def test_forbidden_state_fields_not_whitelisted():
    forbidden = {'x_m', 'y_m', 'z_m', 'yaw_rad', 'roll_rad', 'pitch_rad', 'yaw_vel_rad', 'pos_accuracy', 'se_status'}
    exposed = set(C.STATE_ALLOWED) | set(C.STATE_AMBIGUOUS)
    assert not exposed & forbidden
    assert not any(k.startswith('sn_') for k in exposed)


# ------------------------------------------------------------------ dataset

@needs_data
def test_replay_window_is_causal_and_complete():
    from camloc.data import FrameSet, Replay, StateEvent
    ev = list(Replay(start_s=600, duration_s=3))
    st = [e.stamp_ns for e in ev]
    assert all(np.diff(st) >= 0)
    fs = [e for e in ev if isinstance(e, FrameSet)]
    assert 28 <= len(fs) <= 31 and all(len(f.images) == 7 for f in fs)
    assert all(f.stamp_ns == max(f.stamps.values()) for f in fs)
    states = [e for e in ev if isinstance(e, StateEvent)]
    assert set(states[0].fields) == set(C.STATE_ALLOWED)


@needs_data
def test_track_model_and_pit_branches():
    from camloc.data import load_static
    from camloc.track import TrackModel
    cams, bounds = load_static()
    assert set(cams) == set(C.CAMS)
    tm = TrackModel(bounds)
    assert 2990 < tm.track.length < 3020                      # REPORT.md: 3006 m
    assert 1100 < tm.pit.length < 1140
    assert abs(tm.pit_entry_offset) < 3 and abs(tm.pit_exit_offset) < 3   # pit ends sit on the track


@needs_data
def test_end_to_end_hold_matches_official_evaluator(tmp_path):
    from camloc.evaluate import evaluate, load_sub, official
    from camloc.groundtruth import load_gt
    from camloc.run import run
    from camloc.data import load_static
    from camloc.track import TrackModel
    out = tmp_path / 'sub.csv'
    info = run(C.DEFAULT_MCAP, out, 'hold', start_s=0, duration_s=20)
    assert info['late_drops'] == 0 and 195 <= info['poses'] <= 201
    gt = load_gt()
    track = TrackModel(load_static(cams=())[1])
    sub = load_sub(out)
    w = [(int(sub['t'][0]), int(sub['t'][-1]))]
    res = evaluate(sub, gt, track, w)
    off = official({k: v[(gt['t'] >= w[0][0]) & (gt['t'] <= w[0][1])] for k, v in gt.items()}, sub)
    m = re.search(r'position RMSE\s+\[m\]\s+([\d.]+)', off)
    assert abs(float(m.group(1)) - res['all']['pos_rmse']) < 1e-3


# ------------------------------------------------------------------ D2: odometry + filter + masks

def test_ekf_straight_line_and_covariance_growth():
    from camloc.filter import PoseEKF
    q = {'sigma_v_frac': 0.04, 'sigma_wz': 0.01, 'sigma_v_abs': 0.2}
    f = PoseEKF(0, 0, 0, q)
    for _ in range(100):
        f.predict(0.01, 20.0, 0.0)
    assert abs(f.X[0] - 20.0) < 1e-9 and abs(f.X[1]) < 1e-9
    assert f.P[1, 1] > f.P[0, 0] * 0.01 and np.all(np.linalg.eigvalsh(f.P) > 0)


def test_lane_constraint_pulls_back_inside_on_a_straight():
    from camloc.filter import PoseEKF
    from camloc.track import TrackModel
    x = np.linspace(0, 2000, 2001)
    bounds = {'track_left': np.c_[x, np.full_like(x, 6.0)], 'track_right': np.c_[x, np.full_like(x, -6.0)],
              'pitlane_left': np.c_[x[:500], np.full(500, -40.0)], 'pitlane_right': np.c_[x[:500], np.full(500, -44.0)]}
    # an open straight "track": close the ring far away so TrackModel accepts it
    ring = lambda p, r: np.vstack([p, p[::-1] + [0, r]])
    bounds['track_left'], bounds['track_right'] = ring(bounds['track_left'], 200), ring(bounds['track_right'], 220)
    tm = TrackModel(bounds)
    lane = tm.track
    s0 = 1000.0
    xx, yy, hh = lane.to_xy(s0, 9.0)            # 3 m outside a 6 m half-width (driving direction irrelevant)
    f = PoseEKF(xx[0], yy[0], hh[0] + np.radians(4), {'sigma_v_frac': 0.04, 'sigma_wz': 0.01, 'sigma_v_abs': 0.2},
                p0=(3.0, 3.0, np.radians(5)))
    name, applied = f.constrain_to_lane(tm)
    assert applied and name == 'track'
    _, n, e = lane.project(f.X[0], f.X[1], f.X[2])
    assert abs(n[0]) < 7.5 and abs(np.degrees(e[0])) < 4


def test_masks_cover_expected_regions():
    from camloc.tools.make_masks import load_masks
    m = load_masks()
    assert m['r'].all() and m['rl'].all()                         # no body visible
    for c in ('fr', 'cl', 'cr', 'rr'):
        assert not m[c][-1].any() and m[c][0, 600:900].all()      # bottom row = body, top-centre = scene


@needs_data
def test_odometry_params_and_streaming_matches_batch():
    from camloc.odometry import Odometry, batch
    import json
    p = json.loads((C.ROOT / 'camloc' / 'assets' / 'odometry.json').read_text())
    assert 0.29 < p['r'] < 0.32 and 3.0 < p['L'] < 4.5
    from camloc.data import Replay, StateEvent
    ev = [e for e in Replay(cams=(), start_s=600, duration_s=2) if isinstance(e, StateEvent)]
    o = Odometry(p)
    vs, ws = [], []
    for e in ev:
        o.update(e.stamp_ns, e.fields)
        vs.append(o.v); ws.append(o.wz)
    t = np.array([e.stamp_ns for e in ev])
    F = {k: np.array([e.fields[k] for e in ev]) for k in ev[0].fields}
    v, wz = batch(t, F, p)
    assert np.allclose(vs, v) and np.max(np.abs(np.array(ws)[60:] - wz[60:])) < 0.02
