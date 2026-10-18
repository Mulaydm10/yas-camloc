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
    eval_only = {'groundtruth.py', 'evaluate.py',                                     # evaluation only
                 'calibrate_odometry.py', 'calibrate_cameras.py', 'overlay_borders.py',   # training-time only
                 'fit_road_attitude.py', 'vo_rotation_check.py', 'lateral_sensitivity.py', 'vis_heading_check.py', 'build_map.py', 'vmap_check.py', 'plot_traj.py', 'holdout.py', 'fit_camera_height.py', 'blur_vs_speed.py', 'exposure_fit.py'}
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


# ------------------------------------------------------------------ D3: geometry, road model, line measurement

def _fake_cam(fx=700.0, w=1506, h=728):
    from camloc.data import Camera
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical z -> base x (looking forward)
    T[:3, 3] = (0, 0, 1.0)
    return Camera('t', K, np.zeros(5), w, h, T)


def test_projection_of_point_ahead_hits_principal_row_offset():
    from camloc.geometry import T_map_base, project
    cam = _fake_cam()
    uv, idx, depth = project(cam, T_map_base(0, 0, 0, 0, 0, 0), np.array([[10.0, 0, 0.0]]))
    assert len(uv) == 1 and abs(uv[0, 0] - 753) < 1e-6
    assert abs(uv[0, 1] - (364 + 700 * 1.0 / 10)) < 1e-6 and abs(depth[0] - 10) < 1e-9   # 1 m below camera at 10 m


def test_vertical_residual_finds_a_drawn_line():
    import cv2
    from camloc.geometry import T_map_base
    from camloc.measure import tophat, vertical_residuals
    cam = _fake_cam()
    T = T_map_base(0, 0, 0, 0, 0, 0)
    pts = np.c_[np.full(20, 10.0), np.linspace(-3, 3, 20), np.zeros(20)]     # a lateral line 10 m ahead
    img = np.full((364, 753, 3), 60, np.uint8)
    v_true = (364 + 70) * 0.5 + 3                                              # 3 px (half-res) below prediction
    cv2.line(img, (0, int(v_true)), (752, int(v_true)), (220, 220, 220), 1)
    r = vertical_residuals(cam, tophat(img), np.ones((364, 753), bool), T, pts)
    assert len(r) >= 15 and abs(np.median(r[:, 2]) - 3) < 1.0


@needs_data
def test_road_attitude_is_small_and_finite():
    from camloc.data import load_bounds3d
    from camloc.road import Road3D
    road = Road3D(load_bounds3d())
    z, roll, pitch = road.attitude(7.17, -9.91, 0.126)          # first pose of the training run
    assert np.isfinite([z, roll, pitch]).all() and abs(roll) < 0.1 and abs(pitch) < 0.1


# ------------------------------------------------------------------ D4: heading clone + relative yaw

def test_relative_yaw_update_corrects_heading_drift():
    from camloc.filter import PoseEKF
    q = {'sigma_v_frac': 0.04, 'sigma_wz': 0.01, 'sigma_v_abs': 0.2}
    f = PoseEKF(0, 0, 0, q)
    f.clone_heading()
    for _ in range(100):                      # odometry says +0.1 rad/s for 1 s, but truth turned 0.05 rad
        f.predict(0.01, 10.0, 0.1)
    f.update_relative_yaw(0.05, np.radians(0.1))
    assert abs(f.X[2] - 0.05) < 0.01 and abs(f.X[3]) < 1e-6
    assert np.all(np.linalg.eigvalsh(f.P) > -1e-12)


def test_heading_aware_projection_rejects_opposite_leg():
    from camloc.track import Lane
    x = np.linspace(0, 200, 201)
    # hairpin-like: two parallel legs 10 m apart, driven in opposite directions
    xy = np.vstack([np.c_[x, np.zeros_like(x)], np.c_[x[::-1], np.full_like(x, 10.0)]])
    lane = Lane('h', xy, True, xy + [0, -3], xy + [0, 3])
    s_east = lane.project(100.0, 6.0, 0.0)[0]         # nearer the westbound leg but heading east
    s_west = lane.project(100.0, 6.0, np.pi)[0]
    assert s_east[0] < 200 < s_west[0]


def test_pose_update_pulls_toward_fix_and_shrinks_covariance():
    from camloc.filter import PoseEKF
    q = {'sigma_v_frac': 0.04, 'sigma_wz': 0.01, 'sigma_v_abs': 0.2}
    f = PoseEKF(0, 0, 0, q, p0=(2.0, 2.0, np.radians(2)))
    P0 = f.P.copy()
    nis = f.update_pose(1.0, -1.0, np.radians(1), 0.5, np.radians(0.5))
    assert 0.8 < f.X[0] < 1.0 and -1.0 < f.X[1] < -0.8          # weighted toward the more certain fix
    assert np.all(np.diag(f.P)[:3] < np.diag(P0)[:3]) and np.all(np.linalg.eigvalsh(f.P) > -1e-12)   # PSD (clone starts fully correlated)
    assert nis > 0
    # a fix across the +-pi seam moves the heading the short way round
    f2 = PoseEKF(0, 0, np.radians(179), q)
    f2.update_pose(0, 0, np.radians(-179), 0.5, np.radians(0.5))
    assert abs(wrap(f2.X[2] - np.pi)) < np.radians(1.5)


@needs_data
def test_run_parameter_overrides(tmp_path):
    from camloc.run import run
    with pytest.raises(SystemExit):
        run(C.DEFAULT_MCAP, tmp_path / 'x.csv', 'hold', duration_s=0.5, overrides={'NO_SUCH': '1'})


def test_holdout_is_a_noop_without_excluded_laps_and_augment_keeps_shape():
    from camloc.tools import holdout
    from camloc.tools.vmap_check import augment
    if not C.EXCLUDE_LAPS:
        assert holdout.excluded(123) is False and not holdout.excluded(np.arange(5)).any()
    g = (np.arange(64 * 48) % 256).astype(np.uint8).reshape(48, 64)
    for a in ('none', 'gamma0.5', 'gamma2', 'contrast0.5', 'blur9', 'noise10', 'day'):
        out = augment(g, a)
        assert out.shape == g.shape and out.dtype == np.uint8
    assert augment(g, 'gamma0.5').mean() > g.mean() > augment(g, 'gamma2').mean()


def test_slip_state_learns_a_constant_sideslip_from_position_fixes():
    from camloc.filter import PoseEKF
    q = {'sigma_v_frac': 0.02, 'sigma_wz': 0.005, 'sigma_v_abs': 0.1}
    true_beta, v, dt = np.radians(1.5), 20.0, 0.1
    f = PoseEKF(0, 0, 0, q, slip={'sigma0': np.radians(1.0), 'q': np.radians(0.05), 'tau': 1e6})
    x = y = 0.0
    for _ in range(300):
        x += v * np.cos(true_beta) * dt
        y += v * np.sin(true_beta) * dt
        f.predict(dt, v, 0.0)
        f.update_pose(x, y, 0.0, 0.3, np.radians(0.2))
    assert abs(f.X[4] - true_beta) < np.radians(0.3)
    assert f.P.shape == (5, 5) and np.all(np.linalg.eigvalsh(f.P) > -1e-12)


@needs_data
def test_overrides_reach_the_constructor(tmp_path):
    """--set must act before __init__ (the EKF and its slip state are built there)."""
    from camloc import run as R
    seen = {}
    orig = R.PoseEKF.__init__

    def spy(self, *a, **kw):
        seen['slip'] = kw.get('slip')
        orig(self, *a, **kw)
    R.PoseEKF.__init__ = spy
    try:
        R.run(C.DEFAULT_MCAP, tmp_path / 'x.csv', 'odom', duration_s=0.3, overrides={'SLIP_STATE': '1'})
    finally:
        R.PoseEKF.__init__ = orig
    assert seen['slip'] is not None


def test_scale_state_learns_a_wheel_speed_bias():
    from camloc.filter import PoseEKF
    q = {'sigma_v_frac': 0.02, 'sigma_wz': 0.005, 'sigma_v_abs': 0.1}
    f = PoseEKF(0, 0, 0, q, scale={'sigma0': 0.02, 'q': 1e-4, 'tau': 1e6, 'clip': 0.1})
    x = 0.0
    for _ in range(300):
        x += 20.0 * 1.03 * 0.1          # true speed 3 % above the wheel speed
        f.predict(0.1, 20.0, 0.0)
        f.update_pose(x, 0.0, 0.0, 0.3, np.radians(0.2))
    assert abs(f.X[f.ik] - 0.03) < 0.005


@needs_data
def test_global_init_without_any_ground_truth_pose(tmp_path, monkeypatch):
    """TASK I: the test file gives no starting pose. vismap must lock by itself; /tf map->base_link is never read."""
    from camloc import run as R
    vmap = C.ASSETS / 'vmap.npz'
    if not vmap.exists():
        pytest.skip('visual map not built (run.sh builds it)')

    def no_gt(*a, **k):
        raise AssertionError('initial_pose (GT /tf) must not be read at inference')
    monkeypatch.setattr(R, 'initial_pose', no_gt)
    stats = R.run(C.DEFAULT_MCAP, tmp_path / 'gi.csv', 'vismap', start_s=640, duration_s=3)
    assert stats['estimator_stats']['global_init_locks'] == 1
    assert stats['estimator_stats']['global_init_lock_t_s'] < 2.0
    assert stats['poses'] == stats['frames']


def test_twist_odometry_uses_velocity_and_sideslip():
    from camloc.odometry import Odometry
    o = Odometry(params={'q': {}}, source='wheels')
    o.update(0, {'twist_vx': 30.0, 'twist_vy': 0.5, 'twist_wz': 0.1})
    assert abs(o.v - np.hypot(30, 0.5)) < 1e-9 and o.wz == 0.1 and abs(o.beta_meas - np.arctan2(0.5, 30)) < 1e-12
