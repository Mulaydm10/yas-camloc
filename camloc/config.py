"""Paths, topic names and the car-state field policy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MCAP = ROOT / 'data' / 'c0_2026-10-13_22-53-42_camloc' / 'c0_2026-10-13_22-53-42_camloc_0.mcap'
CACHE = ROOT / 'cache'

CAMS = ('fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr')
IMG_TOPIC = '/constructor0/sensor/camera_{}/compressed_image'
INFO_TOPIC = '/constructor0/sensor/camera_{}/camera_info'
STATE_TOPIC = '/constructor0/state_estimation'
TF_TOPIC = '/tf'
TF_STATIC_TOPIC = '/tf_static'
BOUNDS_TOPIC = '/constructor0/track_bounds'
BASE_FRAME = 'constructor0/base_link'

# Car-state policy (REPORT.md section 5). Pose fields and anything computed from the pose are never exposed.
# ALLOWED: raw vehicle signals, independent of the pose estimate.
STATE_ALLOWED = (
    'omega_w_fl', 'omega_w_fr', 'omega_w_rl', 'omega_w_rr', 'valid_wheelsspeeds_b',
    'delta_wheel_rad', 'gas', 'brake', 'rpm', 'gear',
    'cba_actual_pressure_fl_pa', 'cba_actual_pressure_fr_pa', 'cba_actual_pressure_rl_pa', 'cba_actual_pressure_rr_pa',
)
# AMBIGUOUS: fused estimator outputs. The PDF allows car state; the README calls the topic GT.
# Off by default until the organisers answer (REPORT.md section 9, Q1).
STATE_AMBIGUOUS = (
    'v_mps', 'vx_mps', 'vy_mps', 'v_raw_mps', 'beta_rad', 'dbeta_radps',
    'wx_radps', 'wy_radps', 'wz_radps', 'ax_mps2', 'ay_mps2', 'az_mps2', 'ax_vel_mps2', 'ay_vel_mps2',
    'kappa_radpm', 'alpha_fl_rad', 'alpha_fr_rad', 'alpha_rl_rad', 'alpha_rr_rad', 'diff_fr_alpha_rad',
    'lambda_fl_perc', 'lambda_fr_perc', 'lambda_rl_perc', 'lambda_rr_perc',
)
# FORBIDDEN (never read by inference code): x_m, y_m, z_m, roll/pitch/yaw_rad, yaw_vel_rad, pos_accuracy,
# se_status, se_state, sn_map_state.*

# Track map conventions measured on the training run (map knowledge, not GT at inference):
TRACK_DRIVE_CCW = True      # centreline traversed counter-clockwise in `map`
