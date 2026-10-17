"""Shared paths and helpers for the analysis scripts."""
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'c0_2026-10-13_22-53-42_camloc'
MCAP = DATA / 'c0_2026-10-13_22-53-42_camloc_0.mcap'
OUT = ROOT / 'analysis' / 'out'          # derived data (gitignored: contains GT)
FIG = ROOT / 'analysis'                  # plots
OUT.mkdir(parents=True, exist_ok=True)

CAMS = ['fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr']
IMG_TOPIC = '/constructor0/sensor/camera_{}/compressed_image'
INFO_TOPIC = '/constructor0/sensor/camera_{}/camera_info'


def stamp_ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def quat_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def reader(f):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    return make_reader(f, decoder_factories=[DecoderFactory()])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
