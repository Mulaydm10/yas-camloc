"""TRAINING-TIME: export the track and pit-lane borders from the training bag to camloc/assets/track_bounds.npz,
for test files that carry no /constructor0/track_bounds topic (TASK I.0). The file is dataset-derived, so it is
git-ignored and built by run.sh like the visual map.

    CAMLOC_MCAP=<training.mcap> python -m camloc.tools.export_bounds
"""
import numpy as np

from .. import config as C
from ..data import BOUNDS_ASSET, load_bounds3d


def main():
    b = load_bounds3d(C.DEFAULT_MCAP)
    out = C.ASSETS / BOUNDS_ASSET
    np.savez_compressed(out, **{f'{k}_3d': v for k, v in b.items()})
    print(f'{sorted(b)} -> {out}')


if __name__ == '__main__':
    main()
