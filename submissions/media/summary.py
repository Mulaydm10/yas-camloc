"""Print the key fields of a camloc.run JSON summary (stdin)."""
import json
import sys

d = json.load(sys.stdin)
s = d['estimator_stats']
print(f"frames {d['frames']}   poses {d['poses']}   ms/frame mean {d['ms_per_frame_mean']:.1f}   p95 {d['ms_per_frame_p95']:.1f}")
print(f"lock: {s['global_init_lock_kind']} at {s['global_init_lock_t_s']} s   confirmed {s['lock_confirms']}   switches {s['lock_revokes']}")
print(f"map fixes accepted {s['map_fixes']}   rejected {s['map_rejected']}")
