import argparse

import numpy as np


DEFAULT_KEYS = [
    "rgbd_embed",
    "goal_embed",
    "noise_pred_steps",
    "action_outputs",
    "critic_values",
    "all_trajectory",
    "selected_trajectory",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Jetson NavDP parity outputs against expected tensors.")
    parser.add_argument("--expected", required=True, help="Expected NPZ from make_navdp_jetson_parity_bundle.py.")
    parser.add_argument("--actual", required=True, help="Actual NPZ produced by the Jetson runtime.")
    parser.add_argument("--keys", nargs="*", default=DEFAULT_KEYS, help="NPZ keys to compare.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    expected = np.load(args.expected)
    actual = np.load(args.actual)

    failed = False
    for key in args.keys:
        if key not in expected:
            raise KeyError(f"{key} missing from expected NPZ")
        if key not in actual:
            raise KeyError(f"{key} missing from actual NPZ")

        exp = expected[key]
        act = actual[key]
        if exp.shape != act.shape:
            print(f"FAIL {key}: shape expected {exp.shape}, actual {act.shape}")
            failed = True
            continue

        diff = np.abs(exp.astype(np.float32) - act.astype(np.float32))
        max_abs = float(diff.max()) if diff.size else 0.0
        denom = np.maximum(np.abs(exp.astype(np.float32)), args.atol)
        max_rel = float((diff / denom).max()) if diff.size else 0.0
        ok = np.allclose(exp, act, rtol=args.rtol, atol=args.atol)
        print(f"{'PASS' if ok else 'FAIL'} {key}: max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        failed = failed or not ok

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
