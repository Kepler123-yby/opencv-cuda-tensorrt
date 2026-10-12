#!/usr/bin/env python3
"""Compare two little-endian float32 tensor dumps; no GPU is required.

Exit codes: 0=PASS, 1=numeric mismatch, 2=invalid input.
The first file is the CPU/reference tensor used for relative tolerance.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np


def load_tensor(path, shape):
    path = Path(path)
    expected = math.prod(shape) * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {actual}")
    result = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="CPU/reference float32 .bin file")
    parser.add_argument("actual", help="GPU/actual float32 .bin file")
    parser.add_argument("--shape", nargs="+", type=int, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    if any(dim <= 0 for dim in args.shape):
        parser.error("shape dimensions must be positive")
    if any(not math.isfinite(x) or x < 0 for x in (args.atol, args.rtol)):
        parser.error("atol/rtol must be finite and non-negative")

    try:
        reference = load_tensor(args.reference, args.shape).astype(np.float64)
        actual = load_tensor(args.actual, args.shape).astype(np.float64)
    except (OSError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2

    diff = np.abs(actual - reference)
    passed = diff <= args.atol + args.rtol * np.abs(reference)
    worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
    print(f"shape={tuple(args.shape)}, elements={diff.size}, dtype=float32")
    print(f"max_abs_error={diff.max():.9g}, mean_abs_error={diff.mean():.9g}")
    print(f"worst_index={worst}, reference={reference[worst]:.9g}, "
          f"actual={actual[worst]:.9g}")
    print(f"atol={args.atol:.9g}, rtol={args.rtol:.9g}, "
          f"mismatched={np.count_nonzero(~passed)}")
    if passed.all():
        print("PASS")
        return 0
    first = np.unravel_index(int(np.flatnonzero(~passed)[0]), diff.shape)
    print(f"FAIL: first_mismatch={first}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
