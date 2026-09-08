#!/usr/bin/env python3
# ---- save as: ~/iris/margin_headroom.py   (run from ~/iris) ----
"""Per-position headroom between the C++/PyTorch margin discrepancy and the
margin itself.

Why: tf_pytorch_all.py measured 0 mismatches on all 160 000 positions, but the
global sufficient condition (min margin > max |delta|) fails on s1 (0.52x) and
s3 (0.92x), because the smallest margin and the largest discrepancy occur at
different positions.  The condition that actually governs a flip is
per-position: |delta(p)| < m(p).  This checks it directly, so the claim in
Section 3.2.2 can be stated as a bound rather than as an observed outcome.

Also reports agreement on the runner-up index a2, which the flip measurement
never uses but the margin definition does: if a2 disagreed anywhere, the two
implementations would be comparing different pairs and the recorded margins
would not be commensurable.
"""
import os

import numpy as np

CPP = os.path.expanduser("~/iris-cpp")
SEEDS = [0, 1, 2, 3, 4]
NOBS, RB = 320, 16
CH = NOBS * RB + 20 * 3 * 4 + 20 * 2 * 4


def read_cpp(path):
    buf = np.fromfile(path, dtype=np.uint8)
    r = buf.reshape(buf.size // CH, CH)[:, :NOBS*RB].reshape(-1, RB)
    return (r[:, 0:4].copy().view(np.int32).ravel(),
            r[:, 4:8].copy().view(np.int32).ravel(),
            r[:, 8:12].copy().view(np.float32).ravel())


print(f"{'seed':6} {'a1 dis':>7} {'a2 dis':>7} {'at risk':>8} "
      f"{'min m-|d|':>11} {'min ratio':>10} {'max |d|':>11}")
tot = dict(n=0, a1=0, a2=0, risk=0)
worst_gap, worst_ratio = np.inf, np.inf
for s in SEEDS:
    ca1, ca2, cm = read_cpp(f"{CPP}/tf2_s{s}_f32.bin")
    pa1 = np.load(f"pt_s{s}_a1.npy")
    pa2 = np.load(f"pt_s{s}_a2.npy")
    pg = np.load(f"pt_s{s}_gap.npy")
    d = np.abs(pg.astype(np.float64) - cm.astype(np.float64))
    m = cm.astype(np.float64)

    # per-position: a flip is possible only where the discrepancy is at least
    # as large as the margin it would have to cross
    risk = int((d >= m).sum())
    slack = (m - d).min()
    ratio = (m / np.maximum(d, np.finfo(np.float64).tiny)).min()
    a1d, a2d = int((pa1 != ca1).sum()), int((pa2 != ca2).sum())

    print(f"s{s:<5} {a1d:7d} {a2d:7d} {risk:8d} {slack:11.3e} "
          f"{ratio:10.2f} {d.max():11.3e}")
    tot["n"] += m.size; tot["a1"] += a1d; tot["a2"] += a2d; tot["risk"] += risk
    worst_gap, worst_ratio = min(worst_gap, slack), min(worst_ratio, ratio)

print(f"\nTOTAL over {tot['n']} positions:")
print(f"  top-1 disagreements   : {tot['a1']}")
print(f"  top-2 disagreements   : {tot['a2']}")
print(f"  positions at risk     : {tot['risk']}   (|delta| >= margin)")
print(f"  smallest margin-|delta| slack : {worst_gap:.3e}")
print(f"  smallest margin/|delta| ratio : {worst_ratio:.2f}x")
if tot["risk"] == 0:
    print("\n  => at every position the discrepancy is strictly smaller than the")
    print("     margin, so no position could have changed its argmax.  The zero")
    print("     mismatch is a consequence of this bound, not a coincidence.")
