#!/usr/bin/env python3
# ---- save as: ~/iris/tf_pytorch_all.py   (run from ~/iris) ----
"""Teacher-forced replay in PyTorch over all five traces, mirroring tf.cpp, to
measure the baseline argmax mismatch rate between the C++ port and the
reference model.

s0 gave 0 / 32000 with min margin 4.48e-5 against max |delta| 2.67e-5 -- a
headroom of only 1.68x at the tightest point, so the other four traces are
worth running: a smaller minimum margin anywhere would change the claim.  s1
has the lowest p1 (0.0199 against s0's 0.0212) and is the most likely to hold
one.  Five traces give 160 000 predictions, the same figure the containment
result in Section 4.4 is stated over.

Protocol copied from tf.cpp / dump_wm.py, not from iris.h:
  * 20 blocks of 17 tokens = 340 per forward; blocks independent, 100 groups.
  * p % 17 == 16 is the action; 0..15 are observation tokens.
  * actions use a SEPARATE embedding table (embedding_tables.0) indexed by the
    raw action 0..3.  The 512+a remapping in iris.h is a GGUF-side detail: the
    export concatenates the two tables into one of 516 rows.
  * observation head read at p % 17 != 15 -> 320 per group.  p = 339 predicts
    into the next frame; tf.cpp records it, so it is kept here for alignment.
  * LayerNorm eps 1e-5 (gn_eps = 1e-6 in iris.h is the tokeniser's GroupNorm).
  * GELU stays erf: that substitution is the effect being measured.

Writes pt_s{N}_{a1,a2,gap}.npy and prints a per-trace summary table.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)          # tf.cpp sets n_threads = 1
torch.set_grad_enabled(False)

CKPT = "checkpoints/last.pt"
CPP = os.path.expanduser("~/iris-cpp")
SEEDS = [0, 1, 2, 3, 4]

TPB, NBLK = 17, 20
T = TPB * NBLK                    # 340
E, NH, HS, NLAYER = 256, 4, 64, 10
OBS_V, EPS = 512, 1e-5
NFRAME = 2000

# Published FP32 margin quantiles (Table C.2), one row per trace: a provenance
# check on the C++ reference file, reproducing a known value rather than
# trusting a filename.
PUBLISHED_Q = {
    0: {1: 0.0212, 25: 0.7558, 50: 2.6327, 95: 10.977},
    1: {1: 0.0199, 25: 0.7554, 50: 2.9052, 95: 10.797},
    2: {1: 0.0207, 25: 0.7645, 50: 2.6871, 95: 11.085},
    3: {1: 0.0219, 25: 0.8068, 50: 2.7302, 95: 10.932},
    4: {1: 0.0234, 25: 0.7696, 50: 2.7893, 95: 10.654},
}

POS = [p for p in range(T) if p % TPB != TPB - 2]
assert len(POS) == 320, len(POS)

PREFIX = "world_model."
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw


def pick(name):
    k = PREFIX + name
    if k not in sd:
        raise SystemExit(f"missing {k}")
    return sd[k].float()


emb_act = pick("embedder.embedding_tables.0.weight")     # (4, 256)
emb_obs = pick("embedder.embedding_tables.1.weight")     # (512, 256)
pos_emb = pick("pos_emb.weight")                         # (340, 256)
assert emb_act.shape == (4, E) and emb_obs.shape == (OBS_V, E)
assert pos_emb.shape[0] >= T, f"pos_emb has {pos_emb.shape[0]} rows, need {T}"

W = {}
for L in range(NLAYER):
    p = f"transformer.blocks.{L}."
    for n in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
              "attn.query.weight", "attn.query.bias",
              "attn.key.weight", "attn.key.bias",
              "attn.value.weight", "attn.value.bias",
              "attn.proj.weight", "attn.proj.bias",
              "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"):
        W[p + n] = pick(p + n)
lnf_w, lnf_b = pick("transformer.ln_f.weight"), pick("transformer.ln_f.bias")
h0w = pick("head_observations.head_module.0.weight")
h0b = pick("head_observations.head_module.0.bias")
h2w = pick("head_observations.head_module.2.weight")
h2b = pick("head_observations.head_module.2.bias")

MASK = torch.tril(torch.ones(T, T)) == 0
IS_ACT = torch.tensor([p % TPB == TPB - 1 for p in range(T)])


def forward(seq):
    """seq: LongTensor (T,) with obs tokens 0..511 and raw actions 0..3."""
    x = torch.empty(1, T, E)
    x[0, ~IS_ACT] = emb_obs[seq[~IS_ACT]]
    x[0, IS_ACT] = emb_act[seq[IS_ACT]]
    x = x + pos_emb[:T]

    for L in range(NLAYER):
        p = f"transformer.blocks.{L}."
        h = F.layer_norm(x, (E,), W[p + "ln1.weight"], W[p + "ln1.bias"], EPS)
        q = F.linear(h, W[p + "attn.query.weight"], W[p + "attn.query.bias"])
        k = F.linear(h, W[p + "attn.key.weight"], W[p + "attn.key.bias"])
        v = F.linear(h, W[p + "attn.value.weight"], W[p + "attn.value.bias"])
        qh = q.view(1, T, NH, HS).transpose(1, 2)
        kh = k.view(1, T, NH, HS).transpose(1, 2)
        vh = v.view(1, T, NH, HS).transpose(1, 2)
        att = (qh @ kh.transpose(-2, -1)) * (1.0 / math.sqrt(HS))
        att = F.softmax(att.masked_fill(MASK, float("-inf")), dim=-1)
        y = (att @ vh).transpose(1, 2).reshape(1, T, E)
        x = x + F.linear(y, W[p + "attn.proj.weight"], W[p + "attn.proj.bias"])

        h2 = F.layer_norm(x, (E,), W[p + "ln2.weight"], W[p + "ln2.bias"], EPS)
        h2 = F.linear(h2, W[p + "mlp.0.weight"], W[p + "mlp.0.bias"])
        h2 = F.gelu(h2)                       # erf form, as in dump_wm.py
        x = x + F.linear(h2, W[p + "mlp.2.weight"], W[p + "mlp.2.bias"])

    x = F.layer_norm(x, (E,), lnf_w, lnf_b, EPS)
    h = F.relu(F.linear(x[0, POS], h0w, h0b))
    return F.linear(h, h2w, h2b)              # (320, 512)


def read_cpp(path):
    NOBS, RB = 320, 16
    CH = NOBS * RB + 20*3*4 + 20*2*4
    buf = np.fromfile(path, dtype=np.uint8)
    nblk = buf.size // CH
    r = buf.reshape(nblk, CH)[:, :NOBS*RB].reshape(nblk*NOBS, RB)
    return (r[:, 0:4].copy().view(np.int32).ravel(),
            r[:, 8:12].copy().view(np.float32).ravel())


results = []
for s in SEEDS:
    print(f"\n=== trace s{s} ===", flush=True)
    tok = np.fromfile(f"{CPP}/trace_s{s}/tokens.bin", dtype=np.int32)
    act = np.fromfile(f"{CPP}/trace_s{s}/actions.bin", dtype=np.int32)
    assert tok.size == NFRAME * 16 and act.size == NFRAME, (tok.size, act.size)
    assert 0 <= tok.min() and tok.max() < OBS_V, (tok.min(), tok.max())
    assert 0 <= act.min() and act.max() < 4, (act.min(), act.max())

    A1, A2, GAP = [], [], []
    for c in range(0, NFRAME - NBLK + 1, NBLK):
        seq = np.empty(T, dtype=np.int64)
        for b in range(NBLK):
            seq[b*TPB:b*TPB+16] = tok[(c+b)*16:(c+b)*16+16]
            seq[b*TPB+16] = act[c+b]
        lo = forward(torch.from_numpy(seq))
        top = lo.topk(2, dim=-1)
        A1.append(top.indices[:, 0].numpy().astype(np.int32))
        A2.append(top.indices[:, 1].numpy().astype(np.int32))
        GAP.append((top.values[:, 0] - top.values[:, 1]).numpy().astype(np.float32))
        if (c // NBLK) % 25 == 0:
            print(f"  group {c//NBLK:3d}/100", flush=True)

    a1 = np.concatenate(A1); a2 = np.concatenate(A2); gap = np.concatenate(GAP)
    np.save(f"pt_s{s}_a1.npy", a1)
    np.save(f"pt_s{s}_a2.npy", a2)
    np.save(f"pt_s{s}_gap.npy", gap)

    ca1, cgap = read_cpp(f"{CPP}/tf2_s{s}_f32.bin")
    for q, want in PUBLISHED_Q[s].items():
        got = float(np.percentile(cgap, q))
        assert abs(got - want) < 5e-3, f"s{s} C++ p{q} = {got}, published {want}"

    mis = a1 != ca1
    d = np.abs(gap - cgap)
    # A mismatch is impossible while the smallest margin exceeds the largest
    # margin perturbation; this ratio is how close the trace came.
    head = cgap.min() / d.max()
    results.append(dict(seed=s, n=a1.size, mis=int(mis.sum()),
                        minm=float(cgap.min()), maxd=float(d.max()),
                        medd=float(np.median(d)), head=float(head),
                        mm=cgap[mis]))
    print(f"  mismatches {int(mis.sum())} / {a1.size}   "
          f"min margin {cgap.min():.3e}   max |delta| {d.max():.3e}   "
          f"headroom {head:.2f}x")

print("\n" + "=" * 76)
print(f"{'seed':6} {'n':>8} {'mismatch':>9} {'min margin':>12} "
      f"{'max |delta|':>12} {'med |delta|':>12} {'headroom':>9}")
for r in results:
    print(f"s{r['seed']:<5} {r['n']:8d} {r['mis']:9d} {r['minm']:12.3e} "
          f"{r['maxd']:12.3e} {r['medd']:12.3e} {r['head']:8.2f}x")
tot_n = sum(r["n"] for r in results)
tot_m = sum(r["mis"] for r in results)
print(f"{'TOTAL':6} {tot_n:8d} {tot_m:9d}   "
      f"({100.0*tot_m/tot_n:.4f}%)   worst headroom "
      f"{min(r['head'] for r in results):.2f}x")

if tot_m:
    mm = np.concatenate([r["mm"] for r in results])
    print(f"\nmargin at mismatched positions: median {np.median(mm):.6f}  "
          f"max {mm.max():.6f}")
    print("  a max of order 1 would mean a structural difference, not rounding")
