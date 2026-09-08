#!/usr/bin/env python3
# ---- save as: ~/iris/tf_pytorch.py   (run from ~/iris) ----
"""Teacher-forced replay in PyTorch, mirroring tf.cpp exactly, to measure the
baseline argmax mismatch rate between the C++ port and the reference model.

Why: Section 3.2.2 licenses the whole flip-rate measurement on the claim that at
full precision the port makes exactly the decisions the reference makes -- but
that was checked on ONE verification input.  The margin distribution has p1 =
0.021, and the port agrees with PyTorch only to ~1e-4 relative on activations
(plus a deliberate GELU erf-vs-tanh substitution, ~3e-4 at worst), so a nonzero
baseline mismatch is expected.  predict_baseline.py puts the upper bound at
0.096% of positions, against a q8_0 flip rate of 0.967%.

Protocol copied from tf.cpp / dump_wm.py, not from iris.h:
  * 20 blocks of 17 tokens = 340 per forward; blocks are independent (no cache
    carried across groups), 100 groups over 2000 frames.
  * position p % 17 == 16 is the action; 0..15 are observation tokens.
  * actions use a SEPARATE embedding table (embedding_tables.0), indexed by the
    raw action 0..3.  The 512+a remapping mentioned in iris.h is a GGUF-side
    detail: the export concatenates the two tables into one of 516 rows.
  * the observation head is read at positions p % 17 != 15 -> 320 per group.
    Note p = 339 predicts the first token of the NEXT frame, outside the group;
    tf.cpp records it too, so it is kept here for alignment.
  * LayerNorm eps is 1e-5 (gn_eps = 1e-6 in iris.h is the tokeniser's GroupNorm).
  * GELU stays in the erf form: that substitution is the effect being measured.

Writes pt_s0_{a1,a2,gap}.npy and compares against ~/iris-cpp/tf2_s0_f32.bin.
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
TRACE = f"{CPP}/trace_s0"
REF = f"{CPP}/tf2_s0_f32.bin"

TPB, NBLK = 17, 20
T = TPB * NBLK                    # 340
E, NH, HS, NLAYER = 256, 4, 64, 10
OBS_V, EPS = 512, 1e-5
NFRAME = 2000

# positions at which the observation head is read
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
h0w, h0b = pick("head_observations.head_module.0.weight"), pick("head_observations.head_module.0.bias")
h2w, h2b = pick("head_observations.head_module.2.weight"), pick("head_observations.head_module.2.bias")

MASK = torch.tril(torch.ones(T, T)) == 0


def forward(seq):
    """seq: LongTensor (T,) with obs tokens 0..511 and raw actions 0..3."""
    x = torch.empty(1, T, E)
    is_act = torch.tensor([p % TPB == TPB - 1 for p in range(T)])
    x[0, ~is_act] = emb_obs[seq[~is_act]]
    x[0, is_act] = emb_act[seq[is_act]]
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


tok = np.fromfile(f"{TRACE}/tokens.bin", dtype=np.int32)
act = np.fromfile(f"{TRACE}/actions.bin", dtype=np.int32)
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
    if (c // NBLK) % 10 == 0:
        print(f"  group {c//NBLK:3d}/100", flush=True)

a1 = np.concatenate(A1); a2 = np.concatenate(A2); gap = np.concatenate(GAP)
np.save("pt_s0_a1.npy", a1); np.save("pt_s0_a2.npy", a2); np.save("pt_s0_gap.npy", gap)
print(f"\nwrote pt_s0_*.npy  ({a1.size} positions)")

# ---- compare against the C++ FP32 reference -------------------------------
NOBS, RB = 320, 16
CH = NOBS * RB + 20*3*4 + 20*2*4
buf = np.fromfile(REF, dtype=np.uint8)
nblk = buf.size // CH
r = buf.reshape(nblk, CH)[:, :NOBS*RB].reshape(nblk*NOBS, RB)
ca1 = r[:, 0:4].copy().view(np.int32).ravel()
cgap = r[:, 8:12].copy().view(np.float32).ravel()

# provenance: the C++ margins must reproduce the published Table C.2 s0 row
for q, want in [(1, 0.0212), (25, 0.7558), (50, 2.6327), (95, 10.977)]:
    got = float(np.percentile(cgap, q))
    assert abs(got - want) < 5e-3, f"C++ ref p{q} = {got}, published {want}"
print("C++ reference reproduces Table C.2 (s0)")

mis = a1 != ca1
n = mis.sum()
print(f"\n{'='*58}\nBASELINE MISMATCH: {n} of {a1.size}  ({100.0*n/a1.size:.4f}%)")
print(f"  for scale, q8_0 flip rate on s0 is 0.994%  "
      f"-> {100.0*n/a1.size/0.994:.1%} of it")
print(f"  predicted upper bound (margin < 2e-3): 0.119% on s0")

if n:
    mm = cgap[mis]
    print(f"\n  margin at mismatched positions:")
    print(f"    median {np.median(mm):.6f}   mean {mm.mean():.6f}")
    print(f"    max    {mm.max():.6f}   <-- if O(1), this is NOT rounding")
    print(f"    all below 0.01?  {bool((mm < 0.01).all())}")
    print(f"    sorted: {np.sort(mm)[:15]}")

d = np.abs(gap - cgap)
print(f"\nmargin agreement (all positions):")
print(f"  max |delta| {d.max():.2e}   median {np.median(d):.2e}")
print(f"  relative to margin scale (mean 3.55): {d.max()/3.55:.2e}")
