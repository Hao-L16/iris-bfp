"""
convert_iris_to_gguf.py — export the IRIS checkpoint to GGUF for the C++ runner.

Writes ONE .gguf per component so each can be developed and verified alone:
    iris-tokenizer-f32.gguf
    iris-worldmodel-f32.gguf
    iris-actorcritic-f32.gguf

Skipped on purpose:
  * tokenizer  lpips.*                 — VGG perceptual loss, training only
  * worldmodel transformer.*.attn.mask — a (340,340) causal buffer, identical in
                                         all 10 blocks; the C++ side builds it
  * worldmodel head_*.indices          — fixed position indices; the C++ side
                                         derives them from the block layout
                                         (17 tokens/block: 16 obs + 1 action)

Tensor names are kept EXACTLY as in the PyTorch state_dict. Renaming to some
llama.cpp convention would buy nothing here (IRIS is not a llama) and would make
the C++ side harder to check against PyTorch during numerical alignment.

Usage:
    pip install gguf numpy torch
    python convert_iris_to_gguf.py --ckpt checkpoints/last.pt --outdir gguf/
"""

import argparse
import os

import numpy as np
import torch

try:
    import gguf
except ImportError:
    raise SystemExit("pip install gguf")


SKIP_PREFIXES = {
    "tokenizer": ("lpips.",),
    "world_model": (),
    "actor_critic": (),
}

SKIP_SUFFIXES = {
    "tokenizer": (),
    "world_model": (".attn.mask", ".indices"),
    "actor_critic": (),
}

# Architecture metadata the C++ side needs. Values are read from the tensors
# where possible so this stays honest if the checkpoint ever changes.
def tokenizer_meta(t):
    return {
        "vocab_size":      t["embedding.weight"].shape[0],       # 512
        "embed_dim":       t["embedding.weight"].shape[1],       # 512
        "z_channels":      t["encoder.conv_out.weight"].shape[0],# 512
        "ch":              t["encoder.conv_in.weight"].shape[0], # 64
        "in_channels":     t["encoder.conv_in.weight"].shape[1], # 3
        "num_res_blocks":  2,      # encoder: down.N.block.0/1
        "num_resolutions": 5,      # down.0..4  (64->32->16->8->4)
        "group_norm_eps":  1e-6,   # VQGAN default — must match PyTorch exactly
    }


def world_model_meta(t):
    n_layer = 1 + max(int(k.split(".")[2]) for k in t if k.startswith("transformer.blocks."))
    return {
        "n_layer":         n_layer,                                    # 10
        "n_embd":          t["transformer.ln_f.weight"].shape[0],      # 256
        "n_head":          4,                                          # from config
        "max_tokens":      t["pos_emb.weight"].shape[0],               # 340
        "tokens_per_block": 17,                                        # 16 obs + 1 action
        "obs_vocab_size":  t["embedder.embedding_tables.1.weight"].shape[0],  # 512
        "act_vocab_size":  t["embedder.embedding_tables.0.weight"].shape[0],  # 4
    }


def actor_critic_meta(t):
    return {
        "lstm_dim":     t["lstm.weight_hh"].shape[1],        # 512
        "lstm_input":   t["lstm.weight_ih"].shape[1],        # 1024
        "act_vocab_size": t["actor_linear.weight"].shape[0], # 4
        "in_channels":  t["conv1.weight"].shape[1],          # 3
    }


META = {
    "tokenizer":    tokenizer_meta,
    "world_model":  world_model_meta,
    "actor_critic": actor_critic_meta,
}


def collect(sd, comp):
    """Pull one component's tensors out of the flat checkpoint dict."""
    out = {}
    for k, v in sd.items():
        if not k.startswith(comp + "."):
            continue
        name = k[len(comp) + 1:]
        if any(name.startswith(p) for p in SKIP_PREFIXES[comp]):
            continue
        if any(name.endswith(s) for s in SKIP_SUFFIXES[comp]):
            continue
        out[name] = v
    return out


def write_component(tensors, comp, outdir):
    path = os.path.join(outdir, f"iris-{comp.replace('_','')}-f32.gguf")
    w = gguf.GGUFWriter(path, arch=f"iris-{comp.replace('_','')}")

    for key, val in META[comp](tensors).items():
        if isinstance(val, float):
            w.add_float32(key, val)
        else:
            w.add_uint32(key, int(val))

    n_bytes = 0
    for name, t in tensors.items():
        arr = t.detach().cpu().float().numpy()
        arr = np.ascontiguousarray(arr)
        w.add_tensor(name, arr)
        n_bytes += arr.nbytes

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  wrote {path}: {len(tensors)} tensors, {n_bytes/1e6:.1f} MB")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/last.pt")
    ap.add_argument("--outdir", default="gguf")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sd = torch.load(args.ckpt, map_location="cpu")

    for comp in ("tokenizer", "world_model", "actor_critic"):
        tensors = collect(sd, comp)
        total = sum(1 for k in sd if k.startswith(comp + "."))
        print(f"\n{comp}: {len(tensors)} tensors kept, {total - len(tensors)} skipped")
        write_component(tensors, comp, args.outdir)


if __name__ == "__main__":
    main()
