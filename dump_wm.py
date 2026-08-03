import os, math, numpy as np, torch, torch.nn.functional as F

CKPT   = "checkpoints/last.pt"
OUT    = os.path.expanduser("~/iris-cpp/ref")
PREFIX = "world_model."
os.makedirs(OUT, exist_ok=True)

# ---- config/world_model/default.yaml ----
TPB, NBLK = 17, 2
T         = TPB * NBLK        # 34
E, NH     = 256, 4
HS        = E // NH           # 64
NLAYER    = 10
OBS_V, ACT_V = 512, 4
EPS       = 1e-5

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd  = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

def pick(name):
    key = PREFIX + name
    if key not in sd:
        raise SystemExit(f"[X] 缺少 {key}")
    return sd[key].float()

def save(name, t):
    a = t.detach().numpy().astype(np.float32)
    a.tofile(f"{OUT}/wm_{name}.bin")
    f = a.ravel()
    print(f"  {name:12s} {str(tuple(a.shape)):22s} n={a.size:<7d} "
          f"absmax={np.abs(f).max():9.6f}  first5=" + " ".join(f"{v:9.6f}" for v in f[:5]))

def ln(t, w, b):
    return F.layer_norm(t, (E,), w, b, EPS)

# ---------- 输入 token ----------
g = torch.Generator().manual_seed(0)
tokens = torch.zeros(1, T, dtype=torch.long)
for p in range(T):
    hi = ACT_V if p % TPB == TPB - 1 else OBS_V     # 位置 16 是动作
    tokens[0, p] = torch.randint(0, hi, (1,), generator=g)

tokens[0,TPB-1]   = 1
tokens[0,2*TPB-1] = 3

tokens[0].numpy().astype(np.int32).tofile(f"{OUT}/wm_tokens.bin")
print("[*] tokens:", tokens[0].tolist())

# ---------- embedding + pos ----------
emb_act = pick("embedder.embedding_tables.0.weight")   # (4,256)
emb_obs = pick("embedder.embedding_tables.1.weight")   # (512,256)
pos     = pick("pos_emb.weight")                       # (340,256)

x = torch.zeros(1, T, E)
for p in range(T):
    tok = int(tokens[0, p])
    x[0, p] = emb_act[tok] if p % TPB == TPB - 1 else emb_obs[tok]
print("[*] embedding:")
save("embed", x)
x = x + pos[:T]
save("seq", x)

# ---------- 10 个 block ----------
mask = torch.tril(torch.ones(T, T))
print("[*] blocks (block 0 逐子步骤):")
for L in range(NLAYER):
    p, d = f"transformer.blocks.{L}.", (L == 0)

    h = ln(x, pick(p+"ln1.weight"), pick(p+"ln1.bias"))
    if d: save("b0_ln1", h)

    q = F.linear(h, pick(p+"attn.query.weight"), pick(p+"attn.query.bias"))
    k = F.linear(h, pick(p+"attn.key.weight"),   pick(p+"attn.key.bias"))
    v = F.linear(h, pick(p+"attn.value.weight"), pick(p+"attn.value.bias"))
    if d: save("b0_q", q); save("b0_k", k); save("b0_v", v)

    qh = q.view(1, T, NH, HS).transpose(1, 2)      # (1,NH,T,HS)
    kh = k.view(1, T, NH, HS).transpose(1, 2)
    vh = v.view(1, T, NH, HS).transpose(1, 2)

    att = (qh @ kh.transpose(-2, -1)) * (1.0 / math.sqrt(HS))
    if d: save("b0_att_raw", att)                  # 掩码前

    att = F.softmax(att.masked_fill(mask == 0, float('-inf')), dim=-1)
    if d: save("b0_att_sm", att)                   # softmax 后

    y = (att @ vh).transpose(1, 2).reshape(1, T, E)   # 'b h t e -> b t (h e)'
    if d: save("b0_att_out", y)

    y = F.linear(y, pick(p+"attn.proj.weight"), pick(p+"attn.proj.bias"))
    if d: save("b0_proj", y)

    x = x + y
    if d: save("b0_res1", x)

    h2 = ln(x, pick(p+"ln2.weight"), pick(p+"ln2.bias"))
    if d: save("b0_ln2", h2)
    h2 = F.linear(h2, pick(p+"mlp.0.weight"), pick(p+"mlp.0.bias"))
    if d: save("b0_fc", h2)
    h2 = F.gelu(h2)                                # ← 若 ggml 无 erf 版,改 approximate='tanh'
    if d: save("b0_gelu", h2)
    h2 = F.linear(h2, pick(p+"mlp.2.weight"), pick(p+"mlp.2.bias"))
    if d: save("b0_mlp", h2)

    x = x + h2
    save(f"blk{L}", x)

# ---------- ln_f ----------
print("[*] ln_f + heads:")
x = ln(x, pick("transformer.ln_f.weight"), pick("transformer.ln_f.bias"))
save("lnf", x)

# ---------- 三个 head(取模切片) ----------
def head(name, keep):
    idx = [p for p in range(T) if keep(p)]
    h = x[:, idx]
    h = F.linear(h, pick(f"{name}.head_module.0.weight"), pick(f"{name}.head_module.0.bias"))
    h = F.relu(h)
    return F.linear(h, pick(f"{name}.head_module.2.weight"), pick(f"{name}.head_module.2.bias"))

save("logits_obs",     head("head_observations", lambda p: p % TPB != TPB - 2))
save("logits_rewards", head("head_rewards",      lambda p: p % TPB == TPB - 1))
save("logits_ends",    head("head_ends",         lambda p: p % TPB == TPB - 1))
