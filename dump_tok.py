import os, numpy as np, torch, torch.nn.functional as F

CKPT   = "checkpoints/last.pt"
OUT    = os.path.expanduser("~/iris-cpp/ref")
PREFIX = "tokenizer."
os.makedirs(OUT, exist_ok=True)

NRES, NRB, GN_EPS = 5, 2, 1e-6
ATTN_ENC, ATTN_DEC = (2, 3), (2, 3)      # 有 attn 的层级

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd  = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

def pick(name):
    key = PREFIX + name
    if key not in sd: raise SystemExit(f"[X] 缺少 {key}")
    return sd[key].float()

def save(name, t):
    a = t.detach().numpy().astype(np.float32)
    a.tofile(f"{OUT}/tok_{name}.bin")
    f = a.ravel()
    print(f"  {name:16s} {str(tuple(a.shape)):22s} n={a.size:<8d} "
          f"absmax={np.abs(f).max():9.6f}  first5=" + " ".join(f"{v:9.6f}" for v in f[:5]))

def gn(x, p):  return F.group_norm(x, 32, pick(p+".weight"), pick(p+".bias"), GN_EPS)
def sw(x):     return x * torch.sigmoid(x)
def cv(x, p, s=1, pad=1):
    return F.conv2d(x, pick(p+".weight"), pick(p+".bias"), stride=s, padding=pad)

def resblock(x, p, d=None):
    h = gn(x, p+".norm1");      _ = d and save(d+"_norm1", h)
    h = sw(h);                  _ = d and save(d+"_sw1", h)
    h = cv(h, p+".conv1");      _ = d and save(d+"_conv1", h)
    h = sw(gn(h, p+".norm2"))
    h = cv(h, p+".conv2");      _ = d and save(d+"_conv2", h)
    return x + h                                   # ch_mult 全 1,无 nin_shortcut

def attnblock(x, p, d=None):
    h = gn(x, p+".norm");       _ = d and save(d+"_norm", h)
    q = cv(h, p+".q", pad=0); k = cv(h, p+".k", pad=0); v = cv(h, p+".v", pad=0)
    if d: save(d+"_q", q); save(d+"_k", k); save(d+"_v", v)
    b, c, hh, ww = q.shape
    w_ = torch.bmm(q.reshape(b,c,hh*ww).permute(0,2,1), k.reshape(b,c,hh*ww)) * (int(c) ** -0.5)
    _ = d and save(d+"_w_raw", w_)
    w_ = F.softmax(w_, dim=2);  _ = d and save(d+"_w_sm", w_)
    h2 = torch.bmm(v.reshape(b,c,hh*ww), w_.permute(0,2,1)).reshape(b,c,hh,ww)
    _ = d and save(d+"_attout", h2)
    return x + cv(h2, p+".proj_out", pad=0)

def downsample(x, p, d=None):
    x = F.pad(x, (0,1,0,1), mode="constant", value=0)   # 只补右/下 ⚠️
    _ = d and save(d+"_pad", x)
    return cv(x, p+".conv", s=2, pad=0)

def upsample(x, p, d=None):
    x = F.interpolate(x, scale_factor=2.0, mode="nearest")
    _ = d and save(d+"_up", x)
    return cv(x, p+".conv")

# ---------------- 输入 ----------------
torch.manual_seed(0)
x = torch.rand(1, 3, 64, 64).mul(2).sub(1)      # preprocess_input
print("[*] 输入:"); save("input", x)

# ---------------- Encoder ----------------
E = "encoder."
print("[*] encoder:")
h = cv(x, E+"conv_in");  save("enc_conv_in", h)

for i in range(NRES):
    for j in range(NRB):
        h = resblock(h, f"{E}down.{i}.block.{j}", "enc_rb0" if (i,j)==(0,0) else None)
        if i in ATTN_ENC:
            h = attnblock(h, f"{E}down.{i}.attn.{j}", "enc_at0" if (i,j)==(2,0) else None)
    if i != NRES-1:
        h = downsample(h, f"{E}down.{i}.downsample", "enc_ds0" if i==0 else None)
    save(f"enc_L{i}", h)

h = resblock(h, E+"mid.block_1")
h = attnblock(h, E+"mid.attn_1")
h = resblock(h, E+"mid.block_2");  save("enc_mid", h)
h = cv(sw(gn(h, E+"norm_out")), E+"conv_out");  save("enc_z", h)

# ---------------- 量化 ----------------
print("[*] 量化:")
z = cv(h, "pre_quant_conv", pad=0);  save("z", z)
b, e, hh, ww = z.shape
zf = z.permute(0,2,3,1).reshape(-1, e)          # 'b e h w -> (b h w) e' = 转置 ⚠️
save("z_flat", zf)

emb = pick("embedding.weight")                  # (512, 512)
dist = (zf**2).sum(1, keepdim=True) + (emb**2).sum(1) - 2 * (zf @ emb.t())
save("dist", dist)

tokens = dist.argmin(dim=-1)
print("[*] tokens:", tokens.tolist())
tokens.numpy().astype(np.int32).tofile(f"{OUT}/tok_tokens.bin")

zq = emb[tokens].reshape(b, hh, ww, e).permute(0,3,1,2).contiguous()
save("z_q", zq)

# ---------------- Decoder ----------------
D = "decoder."
print("[*] decoder:")
h = cv(zq, "post_quant_conv", pad=0);  save("dec_in", h)
h = cv(h, D+"conv_in");                save("dec_conv_in", h)

h = resblock(h, D+"mid.block_1")
h = attnblock(h, D+"mid.attn_1")
h = resblock(h, D+"mid.block_2");      save("dec_mid", h)

for i in reversed(range(NRES)):
    for j in range(NRB+1):                       # decoder 每层多一个 block
        h = resblock(h, f"{D}up.{i}.block.{j}")
        if i in ATTN_DEC:
            h = attnblock(h, f"{D}up.{i}.attn.{j}")
    if i != 0:
        h = upsample(h, f"{D}up.{i}.upsample", "dec_us0" if i==NRES-1 else None)
    save(f"dec_L{i}", h)

rec = cv(sw(gn(h, D+"norm_out")), D+"conv_out");  save("rec", rec)
