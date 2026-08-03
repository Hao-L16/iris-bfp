import os, numpy as np, torch, torch.nn.functional as F

CKPT   = "checkpoints/last.pt"
OUT    = os.path.expanduser("~/iris-cpp/ref")
PREFIX = "actor_critic."
os.makedirs(OUT, exist_ok=True)

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd  = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

def pick(name):
    key = PREFIX + name
    if key not in sd:
        raise SystemExit(f"[X] 缺少 {key}")
    return sd[key].float()

def save(name, t):
    a = t.detach().numpy().astype(np.float32)
    a.tofile(f"{OUT}/{name}.bin")
    f = a.ravel()
    print(f"  {name:10s} {str(tuple(a.shape)):18s} n={a.size:<7d} "
          f"absmax={np.abs(f).max():9.6f}  first5=" + " ".join(f"{v:9.6f}" for v in f[:5]))

torch.manual_seed(0)
x  = torch.rand(1, 3, 64, 64)
hx = torch.rand(1, 512) * 2 - 1      # 必须非零,否则 weight_hh 测不到
cx = torch.rand(1, 512) * 2 - 1

print("[*] 输入:")
save("inp", x); save("hx", hx); save("cx", cx)

print("[*] 卷积段:")
h = x.mul(2).sub(1)
for i in (1, 2, 3, 4):
    h = F.conv2d(h, pick(f"conv{i}.weight"), pick(f"conv{i}.bias"), stride=1, padding=1)
    save(f"conv{i}", h)
    h = F.max_pool2d(h, 2, 2);  save(f"pool{i}", h)
    h = F.relu(h);              save(f"relu{i}", h)

flat = h.reshape(1, -1)
save("flat", flat)

print("[*] LSTM:")
gates = (flat @ pick("lstm.weight_ih").t() + pick("lstm.bias_ih")
       + hx   @ pick("lstm.weight_hh").t() + pick("lstm.bias_hh"))
save("gates", gates)

i_, f_, g_, o_ = gates.chunk(4, dim=1)          # PyTorch 门序: i, f, g, o
i_, f_, g_, o_ = i_.sigmoid(), f_.sigmoid(), g_.tanh(), o_.sigmoid()
cx2 = f_ * cx + i_ * g_;   save("cx_out", cx2)
hx2 = o_ * cx2.tanh();     save("hx_out", hx2)

print("[*] 输出头:")
save("logits", hx2 @ pick("actor_linear.weight").t()  + pick("actor_linear.bias"))
save("value",  hx2 @ pick("critic_linear.weight").t() + pick("critic_linear.bias"))
