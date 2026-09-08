# dump_conv1.py
import numpy as np
import torch
import torch.nn.functional as F
import os

CKPT = "checkpoints/last.pt"
OUT  = os.path.expanduser("~/iris-cpp/ref")          # 输出目录
os.makedirs(OUT,exist_ok=True)

# ---------- 1. 造固定输入 ----------
torch.manual_seed(0)
x = torch.rand(1, 3, 64, 64, dtype=torch.float32)   # [0,1) 区间,和真实帧量级一致

# ---------- 2. 取 conv1 权重 ----------
raw = torch.load(CKPT, map_location="cpu")
sd  = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

PREFIX = "actor_critic."

def pick(name):
    key = PREFIX + name
    if key not in sd:
        print(f"[!] 找不到 '{key}'")
        print(f"[!] 所有 {PREFIX}* key:")
        for k in sd:
            if k.startswith(PREFIX):
                print("   ", k, tuple(sd[k].shape))
        raise SystemExit(1)
    print(f"    {key}  ->  {tuple(sd[key].shape)}")
    return sd[key].float()

print("[*] 取权重:")
w = pick("conv1.weight")     # 期望 (32, 3, 3, 3)
b = pick("conv1.bias")       # 期望 (32,)
assert tuple(w.shape) == (32, 3, 3, 3), w.shape
assert tuple(b.shape) == (32,), b.shape

# ---------- 3. 算参考输出 ----------
h = x.mul(2).sub(1)                                  # forward 第一步
y = F.conv2d(h, w, b, stride=1, padding=1)           # (1, 32, 64, 64)

# ---------- 4. 存 raw float32 ----------
x.numpy().astype(np.float32).tofile(f"{OUT}/inp.bin")
y.numpy().astype(np.float32).tofile(f"{OUT}/conv1_ref.bin")

# ---------- 5. 打印校验信息 ----------
print()
print("[*] 参考值(C++ 侧要打出一模一样的):")
print(f"    out shape : {tuple(y.shape)}")
print(f"    abs max   : {y.abs().max().item():.6f}")
f5 = y.flatten()[:5].tolist()
print(f"    first 5   : {f5[0]:.6f} {f5[1]:.6f} {f5[2]:.6f} {f5[3]:.6f} {f5[4]:.6f}")
print()
print(f"    inp.bin       = {x.numel()} floats = {x.numel()*4} bytes")
print(f"    conv1_ref.bin = {y.numel()} floats = {y.numel()*4} bytes")
