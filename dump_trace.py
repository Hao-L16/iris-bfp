import os, sys
sys.path.insert(0, os.path.expanduser("~/iris/src"))

import numpy as np
from envs.wrappers import make_atari

OUT = os.path.expanduser("~/iris-cpp/trace")
os.makedirs(OUT, exist_ok=True)
N = 2000

env = make_atari('BreakoutNoFrameskip-v4', noop_max=1, max_episode_steps=108000)
obs = env.reset()
a0 = np.asarray(obs)
print("[*] obs:", a0.shape, a0.dtype, "range", a0.min(), a0.max())

frames = []
for t in range(N):
    frames.append(np.asarray(obs))
    out = env.step(env.action_space.sample())
    obs, done = out[0], out[2]
    if done:
        obs = env.reset()

a = np.stack(frames)
print("[*] stacked:", a.shape, a.dtype)
a.astype(np.uint8).tofile(f"{OUT}/frames.bin")
print(f"[*] 写出 {a.nbytes/1e6:.1f} MB → {OUT}/frames.bin")
