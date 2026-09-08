"""
dump_trace_agent.py — 用训练好的 agent 驱动环境，生成可复现的 trace。

相对旧 dump_trace.py 的三处改动：
  1. 动作来自 actor_critic，不是 env.action_space.sample()
     -> 状态落在部署分布上，而不是随机策略分布上
  2. 记录实际执行的动作 -> trace/actions.bin
     -> obs 和 action 因果一致；旧版把随机产生的画面配上 agent 的动作，是矛盾输入
  3. 全部随机源都 seed，输出目录可指定 -> 可复现、可多 seed

参数走环境变量（hydra 会吃掉 argv，所以不能用 sys.argv）：
  TRACE_SEED=0  TRACE_OUT=~/iris-cpp/trace_s0  TRACE_N=2000  TRACE_TEMP=0.5

用法：
  cd ~/iris
  TRACE_SEED=1 TRACE_OUT=~/iris-cpp/trace_s1 python dump_trace_agent.py
"""
import os
import sys
sys.path.insert(0, os.path.expanduser("~/iris/src"))

import random
import numpy as np
import torch
import torchvision
from torch.distributions.categorical import Categorical

import hydra
from hydra.utils import instantiate

# ---- 这一段必须和 test_plan_clean.py 的 import 完全一致 ----
# 如果报 ImportError，直接把 test_plan_clean.py 里对应的 import 行抄过来替换
from envs.wrappers import make_atari
from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from utils import extract_state_dict
# ------------------------------------------------------------

CHECKPOINT = os.path.expanduser("~/iris/checkpoints/last.pt")

SEED = int(os.environ.get("TRACE_SEED", "0"))
OUT  = os.path.expanduser(os.environ.get("TRACE_OUT", "~/iris-cpp/trace_s0"))
N    = int(os.environ.get("TRACE_N", "2000"))
TEMPERATURE = float(os.environ.get("TRACE_TEMP", "0.5"))


@hydra.main(config_path="config", config_name="trainer")
def main(cfg):
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu")

    # ---- 所有随机源 ----
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- 模型 ----
    tokenizer = instantiate(cfg.tokenizer)
    wm_config = instantiate(cfg.world_model)
    sd = torch.load(CHECKPOINT, map_location=device)

    env = make_atari('BreakoutNoFrameskip-v4', noop_max=1, max_episode_steps=108000)
    num_actions = env.action_space.n

    world_model = WorldModel(obs_vocab_size=cfg.tokenizer.vocab_size,
                             act_vocab_size=num_actions, config=wm_config)
    actor_critic = ActorCritic(act_vocab_size=num_actions)
    tokenizer.load_state_dict(extract_state_dict(sd, "tokenizer"))
    world_model.load_state_dict(extract_state_dict(sd, "world_model"))
    actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))
    tokenizer.to(device).eval()
    world_model.to(device).eval()
    actor_critic.to(device).eval()

    # ---- 环境 seed（旧 gym API）----
    try:
        env.seed(SEED)
    except Exception as e:
        print(f"[!] env.seed 失败: {e}")
    try:
        env.action_space.seed(SEED)
    except Exception as e:
        print(f"[!] action_space.seed 失败: {e}")

    obs = env.reset()
    actor_critic.reset(n=1)

    frames, actions, dones = [], [], []
    n_ep = 1

    for t in range(N):
        frames.append(np.asarray(obs))

        obs_t = torchvision.transforms.functional.to_tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            # 与 test_plan_clean.py / run_iris.cpp 完全相同的预处理链
            input_ac = torch.clamp(
                tokenizer.encode_decode(obs_t, should_preprocess=True,
                                        should_postprocess=True), 0, 1)
            ac_real = actor_critic(input_ac)          # 每帧恰好一次 LSTM 推进
            logits  = ac_real.logits_actions.squeeze(1)
            action  = Categorical(logits=logits / TEMPERATURE).sample().item()

        actions.append(action)

        out = env.step(action)
        obs, done = out[0], out[2]
        dones.append(int(done))
        if done:
            obs = env.reset()
            actor_critic.reset(n=1)               # 新的一局要清空 LSTM 记忆
            n_ep += 1

        if (t + 1) % 100 == 0:
            print(f"\r  {t+1}/{N}  episodes={n_ep}", end="", flush=True)

    print()

    a = np.stack(frames).astype(np.uint8)
    a.tofile(f"{OUT}/frames.bin")
    np.array(actions, dtype=np.int32).tofile(f"{OUT}/actions.bin")
    np.array(dones,   dtype=np.int32).tofile(f"{OUT}/dones.bin")

    uniq = len(np.unique(a.reshape(len(a), -1), axis=0))
    print(f"[*] seed={SEED}  帧数={N}  局数={n_ep}  不重复帧={uniq} ({100*uniq/N:.1f}%)")
    print(f"[*] 动作分布: " + "  ".join(
        f"{k}:{int((np.array(actions)==k).sum())}" for k in range(num_actions)))
    print(f"[*] frames.bin  {a.nbytes/1e6:.1f} MB")
    print(f"[*] actions.bin {len(actions)*4} B")
    print(f"[*] dones.bin   {len(dones)*4} B   (done=1 的帧: {int(np.sum(dones))})")
    print(f"[*] -> {OUT}")


if __name__ == "__main__":
    main()
