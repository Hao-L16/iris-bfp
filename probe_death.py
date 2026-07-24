"""
probe_death.py — measure how far in advance the world model can see a life loss.

Why: a safety-veto planner can only work if P(done) rises BEFORE the ball is
lost. H=5 covers only 20 game frames, while a ball takes ~40-60 frames to fall,
so the veto may never fire. This script measures the model's actual warning
lead time, which is what should set H — not physics, and not guesswork.

What it does, per real step (episode is played by the stock actor):
  - records the true life count from ALE (ground truth for death events)
  - runs a diagnostic H_PROBE-step rollout for EACH first action (batched, K=4),
    following the actor thereafter, and records for each action:
      * cumulative death probability  1 - prod_t (1 - P(done)_t)
      * the earliest imagined step where P(done) exceeds P_STEP_THRESH
    The probe never influences the action taken.

Then it post-processes: for every life loss, how many real steps earlier did the
danger signal first cross DANGER_THRESH, and was a safer alternative action
available at that moment.

Run from repo root:
    PYTHONPATH=src python probe_death.py
Set DEVICE = "cuda:0" on Colab (strongly recommended — the probe is ~4x the
cost of plan() v3 per real step).
"""

import csv

import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.distributions.categorical import Categorical

from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from envs.world_model_env import WorldModelEnv
from envs.wrappers import make_atari
from utils import extract_state_dict

DEVICE = "cpu"                  # "cuda:0" on Colab
CHECKPOINT = "/home/lhao16/iris/checkpoints/last.pt"
N_STEPS = 2000                  # episode ends on its own before this
H_PROBE = 20                    # diagnostic horizon (80 game frames)
TEMPERATURE = 0.5               # actor sampling temperature (official eval)
PROBE_EVERY = 1                 # set >1 to subsample if too slow
P_STEP_THRESH = 0.10            # per-imagined-step P(done) counted as a warning
DANGER_THRESH = 0.30            # cumulative death prob counted as "danger"
LOOKBACK = 40                   # how far back to search for the first warning
CSV_PATH = "probe_death.csv"


def get_lives(env):
    try:
        return env.unwrapped.ale.lives()
    except AttributeError:
        return -1


@torch.no_grad()
def probe(tokenizer, world_model, actor_critic, obs, device, num_actions, h_probe):
    """
    For each first action k, imagine h_probe steps following the actor and
    return (cum_death_prob (K,), first_warn_t (K,)) where first_warn_t is the
    earliest imagined step with P(done) > P_STEP_THRESH, or -1 if never.
    Purely diagnostic — does not affect the action taken.
    """
    K = num_actions
    obs_K = obs.repeat(K, 1, 1, 1)
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    imagined_obs = wm_env.reset_from_initial_observations(obs_K)

    saved_hx, saved_cx = actor_critic.hx, actor_critic.cx
    actor_critic.hx = saved_hx.repeat(K, 1).clone()
    actor_critic.cx = saved_cx.repeat(K, 1).clone()

    survive = torch.ones(K, device=device)          # prod (1 - p_done)
    first_warn = torch.full((K,), -1, dtype=torch.long, device=device)

    wm = world_model
    n_obs_tok = wm_env.num_observations_tokens
    num_passes = 1 + n_obs_tok

    for t in range(h_probe):
        if t == 0:
            actions = torch.arange(K, device=device)
        else:
            ac_out = actor_critic(imagined_obs)
            actions = Categorical(
                logits=ac_out.logits_actions.squeeze(1) / TEMPERATURE).sample()

        if wm_env.keys_values_wm.size + num_passes > wm.config.max_tokens:
            _ = wm_env.refresh_keys_values_with_initial_obs_tokens(wm_env.obs_tokens)

        token = actions.reshape(-1, 1)
        obs_tokens = []
        for k in range(num_passes):
            out = wm(token, past_keys_values=wm_env.keys_values_wm)
            if k == 0:
                p_done = F.softmax(out.logits_ends, dim=-1).reshape(-1, 2)[:, 1]
            if k < n_obs_tok:
                token = out.logits_observations.argmax(dim=-1)
                obs_tokens.append(token)
        wm_env.obs_tokens = torch.cat(obs_tokens, dim=1)
        imagined_obs = wm_env.decode_obs_tokens()

        newly = (p_done > P_STEP_THRESH) & (first_warn < 0)
        first_warn = torch.where(newly, torch.full_like(first_warn, t), first_warn)
        survive = survive * (1.0 - p_done)

    actor_critic.hx, actor_critic.cx = saved_hx, saved_cx
    return (1.0 - survive), first_warn


def analyse(rows):
    """rows: list of dicts with step, lives, actor_pick, p_death (list of K)."""
    steps = [r["step"] for r in rows]
    lives = [r["lives"] for r in rows]
    idx_of_step = {s: i for i, s in enumerate(steps)}

    deaths = [steps[i] for i in range(1, len(rows)) if lives[i] < lives[i - 1]]
    print(f"\n=== {len(deaths)} life losses at steps: {deaths}")
    if not deaths:
        print("no life losses recorded — nothing to analyse")
        return

    leads, escapes, missed = [], [], 0
    for s in deaths:
        first = None
        for back in range(1, LOOKBACK + 1):
            j = idx_of_step.get(s - back)
            if j is None:
                continue
            r = rows[j]
            if r["p_death"][r["actor_pick"]] > DANGER_THRESH:
                first = s - back        # keep walking back to find the earliest
        if first is None:
            missed += 1
            continue
        leads.append(s - first)
        j = idx_of_step[first]
        r = rows[j]
        escapes.append(max(r["p_death"]) - min(r["p_death"]))

    if leads:
        a = np.array(leads)
        print(f"warning lead time (real steps before the life loss):")
        print(f"  n={len(a)}  min={a.min()}  median={np.median(a):.1f}  "
              f"max={a.max()}  mean={a.mean():.1f}")
        print(f"  -> H should be at least ~{int(np.median(a))}, "
              f"with margin ~{int(np.percentile(a, 75))}")
        e = np.array(escapes)
        print(f"escape-action spread at first warning "
              f"(max-min cumulative death prob across the {len(rows[0]['p_death'])} actions):")
        print(f"  median={np.median(e):.3f}  max={e.max():.3f}")
        print("  large spread => a safer action existed, so a veto could help;")
        print("  near zero    => all actions equally doomed by then.")
    print(f"life losses with NO warning above {DANGER_THRESH} within "
          f"{LOOKBACK} steps: {missed}/{len(deaths)}")
    if missed == len(deaths):
        print("  ** the world model never predicted these deaths in advance — "
              "a safety veto cannot work with this signal **")


@hydra.main(config_path="config", config_name="trainer")
def main(cfg):
    device = torch.device(DEVICE)

    tokenizer = instantiate(cfg.tokenizer)
    wm_config = instantiate(cfg.world_model)

    sd = torch.load(CHECKPOINT, map_location=device)
    num_actions = extract_state_dict(sd, "actor_critic")["actor_linear.weight"].shape[0]
    print(f"num_actions: {num_actions}, H_PROBE={H_PROBE}")

    world_model = WorldModel(obs_vocab_size=cfg.tokenizer.vocab_size,
                             act_vocab_size=num_actions, config=wm_config)
    actor_critic = ActorCritic(act_vocab_size=num_actions)

    tokenizer.load_state_dict(extract_state_dict(sd, "tokenizer"))
    world_model.load_state_dict(extract_state_dict(sd, "world_model"))
    actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))

    tokenizer.to(device).eval()
    world_model.to(device).eval()
    actor_critic.to(device).eval()

    env = make_atari('BreakoutNoFrameskip-v4', noop_max=1, max_episode_steps=108000)
    obs = env.reset()
    actor_critic.reset(n=1)
    print(f"starting lives: {get_lives(env)}")

    rows = []
    total_reward = 0.0
    for step in range(N_STEPS):
        obs_t = torchvision.transforms.functional.to_tensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            input_ac = torch.clamp(
                tokenizer.encode_decode(obs_t, should_preprocess=True,
                                        should_postprocess=True), 0, 1)
            ac_real = actor_critic(input_ac)
            logits = ac_real.logits_actions.squeeze(1)
            actor_pick = logits.argmax(dim=-1).item()
            action = Categorical(logits=logits / TEMPERATURE).sample().item()

        if step % PROBE_EVERY == 0:
            p_death, first_warn = probe(tokenizer, world_model, actor_critic,
                                        obs_t, device, num_actions, H_PROBE)
            rows.append({
                "step": step,
                "lives": get_lives(env),
                "action": action,
                "actor_pick": actor_pick,
                "p_death": [round(x, 4) for x in p_death.tolist()],
                "first_warn": first_warn.tolist(),
            })
            if p_death[actor_pick] > DANGER_THRESH:
                print(f"  step {step}: DANGER p_death={p_death[actor_pick]:.3f} "
                      f"(all: {[round(x,3) for x in p_death.tolist()]}) "
                      f"lives={get_lives(env)}")

        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            print(f"episode ended at step {step}")
            break

    print(f"\nran {step+1} steps, total reward: {total_reward}, "
          f"final lives: {get_lives(env)}")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lives", "action", "actor_pick"]
                   + [f"p_death_a{i}" for i in range(num_actions)]
                   + [f"first_warn_a{i}" for i in range(num_actions)])
        for r in rows:
            w.writerow([r["step"], r["lives"], r["action"], r["actor_pick"]]
                       + r["p_death"] + r["first_warn"])
    print(f"wrote {len(rows)} rows to {CSV_PATH}")

    analyse(rows)


if __name__ == "__main__":
    main()
