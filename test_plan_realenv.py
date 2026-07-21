"""
plan() v3 — three structural fixes based on the planning literature:

  Fix 1 (enumerate): candidate k's FIRST action is forced to be action k
         (K = num_actions = 4). The planner now compares Q(s, NOOP) vs
         Q(s, FIRE) vs Q(s, RIGHT) vs Q(s, LEFT) — a real decision.
         (MuZero-style root expansion, minimal version.)

  Fix 2 (expected reward): score with E[r] = P(+1) - P(-1) computed from
         the reward head's logits, instead of sampling a -1/0/+1 value.
         Turns a sparse discrete signal into a small but rankable
         continuous one. (Evaluate with expectations; sample only to
         generate — PlaNet/Dreamer convention.)

  Fix 3 (deterministic rollout): imagined obs tokens are generated with
         argmax instead of sampling, so each candidate's score is not
         drowned in rollout noise (we estimate an expectation with a
         single trajectory, so kill the variance we can kill for free).

Run from repo root:
    PYTHONPATH=src python test_plan_realenv.py
Set DEVICE = "cuda:0" on Colab.
"""

import hydra
from hydra.utils import instantiate
import torch
import torch.nn.functional as F
import torchvision
from torch.distributions.categorical import Categorical

from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from envs.world_model_env import WorldModelEnv
from envs.wrappers import make_atari
from utils import extract_state_dict

DEVICE = "cpu"          # "cuda:0" on Colab
CHECKPOINT = "/home/lhao16/iris/checkpoints/last.pt"
H = 5                   # imagination horizon
GAMMA = 0.995
TEMPERATURE = 0.5       # policy temperature for steps 1..H-1 (same as eval config)
N_STEPS = 60


@torch.no_grad()
def imagination_step(wm_env, actions, use_argmax_obs=True):
    """
    One imagined step, like WorldModelEnv.step, but:
      - returns EXPECTED reward E[r] = P(+1) - P(-1)  (float, per candidate)
      - returns P(done) instead of a sampled done
      - generates obs tokens with argmax (deterministic) if use_argmax_obs

    actions: (K,) LongTensor. Returns (obs, expected_reward, p_done).
    """
    wm = wm_env.world_model
    n_obs_tok = wm_env.num_observations_tokens
    num_passes = 1 + n_obs_tok

    if wm_env.keys_values_wm.size + num_passes > wm.config.max_tokens:
        _ = wm_env.refresh_keys_values_with_initial_obs_tokens(wm_env.obs_tokens)

    token = actions.reshape(-1, 1)          # (K, 1)
    obs_tokens = []

    for k in range(num_passes):
        outputs_wm = wm(token, past_keys_values=wm_env.keys_values_wm)

        if k == 0:
            # reward head logits: (K, 1, 3) over classes {-1, 0, +1}
            probs_r = F.softmax(outputs_wm.logits_rewards, dim=-1).reshape(-1, 3)   # (K, 3)
            expected_reward = probs_r[:, 2] - probs_r[:, 0]                          # P(+1)-P(-1)
            probs_d = F.softmax(outputs_wm.logits_ends, dim=-1).reshape(-1, 2)      # (K, 2)
            p_done = probs_d[:, 1]                                                   # P(done)

        if k < n_obs_tok:
            if use_argmax_obs:
                token = outputs_wm.logits_observations.argmax(dim=-1)   # (K, 1) most likely token
            else:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
            obs_tokens.append(token)

    wm_env.obs_tokens = torch.cat(obs_tokens, dim=1)    # (K, 16)
    obs = wm_env.decode_obs_tokens()
    return obs, expected_reward, p_done


@torch.no_grad()
def plan(tokenizer, world_model, actor_critic, obs, device, num_actions, H, gamma, verbose=False):
    """
    Enumerative one-step lookahead:
      candidate k executes action k first, then follows the policy for H-1
      more imagined steps. Score = sum of discounted E[r] + discounted
      terminal value, with survival discounting via P(done).
      Returns argmax_k score — i.e. the best first action itself.
    """
    K = num_actions
    obs_K = obs.repeat(K, 1, 1, 1)
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    imagined_obs = wm_env.reset_from_initial_observations(obs_K)

    # Branch the real LSTM memory K ways (the earlier fix, kept).
    saved_hx, saved_cx = actor_critic.hx, actor_critic.cx
    actor_critic.hx = saved_hx.repeat(K, 1).clone()
    actor_critic.cx = saved_cx.repeat(K, 1).clone()

    cum_reward = torch.zeros(K, device=device)
    alive = torch.ones(K, device=device)     # survival probability per candidate
    discount = 1.0

    for t in range(H):
        ac_out = actor_critic(imagined_obs)   # keeps LSTM advancing every step

        if t == 0:
            actions = torch.arange(K, device=device)    # Fix 1: enumerate actions
        else:
            actions = Categorical(logits=ac_out.logits_actions.squeeze(1) / TEMPERATURE).sample()

        imagined_obs, exp_r, p_done = imagination_step(wm_env, actions)   # Fix 2 + 3

        cum_reward = cum_reward + discount * alive * exp_r
        alive = alive * (1.0 - p_done)        # discount futures by survival prob
        discount = discount * gamma

    final_value = actor_critic(imagined_obs).means_values.reshape(K)
    scores = cum_reward + discount * alive * final_value

    best = scores.argmax()
    chosen = torch.tensor([best], device=device)   # best first action IS the action

    actor_critic.hx, actor_critic.cx = saved_hx, saved_cx

    if verbose:
        print(f"  E[r]sum {[round(x,4) for x in cum_reward.tolist()]}  "
              f"V(end) {[round(x,3) for x in final_value.tolist()]}  "
              f"scores {[round(x,4) for x in scores.tolist()]}  -> {best.item()}")
    return chosen


@hydra.main(config_path="config", config_name="trainer")
def main(cfg):
    device = torch.device(DEVICE)

    tokenizer = instantiate(cfg.tokenizer)
    wm_config = instantiate(cfg.world_model)

    sd = torch.load(CHECKPOINT, map_location=device)
    num_actions = extract_state_dict(sd, "actor_critic")["actor_linear.weight"].shape[0]
    print(f"num_actions: {num_actions}")

    world_model = WorldModel(obs_vocab_size=cfg.tokenizer.vocab_size,
                             act_vocab_size=num_actions, config=wm_config)
    actor_critic = ActorCritic(act_vocab_size=num_actions)

    tokenizer.load_state_dict(extract_state_dict(sd, "tokenizer"))
    world_model.load_state_dict(extract_state_dict(sd, "world_model"))
    actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))

    tokenizer.to(device).eval()
    world_model.to(device).eval()
    actor_critic.to(device).eval()

    env = make_atari('BreakoutNoFrameskip-v4')
    obs = env.reset()
    actor_critic.reset(n=1)

    total_reward = 0.0
    for step in range(N_STEPS):
        obs_t = torchvision.transforms.functional.to_tensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = actor_critic(obs_t)           # real LSTM sees the real frame

        action = plan(tokenizer, world_model, actor_critic, obs_t, device,
                      num_actions=num_actions, H=H, gamma=GAMMA, verbose=True)

        obs, reward, done, _ = env.step(action.item())
        total_reward += reward
        if reward != 0:
            print(f"  *** step {step}: real reward {reward}! total={total_reward}")
        if done:
            print(f"episode ended at step {step}")
            break

    print(f"\nOK — ran {step+1} real steps, total reward: {total_reward}")


if __name__ == "__main__":
    main()
