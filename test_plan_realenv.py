"""
plan() v5 — v3 plus three correctness fixes.

Fix A: the real loop now feeds clamp(tokenizer.encode_decode(obs)) to the
       actor-critic, matching official agent.act. The actor was trained
       entirely on VQ-VAE reconstructions, so feeding raw env frames put
       it out of distribution and made the real LSTM memory mismatched
       with the imagined rollouts that branch from it.

Fix B/C: plan() no longer runs an actor forward at t=0. The first actions
       are enumerated, so ac_out was unused there — and after Fix A the
       t=0 imagined frame IS the same reconstruction the real loop just
       showed the LSTM, so forwarding again advanced the memory twice per
       real frame. Now: one LSTM advance per real frame.

Fix D: prints actor_pick (what the actor alone would choose on the real
       frame) next to the planner's choice, so imagination's influence on
       each decision is visible.

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
TEMPERATURE = 0.5       # policy temperature for continuation actions
N_STEPS = 60


@torch.no_grad()
def imagination_step(wm_env, actions, use_argmax_obs=True):
    """One imagined step: expected reward E[r]=P(+1)-P(-1), P(done),
    argmax-generated next-obs tokens. Returns (obs, expected_reward, p_done)."""
    wm = wm_env.world_model
    n_obs_tok = wm_env.num_observations_tokens
    num_passes = 1 + n_obs_tok

    if wm_env.keys_values_wm.size + num_passes > wm.config.max_tokens:
        _ = wm_env.refresh_keys_values_with_initial_obs_tokens(wm_env.obs_tokens)

    token = actions.reshape(-1, 1)
    obs_tokens = []

    for k in range(num_passes):
        outputs_wm = wm(token, past_keys_values=wm_env.keys_values_wm)

        if k == 0:
            probs_r = F.softmax(outputs_wm.logits_rewards, dim=-1).reshape(-1, 3)
            expected_reward = probs_r[:, 2] - probs_r[:, 0]
            probs_d = F.softmax(outputs_wm.logits_ends, dim=-1).reshape(-1, 2)
            p_done = probs_d[:, 1]

        if k < n_obs_tok:
            if use_argmax_obs:
                token = outputs_wm.logits_observations.argmax(dim=-1)
            else:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
            obs_tokens.append(token)

    wm_env.obs_tokens = torch.cat(obs_tokens, dim=1)
    obs = wm_env.decode_obs_tokens()
    return obs, expected_reward, p_done


@torch.no_grad()
def plan(tokenizer, world_model, actor_critic, obs, device, num_actions, H, gamma,
         actor_pick=None, verbose=False):
    """
    Enumerative one-step lookahead.
      obs: (1,3,64,64) RAW current frame — wm_env encodes it, and the
      reconstruction it decodes back is exactly the frame the caller
      already showed the actor, which is why t=0 skips the actor forward.
    """
    K = num_actions
    obs_K = obs.repeat(K, 1, 1, 1)
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    imagined_obs = wm_env.reset_from_initial_observations(obs_K)

    # Branch the real LSTM memory K ways.
    saved_hx, saved_cx = actor_critic.hx, actor_critic.cx
    actor_critic.hx = saved_hx.repeat(K, 1).clone()
    actor_critic.cx = saved_cx.repeat(K, 1).clone()

    cum_reward = torch.zeros(K, device=device)
    alive = torch.ones(K, device=device)
    discount = 1.0

    for t in range(H):
        if t == 0:
            # Enumerate: candidate k takes action k. No actor forward here —
            # the caller already advanced the LSTM on this frame (Fix B/C).
            actions = torch.arange(K, device=device)
        else:
            ac_out = actor_critic(imagined_obs)
            actions = Categorical(
                logits=ac_out.logits_actions.squeeze(1) / TEMPERATURE).sample()

        imagined_obs, exp_r, p_done = imagination_step(wm_env, actions)
        cum_reward = cum_reward + discount * alive * exp_r
        alive = alive * (1.0 - p_done)
        discount = discount * gamma

    final_value = actor_critic(imagined_obs).means_values.reshape(K)
    scores = cum_reward + discount * alive * final_value

    best = scores.argmax()
    chosen = best.reshape(1)

    actor_critic.hx, actor_critic.cx = saved_hx, saved_cx

    if verbose:
        flag = "" if actor_pick is None or actor_pick == best.item() else "  <-- overrode"
        print(f"  actor {actor_pick}  "
              f"E[r]sum {[round(x, 4) for x in cum_reward.tolist()]}  "
              f"V(end) {[round(x, 3) for x in final_value.tolist()]}  "
              f"scores {[round(x, 4) for x in scores.tolist()]}  "
              f"-> {best.item()}{flag}")
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
    n_override = 0
    for step in range(N_STEPS):
        obs_t = torchvision.transforms.functional.to_tensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            # Fix A: same input distribution the actor was trained on, and the
            # same frame plan() will imagine from.
            input_ac = torch.clamp(
                tokenizer.encode_decode(obs_t, should_preprocess=True,
                                        should_postprocess=True), 0, 1)
            ac_real = actor_critic(input_ac)          # the one LSTM advance
            actor_pick = ac_real.logits_actions.squeeze(1).argmax(dim=-1).item()

        action = plan(tokenizer, world_model, actor_critic, obs_t, device,
                      num_actions=num_actions, H=H, gamma=GAMMA,
                      actor_pick=actor_pick, verbose=True)
        if action.item() != actor_pick:
            n_override += 1

        obs, reward, done, _ = env.step(action.item())
        total_reward += reward
        if reward != 0:
            print(f"  *** step {step}: real reward {reward}! total={total_reward}")
        if done:
            print(f"episode ended at step {step}")
            break

    n = step + 1
    print(f"\nOK — ran {n} real steps, total reward: {total_reward}")
    print(f"planner overrode the actor on {n_override}/{n} steps ({100*n_override/n:.1f}%)")


if __name__ == "__main__":
    main()
