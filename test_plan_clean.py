"""
test_plan_clean.py — CLEAN gate baseline (no trajectory_confidence term).

Identical to test_plan_realenv.py except that plan() scores are the pure
discounted return:  sum gamma^t * alive * E[r_t]  +  gamma^H * alive * V(s_H).
No actor log_prob is added to the scores. The [Imagine]/[Decision] debug
prints are kept so runs stay inspectable.

Why no confidence term: adding the actor's log_prob to the scores is the v4
policy-prior mistake in another form — it feeds the actor's own preference back
into the ranking, and empirically it inflates the score spread so much that the
SPREAD_THRESH gate stops gating (takeovers went from ~10/ep to ~150/ep).

Three modes in one harness for a like-for-like comparison.

  MODE = "actor"  : stock actor-critic only (temperature-0.5 sampling). Baseline.
  MODE = "plan"   : v3 planner — enumerative one-step lookahead, score =
                    discounted E[r] + gamma^H * alive * V(end). Decides by argmax.
  MODE = "veto"   : safety veto — the actor decides by default; imagination only
                    OVERRIDES when it predicts the actor's action is likely to
                    lose a life AND a clearly safer action exists.

Why "veto": probe_death.py showed the world model predicts every life loss in
advance (0/4 missed, median lead ~14 steps) with a clear safer alternative
present (escape spread ~0.5), BUT it over-warns (p_death=1.0 fires dozens of
times per episode vs 4 real deaths). So a naive "veto whenever p_death is high"
would over-intervene. The veto therefore fires only when BOTH:
  (1) p_death[actor_action] > VETO_THRESH        — the chosen action looks fatal
  (2) min_a p_death[a] < p_death[actor_action] - ESCAPE_MARGIN  — escape exists.
Otherwise the actor's action stands (either it's safe, or nothing is safer).

Metrics reported: return, steps survived, life losses, and (veto mode) how many
times the veto fired. Deaths-per-episode is the low-variance signal to watch;
return is high-variance in Breakout so don't read too much into a single run.

Run from repo root:
    PYTHONPATH=src python test_plan_realenv.py
Set DEVICE = "cuda:0" on Colab.
"""
import os
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

DEVICE = "cpu"                  # "cuda:0" on Colab
CHECKPOINT = os.environ.get("IRIS_CHECKPOINT", "checkpoints/last.pt")
MODE = "gate"                   # "actor" | "plan" | "veto" | "gate"
N_EPISODES = 20                 # episodes to average over
N_STEPS = 2000                  # per-episode step cap (episode ends on its own first)
GAMMA = 0.995
TEMPERATURE = 0.5

# --- plan mode ---
H = 5

# --- veto mode (starting values; tune later) ---
H_VETO = 15                     # imagination horizon for the death check
VETO_THRESH = 0.7               # veto only if actor action's death prob exceeds this
ESCAPE_MARGIN = 0.3             # ...and the safest action is at least this much lower

VERBOSE = True                  # print [Imagine]/[Decision] lines each step

# --- gate mode ---
# Use the planner's pick ONLY when its top score beats the runner-up by more than
# SPREAD_THRESH (i.e. the evidence clears the noise floor); else defer to the actor.
# Noise-level spreads were ~0.02-0.04, real signal ~0.5-3.0, so 0.3 sits between.
SPREAD_THRESH = 0.3


@torch.no_grad()
def imagination_step(wm_env, actions, want_reward=True):
    """One imagined step. Returns (obs, E[r] or None, p_done). argmax obs tokens."""
    wm = wm_env.world_model
    n_obs_tok = wm_env.num_observations_tokens
    num_passes = 1 + n_obs_tok

    if wm_env.keys_values_wm.size + num_passes > wm.config.max_tokens:
        _ = wm_env.refresh_keys_values_with_initial_obs_tokens(wm_env.obs_tokens)

    token = actions.reshape(-1, 1)
    obs_tokens = []
    exp_r = None
    for k in range(num_passes):
        out = wm(token, past_keys_values=wm_env.keys_values_wm)
        if k == 0:
            if want_reward:
                pr = F.softmax(out.logits_rewards, dim=-1).reshape(-1, 3)
                exp_r = pr[:, 2] - pr[:, 0]
            p_done = F.softmax(out.logits_ends, dim=-1).reshape(-1, 2)[:, 1]
        if k < n_obs_tok:
            token = out.logits_observations.argmax(dim=-1)
            obs_tokens.append(token)
    wm_env.obs_tokens = torch.cat(obs_tokens, dim=1)
    obs = wm_env.decode_obs_tokens()
    return obs, exp_r, p_done


@torch.no_grad()
def branch_memory(actor_critic, K):
    saved = (actor_critic.hx, actor_critic.cx)
    actor_critic.hx = saved[0].repeat(K, 1).clone()
    actor_critic.cx = saved[1].repeat(K, 1).clone()
    return saved


@torch.no_grad()
def plan(tokenizer, world_model, actor_critic, obs, device, num_actions, H, gamma,
         verbose=False):
    """v3 planner: pure discounted return, no confidence/prior term.
    Returns the full (K,) scores; the caller decides how to use them."""
    K = num_actions
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    imagined_obs = wm_env.reset_from_initial_observations(obs.repeat(K, 1, 1, 1))
    saved = branch_memory(actor_critic, K)

    cum_reward = torch.zeros(K, device=device)
    alive = torch.ones(K, device=device)
    discount = 1.0
    for t in range(H):
        if t == 0:
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
    scores = cum_reward + discount * alive * final_value      # CLEAN: nothing else

    actor_critic.hx, actor_critic.cx = saved

    if verbose:
        print(f"  [Imagine] E[r]sum {[round(x,4) for x in cum_reward.tolist()]}  "
              f"V(end) {[round(x,3) for x in final_value.tolist()]}  "
              f"scores {[round(x,4) for x in scores.tolist()]}  "
              f"-> Planner: {int(scores.argmax().item())}")
    return scores


@torch.no_grad()
def death_probs(tokenizer, world_model, actor_critic, obs, device, num_actions, h_veto):
    """
    For each first action, imagine h_veto steps following the actor and return
    cumulative death probability (K,) = 1 - prod_t (1 - P(done)_t).
    Does not affect the actor's memory (branch is restored).
    """
    K = num_actions
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    imagined_obs = wm_env.reset_from_initial_observations(obs.repeat(K, 1, 1, 1))
    saved = branch_memory(actor_critic, K)

    survive = torch.ones(K, device=device)
    for t in range(h_veto):
        if t == 0:
            actions = torch.arange(K, device=device)
        else:
            ac_out = actor_critic(imagined_obs)
            actions = Categorical(
                logits=ac_out.logits_actions.squeeze(1) / TEMPERATURE).sample()
        imagined_obs, _, p_done = imagination_step(wm_env, actions, want_reward=False)
        survive = survive * (1.0 - p_done)

    actor_critic.hx, actor_critic.cx = saved
    return 1.0 - survive


def get_lives(env):
    try:
        return env.unwrapped.ale.lives()
    except AttributeError:
        return -1


@hydra.main(config_path="config", config_name="trainer")
def main(cfg):
    device = torch.device(DEVICE)

    tokenizer = instantiate(cfg.tokenizer)
    wm_config = instantiate(cfg.world_model)

    sd = torch.load(CHECKPOINT, map_location=device)
    num_actions = extract_state_dict(sd, "actor_critic")["actor_linear.weight"].shape[0]
    print(f"MODE={MODE}, num_actions={num_actions}"
          + (f", H_VETO={H_VETO}, VETO_THRESH={VETO_THRESH}, ESCAPE_MARGIN={ESCAPE_MARGIN}"
             if MODE == "veto" else "")
          + (f", H={H}, SPREAD_THRESH={SPREAD_THRESH}" if MODE == "gate" else ""))

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

    ep_returns, ep_deaths, ep_steps, ep_vetoes = [], [], [], []

    for ep in range(N_EPISODES):
        obs = env.reset()
        actor_critic.reset(n=1)          # fresh LSTM memory per episode
        prev_lives = get_lives(env)
        total_reward = 0.0
        n_veto = 0
        n_deaths = 0

        for step in range(N_STEPS):
            obs_t = torchvision.transforms.functional.to_tensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                input_ac = torch.clamp(
                    tokenizer.encode_decode(obs_t, should_preprocess=True,
                                            should_postprocess=True), 0, 1)
                ac_real = actor_critic(input_ac)               # one LSTM advance
                logits = ac_real.logits_actions.squeeze(1)
                actor_action = Categorical(logits=logits / TEMPERATURE).sample().item()

            if MODE == "actor":
                action = actor_action
            elif MODE == "plan":
                scores = plan(tokenizer, world_model, actor_critic, obs_t, device,
                              num_actions, H, GAMMA)
                action = int(scores.argmax().item())
            elif MODE == "gate":
                scores = plan(tokenizer, world_model, actor_critic, obs_t, device,
                              num_actions, H, GAMMA, verbose=VERBOSE)
                top2 = torch.topk(scores, 2).values
                spread = (top2[0] - top2[1]).item()
                if spread > SPREAD_THRESH:
                    action = int(scores.argmax().item())   # evidence clears noise floor
                    n_veto += 1                            # reuse counter: takeovers
                    src = "Planner"
                else:
                    action = actor_action                  # defer to the actor
                    src = "Actor"
                if VERBOSE:
                    print(f"  [Decision] actor:{actor_action}  spread:{spread:.4f}  "
                          f"-> action:{action} ({src})\n")
            elif MODE == "veto":
                pd = death_probs(tokenizer, world_model, actor_critic, obs_t, device,
                                 num_actions, H_VETO)
                action = actor_action
                safest = int(pd.argmin().item())
                if (pd[actor_action] > VETO_THRESH
                        and pd[actor_action] - pd[safest] > ESCAPE_MARGIN):
                    action = safest
                    n_veto += 1
            else:
                raise ValueError(MODE)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            lives = get_lives(env)
            if lives < prev_lives:
                n_deaths += 1
                prev_lives = lives
            if done:
                break

        ep_returns.append(total_reward)
        ep_deaths.append(n_deaths)
        ep_steps.append(step + 1)
        ep_vetoes.append(n_veto)
        label = {"veto": "vetoes", "gate": "takeovers"}.get(MODE)
        print(f"ep {ep+1:2d}/{N_EPISODES}: return {total_reward:5.0f}  "
              f"deaths {n_deaths}  steps {step+1:4d}"
              + (f"  {label} {n_veto}" if label else ""))

    def stats(x):
        x = torch.tensor(x, dtype=torch.float32)
        n = x.numel()
        mean = x.mean().item()
        sem = (x.std(unbiased=True) / (n ** 0.5)).item() if n > 1 else 0.0
        return mean, sem

    r_mean, r_sem = stats(ep_returns)
    d_mean, d_sem = stats(ep_deaths)
    s_mean, _ = stats(ep_steps)
    print(f"\n=== MODE={MODE}, {N_EPISODES} episodes ===")
    print(f"return: mean {r_mean:.1f} +/- {r_sem:.1f} (sem)   "
          f"median {sorted(ep_returns)[len(ep_returns)//2]:.0f}")
    print(f"deaths: mean {d_mean:.2f} +/- {d_sem:.2f} (sem)")
    print(f"steps:  mean {s_mean:.0f}")
    if MODE in ("veto", "gate"):
        v_mean, _ = stats(ep_vetoes)
        label = "vetoes" if MODE == "veto" else "takeovers"
        print(f"{label}: mean {v_mean:.1f} per episode")
    print(f"raw returns: {[int(x) for x in ep_returns]}")
    print(f"raw deaths:  {ep_deaths}")


if __name__ == "__main__":
    main()
