"""
Standalone test for the MPC-style plan() function.

Purpose: verify that a Transformer-in-the-loop planner runs end to end and
returns a legal action, BEFORE wiring it into agent.act(). Function-level
smoke test — we care that it runs and returns something legal, not that the
return is good (the input frame is random).

Run from the repo root (~/iris):
    PYTHONPATH=src python test_plan.py

On Colab (GPU), set DEVICE = "cuda:0" below.
"""
import os
import hydra
from hydra.utils import instantiate
import torch
from torch.distributions.categorical import Categorical

from models import TransformerConfig
from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from envs.world_model_env import WorldModelEnv
from utils import extract_state_dict

# ---------------------------------------------------------------------------
# Config you may want to touch
# ---------------------------------------------------------------------------
DEVICE = "cpu"          # "cuda:0" on Colab
CHECKPOINT = os.environ.get("IRIS_CHECKPOINT", "checkpoints/last.pt")
K = 4                   # number of candidate action sequences
H = 5                   # imagination horizon (steps)
GAMMA = 0.995           # trainer.yaml: training.actor_critic.gamma


@torch.no_grad()
def plan(tokenizer, world_model, actor_critic, obs, device, K=4, H=5, gamma=0.995):
    """
    obs: (1, 3, 64, 64) float in [0,1] — the current real frame.
    Returns: LongTensor (1,) — the chosen action.

    Replicate the frame K times, roll out K independent imagined trajectories
    of H steps in the world model, score each by discounted predicted rewards
    plus discounted terminal value, return the first action of the best one.
    """
    obs_K = obs.repeat(K, 1, 1, 1)                                  # (K, 3, 64, 64)

    wm_env = WorldModelEnv(tokenizer, world_model, device)
    imagined_obs = wm_env.reset_from_initial_observations(obs_K)    # (K, 3, 64, 64)

    actor_critic.reset(n=K)                                         # fresh LSTM for K rollouts

    cum_reward = torch.zeros(K, device=device)
    discount = 1.0
    first_actions = None
    last_frame = imagined_obs

    for t in range(H):
        ac_out = actor_critic(imagined_obs)
        actions = Categorical(logits=ac_out.logits_actions).sample().reshape(K)   # (K,)
        if t == 0:
            first_actions = actions.clone()

        predict_next = (t < H - 1)
        imagined_obs, reward, done, _ = wm_env.step(actions, should_predict_next_obs=predict_next)

        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).reshape(K)
        cum_reward = cum_reward + discount * reward_t
        discount = discount * gamma

        if predict_next:
            last_frame = imagined_obs

    final_value = actor_critic(last_frame).means_values.reshape(K)  # (K,)
    scores = cum_reward + discount * final_value                    # (K,)

    best = scores.argmax()
    chosen = first_actions[best].reshape(1)

    print(f"  first_actions: {first_actions.tolist()}")
    print(f"  cum_reward:    {[round(x, 3) for x in cum_reward.tolist()]}")
    print(f"  final_value:   {[round(x, 3) for x in final_value.tolist()]}")
    print(f"  scores:        {[round(x, 3) for x in scores.tolist()]}")
    print(f"  chosen action: {chosen.item()}  (from trajectory {best.item()})")
    return chosen


@hydra.main(config_path="config", config_name="trainer")
def main(cfg):
    device = torch.device(DEVICE)

    # tokenizer built via hydra (its config has nested _target_ for encoder/decoder)
    tokenizer = instantiate(cfg.tokenizer)

    # world_model / actor_critic built directly — no hydra dict instantiation,
    # which is what got the arguments tangled before.
    obs_vocab_size = cfg.tokenizer.vocab_size
    wm_config = instantiate(cfg.world_model)          # -> TransformerConfig

    # Infer the action-vocab size from the checkpoint's actor head so that
    # world_model and actor_critic agree with the stored weights.
    sd = torch.load(CHECKPOINT, map_location=device)
    act_head_w = extract_state_dict(sd, "actor_critic")["actor_linear.weight"]
    num_actions = act_head_w.shape[0]
    print(f"num_actions inferred from checkpoint: {num_actions}")

    world_model = WorldModel(obs_vocab_size=obs_vocab_size, act_vocab_size=num_actions, config=wm_config)
    actor_critic = ActorCritic(act_vocab_size=num_actions)

    tokenizer.load_state_dict(extract_state_dict(sd, "tokenizer"))
    world_model.load_state_dict(extract_state_dict(sd, "world_model"))
    actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))

    tokenizer.to(device).eval()
    world_model.to(device).eval()
    actor_critic.to(device).eval()

    obs = torch.rand(1, 3, 64, 64, device=device)   # dummy current frame

    print("Running plan()...")
    action = plan(tokenizer, world_model, actor_critic, obs, device, K=K, H=H, gamma=GAMMA)

    assert action.shape == (1,), f"bad action shape {action.shape}"
    assert 0 <= action.item() < num_actions, f"illegal action {action.item()}"
    print(f"\nOK — plan() returned a legal action: {action.item()}")


if __name__ == "__main__":
    main()
