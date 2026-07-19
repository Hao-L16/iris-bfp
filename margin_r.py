"""Certified safe radius r for the IRIS tokenizer codebook.
r = min_k (d_k - d_k1) / (2 ||e_k - e_k1||),  d_k = ||z - e_k||^2.
If ||delta|| <= r, the token provably cannot flip, so encode_decode is bit-identical."""
import sys; sys.path.insert(0, 'src')
import numpy as np, torch
from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig
from models.actor_critic import ActorCritic
from envs import make_atari

DEV, CKPT, GAME, N_FRAMES = 'cuda:0', 'checkpoints/last.pt', 'BreakoutNoFrameskip-v4', 2000

cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64,
                           ch_mult=[1,1,1,1,1], num_res_blocks=2, attn_resolutions=[8,16],
                           out_ch=3, dropout=0.0)
tok = Tokenizer(512, 512, Encoder(cfg), Decoder(cfg), with_lpips=False).to(DEV).eval()

env = make_atari(GAME, size=64, max_episode_steps=108000, noop_max=1,
                 frame_skip=4, done_on_life_loss=False, clip_reward=False)
ac = ActorCritic(act_vocab_size=env.action_space.n, use_original_obs=False).to(DEV).eval()

sd = torch.load(CKPT, map_location=DEV, weights_only=False)
tok.load_state_dict({k[len('tokenizer.'):]: v for k, v in sd.items()
                     if k.startswith('tokenizer.') and not k.startswith('tokenizer.lpips.')})
ac.load_state_dict({k[len('actor_critic.'):]: v for k, v in sd.items()
                    if k.startswith('actor_critic.')})

frames, obs = [], env.reset()
ac.reset(n=1)
with torch.no_grad():
    while len(frames) < N_FRAMES:
        x = torch.FloatTensor(obs).div(255).permute(2, 0, 1).unsqueeze(0).to(DEV)
        frames.append(x.cpu())
        rec = torch.clamp(tok.encode_decode(x, should_preprocess=True, should_postprocess=True), 0, 1)
        a = torch.distributions.Categorical(logits=ac(rec).logits_actions[:, -1] / 0.5).sample()
        obs, _, done, _ = env.step(a.item())
        if done:
            obs = env.reset()
            ac.reset(n=1)
print(f'collected {len(frames)} on-policy frames')

E = tok.embedding.weight.data
D_cb = torch.cdist(E, E)
rs, znorms = [], []
with torch.no_grad():
    for i in range(0, len(frames), 64):
        x = torch.cat(frames[i:i+64]).to(DEV)
        z = tok.pre_quant_conv(tok.encoder(tok.preprocess_input(x)))
        zf = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        d = (zf**2).sum(1, keepdim=True) + (E**2).sum(1) - 2 * zf @ E.t()
        k1 = d.argmin(1)
        num = d - d.gather(1, k1[:, None])
        den = 2 * D_cb[k1]
        ratio = torch.where(den > 1e-8, num / den.clamp(min=1e-8),
                            torch.full_like(num, float('inf')))
        ratio.scatter_(1, k1[:, None], float('inf'))
        rs.append(ratio.min(1).values.cpu())
        znorms.append(zf.norm(dim=1).cpu())

r = torch.cat(rs).numpy(); zn = torch.cat(znorms).numpy()
rel = r / zn
bits = np.log2(1.0 / np.maximum(rel, 1e-12))

print(f'\ntokens: {len(r)}  ({len(frames)} frames x 16)')
for name, v in [('r (absolute)', r), ('||z||', zn), ('r / ||z||', rel)]:
    p = np.percentile(v, [1, 5, 25, 50, 75, 95])
    print(f'{name:14s} p1={p[0]:.4g}  p5={p[1]:.4g}  p25={p[2]:.4g}  '
          f'median={p[3]:.4g}  p75={p[4]:.4g}  p95={p[5]:.4g}')
print(f'\ncertified mantissa bits log2(||z||/r): median={np.median(bits):.1f}  '
      f'p95={np.percentile(bits,95):.1f}  p99={np.percentile(bits,99):.1f}  max={bits.max():.1f}')
