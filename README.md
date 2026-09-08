# iris-bfp

> **This is a fork.** The original IRIS README follows below, unchanged. This
> section describes what has been added.

A fork of [IRIS](https://github.com/eloialonso/iris) extended with the tooling
for an MSc dissertation on low-bit quantisation of discrete-token world models.
Nothing in the original training pipeline has been modified — the additions
export the trained checkpoint to GGUF, record reproducible traces, and provide
a PyTorch reference for verifying a C++ port.

The C++/GGML runtime that consumes all of this lives in
[iris-cpp](https://github.com/Hao-L16/iris-cpp).

## What was added

### Export

| Script | What it does |
|---|---|
| `convert_iris_to_gguf.py` | Exports the IRIS checkpoint to GGUF, one file per component (`iris-tokenizer-f32.gguf`, `iris-worldmodel-f32.gguf`, `iris-actorcritic-f32.gguf`) so each can be developed and verified alone. 578 tensors in total, with no restructuring — each keeps its name, shape and values, so any discrepancy in the port is attributable to the compute graph rather than to the export. |

### Reference activations

`dump_tok.py`, `dump_wm.py` and `dump_ac.py` instrument the PyTorch model and
dump intermediate activations at 80 points across the three components — after
each convolution and normalisation in the tokeniser, after attention and each
MLP stage in every Transformer block, at the codebook lookup, and at each
output head. The C++ implementation is required to reproduce every one of them
from the same input.

These write to `~/iris-cpp/ref` by default. Edit `OUT` at the head of each
script if your checkout lives elsewhere.

Two criteria apply, and the distinction is the point. Continuous intermediates
are judged on **relative** error — an activation of magnitude 50 and one of
magnitude 0.001 tolerate very different absolute deviations. The discrete path
is judged on **exactness**: a model whose outputs are consumed as categorical
choices is verified not by how close its logits are but by whether it chooses
the same thing.

### Traces

| Script | What it does |
|---|---|
| `dump_trace_agent.py` | Drives the environment with the trained agent and records a reproducible trace: frames, tokens, actions and dones. |
| `dump_trace.py` | The earlier random-policy version, kept for reference. |

The switch from random policy to agent-driven traces is a correction, not a
preference. A random policy dies quickly against the dense early wall, so its
states are unrepresentative in a way that **changes conclusions and not merely
their precision**: on random-policy traces the termination head is the
sensitive one and the reward head is saturated, and on agent-driven traces the
ordering reverses. Use `dump_trace_agent.py`.

### PyTorch reference for the port

| Script | What it does |
|---|---|
| `tf_pytorch.py` | Teacher-forced replay in PyTorch, mirroring `tf.cpp` exactly, to measure the baseline argmax mismatch rate between the C++ port and the reference model. |
| `tf_pytorch_all.py` | The same over all five traces. |
| `margin_headroom.py` | Per-position headroom between the C++/PyTorch margin discrepancy and the margin itself. The statement has to be made per position rather than globally, because the smallest margin anywhere is not larger than the largest discrepancy anywhere on some traces — the two extremes fall at different positions. |

**Two traps if you repeat this.** These scripts read the per-seed trace
directories directly rather than the working copy the C++ runner consumes, so
the comparison cannot be run against a stale trace. And they reproduce the
*reference implementation* rather than the port's description of it: action
tokens occupy a separate embedding table indexed by the raw action, where the
GGUF export concatenates that table with the observation table into one of 516
rows. Following the exported layout in PyTorch produces a program that runs and
computes a different function.

### Planner experiments

| Script | What it does |
|---|---|
| `test_plan_realenv.py` | Three integration modes in one harness for a like-for-like comparison: `actor` (stock actor-critic, the baseline), `plan` (enumerative one-step lookahead, decides by argmax), and `veto` (the actor decides by default; imagination only overrides when it predicts a life loss and a clearly safer action exists). |
| `test_plan_clean.py` | The gate baseline with no trajectory-confidence term. Adding the actor's log-prob to the scores feeds the actor's own preference back into the ranking and inflates the score spread so much that the gate stops gating — takeovers went from ~10 per episode to ~150. |
| `test_plan.py` | A function-level smoke test: verifies that a Transformer-in-the-loop planner runs end to end and returns a legal action, before wiring it into `agent.act()`. |
| `probe_death.py` | Establishes that the world model anticipates life losses — none missed over an episode, median lead ~14 steps — but over-warns, firing dozens of times per episode against four real events. This is why a naive "veto whenever p_death is high" over-intervenes. |

Set `IRIS_CHECKPOINT` to point at the checkpoint; it defaults to
`checkpoints/last.pt`.

## Not included

Weights, traces and generated `.npy` intermediates are excluded — they are
large, and the IRIS checkpoint is not mine to redistribute. Train or download
the checkpoint following the original instructions below, then regenerate the
rest with the scripts above.

---

*The original IRIS README follows.*
# Transformers are Sample-Efficient World Models (IRIS)

[Transformers are Sample-Efficient World Models](https://openreview.net/forum?id=vhFu1Acb0xb) <br>
[Vincent Micheli](https://vmicheli.github.io)\*, [Eloi Alonso](https://eloialonso.github.io)\*, [François Fleuret](https://fleuret.org/francois/) <br>
\* Denotes equal contribution


<div align='center'>
  IRIS agent after 100k environment steps, i.e. two hours of real-time experience
  <img alt="IRIS playing on Asterix, Boxing, Breakout, Demon Attack, Freeway, Gopher, Kung Fu Master, Pong" src="assets/iris.gif">
</div>

**tl;dr**

- IRIS is a data-efficient agent trained over millions of imagined trajectories in a world model.
- The world model is composed of a discrete autoencoder and an autoregressive Transformer.
- Our approach casts dynamics learning as a sequence modeling problem, where the autoencoder builds a language of image tokens and the Transformer composes that language over time.


## BibTeX

If you find this code or paper useful, please use the following reference:

```
@inproceedings{
  iris2023,
  title={Transformers are Sample-Efficient World Models},
  author={Vincent Micheli and Eloi Alonso and Fran{\c{c}}ois Fleuret},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=vhFu1Acb0xb}
}
```

## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## Launch a training run

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0 wandb.mode=online
```

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

## Configuration

- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- The simplest way to customize the configuration is to edit these files directly.
- Please refer to [Hydra](https://github.com/facebookresearch/hydra) for more details regarding configuration management.

## Run folder

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   last.pt
|   |   optimizer.pt
|   |   ...
│   │
│   └─── dataset
│       │   0.pt
│       │   1.pt
│       │   ...
│
└─── config
│   |   trainer.yaml
|
└─── media
│   │
│   └─── episodes
│   |   │   ...
│   │
│   └─── reconstructions
│   |   │   ...
│
└─── scripts
|   |   eval.py
│   │   play.sh
│   │   resume.sh
|   |   ...
|
└─── src
|   |   ...
|
└─── wandb
    |   ...
```

- `checkpoints`: contains the last checkpoint of the model, its optimizer and the dataset.
- `media`:
  - `episodes`: contains train / test / imagination episodes for visualization purposes.
  - `reconstructions`: contains original frames alongside their reconstructions with the autoencoder.
- `scripts`: **from the run folder**, you can use the following three scripts.
  - `eval.py`: Launch `python ./scripts/eval.py` to evaluate the run.
  - `resume.sh`: Launch `./scripts/resume.sh` to resume a training that crashed.
  - `play.sh`: Tool to visualize some interesting aspects of the run.
    - Launch `./scripts/play.sh` to watch the agent play live in the environment. If you add the flag `-r`, the left panel displays the original frame, the center panel displays the same frame downscaled to the input resolution of the discrete autoencoder, and the right panel shows the output of the autoencoder (what the agent actually sees).
    - Launch `./scripts/play.sh -w` to unroll live trajectories with your keyboard inputs (i.e. to play in the world model). Note that for faster interaction, the memory of the Transformer is flushed every 20 frames.
    - Launch `./scripts/play.sh -a` to watch the agent play live in the world model. Note that for faster interaction, the memory of the Transformer is flushed every 20 frames.
    - Launch `./scripts/play.sh -e` to visualize the episodes contained in `media/episodes`.
    - Add the flag `-h` to display a header with additional information.
    - Press '`,`' to start and stop recording. The corresponding segment is saved in `media/recordings` in mp4 and numpy formats.
    - Add the flag `-s` to enter 'save mode', where the user is prompted to save trajectories upon completion.

## Results notebook

The folder `results/data/` contains raw scores (for each game, and for each training run) for IRIS and the baselines.

Use the notebook `results/results_iris.ipynb` to reproduce the figures from the paper.

## Pretrained models

Pretrained models are available [here](https://huggingface.co/eloialonso/iris/tree/main/pretrained_models).

- To start a training run from one of these checkpoints, in the section `initialization` of  `config/trainer.yaml`, set `path_to_checkpoint` to the corresponding path, and `load_tokenizer`, `load_world_model`, and `load_actor_critic` to `True`.

- To visualize one of these checkpoints, set `train.id` to the corresponding game in `config/env/default.yaml`, create a `checkpoints` directory and copy the checkpoint to `checkpoints/last.pt`. You can then visualize the agent with `./scripts/play.sh` as described above.

## Credits

- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
