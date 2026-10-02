# LeRobot GR00T Fine-Tuning — UR10e

Fine-tune NVIDIA's **GR00T N1.7** foundation model
(`nvidia/GR00T-N1.7-3B`) on a local UR10e + Robotiq 2F-85 LeRobot v3 dataset
using the official `lerobot-train` pipeline.

## 1. Introduction

This project adapts GR00T N1.7 to a custom UR10e embodiment. It does three things:

1. **Action-space projection** — the collected dataset stores a 72-dim action
   vector (positions + flags + velocities + flags + efforts + flags, most of it
   constant). Training uses a compact **7D action**: 6 joint velocities for the
   physical arm joints plus 1 gripper position, encoded in normalized [0, 1]
   space. The GR00T N1.7 checkpoint ships no gripper specification for the
   `new_embodiment` tag, so the gripper encoding falls back to the one used for
   the pi0.5-DROID fine-tuning in the companion Isaac Sim project
   (`clip((value − 0.0) / 0.376, 0, 1)`, hold-frames reuse the current position).
2. **Dataset sanity checks** — read-only audit of the source dataset (layout,
   episodes, video decoding, action-group statistics, gripper range) and
   validation of the projected dataset.
3. **Training launcher** — a thin, YAML-driven wrapper around the official
   `lerobot-train` pipeline (optimizer/scheduler come from the GR00T training
   preset: AdamW lr=1e-4, HF cosine schedule with 5% warmup, grad-clip 1.0).

> **Dependency:** this project is not standalone. It depends on a local
> LeRobot installation and **must reside in the LeRobot local installation
> directory** (as a `groot/` folder next to the checkout, using its Python
> environment). Clone LeRobot first:
> **https://github.com/huggingface/lerobot**

## 2. Contents

| File                  | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| `train_config.yaml`   | Single source of truth for every tunable (see §4).           |
| `gripper_encoding.py` | Gripper normalize/denormalize + 72D→7D projection (shared).  |
| `project_actions_7d.py` | Builds the 7D training dataset from the 72D source dataset. |
| `train_groot_ur10e.py`  | Launches GR00T fine-tuning via the `lerobot-train` pipeline. |
| `check_dataset.py`    | Read-only dataset health checks (source + projected).        |
| `outputs/`            | Training run outputs (checkpoints, logs). Created on demand. |

## 3. Installation (from scratch)

Run these steps in order on a new machine. They reproduce the exact tree this
project expects — the virtual environment lives in the project root, next to
the checkouts (not nested inside `lerobot/`):

```bash
mkdir <root> && cd <root>                                   # plain dir, no pyproject here
git clone https://github.com/huggingface/lerobot.git lerobot   # provides pyproject.toml + uv.lock
git clone git@github.com:Saadalh/lerobot-groot.git groot        # this repo
uv venv .venv --python 3.12                            # root venv (one-time)
cd lerobot                                                 # sync runs from the checkout...
VIRTUAL_ENV=$PWD/../.venv uv sync --active --locked --inexact \
  --extra groot --extra training --extra test              # ...but installs into ../.venv
```

Result:

```bash
<root>/
  lerobot/      # the https://github.com/huggingface/lerobot checkout
  groot/        # this repo
  .venv/        # project environment (123 locked packages)
```

Why these flags:

* `--active` + `VIRTUAL_ENV=…` — sync into the root venv instead of uv's
  default `<project>/.venv`. (`uv sync` has no venv-path flag; targeting via
  the active environment or `--python ../.venv/bin/python` is the supported
  mechanism.) Set inline for one command only — no shell state is changed.
* `--locked` — versions come from `lerobot/uv.lock`, so every machine
  converges to the same stack.
* `--inexact` — additive: install what's locked without uninstalling anything
  else. A bare `uv sync` is exact-state and prunes every package outside the
  requested extras (this once deleted ~200 packages here).
* `--extra test` — ships pytest so the verification gate below can run.

Then authenticate for model/HF access as needed: `hf auth login`
(base-model download) and `wandb login` (if `wandb.enable` is true).

This procedure was validated end-to-end (fresh `.venv` + checks + tests green)
before being written down. Occasional exact cleanup:
`VIRTUAL_ENV=$PWD/../.venv uv sync --active --locked --project . --extra all`
removes stale packages `--inexact` never prunes.

No other installation is required — there is no separate `gr00t` pip package;
GR00T N1.7 support is native to LeRobot (the `lerobot[groot]` extra).

Verify a fresh install from `<root>` before training:

```bash
.venv/bin/python -c "import lerobot, yaml; print('lerobot OK')"
.venv/bin/python groot/check_dataset.py   # dataset audit (needs the dataset paths in §4)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  lerobot/tests/policies/groot/test_groot_training_optim_contract.py -q
```

## 4. Usage

All commands run from `<install-root>` (the directory containing `groot/`
and `.venv/`). All tunables live in `groot/train_config.yaml` — edit that
file, never the scripts.

### Critical variables to configure before any training

| YAML key | Why it matters | Default |
| -------- | -------------- | ------- |
| `paths.source_dataset_root` | 72D LeRobot v3 source dataset (never modified) | `…/isaac_simulations/ur10e_basic/dataset_72_2step_clean_action` |
| `paths.projected_dataset_root` | Where the 7D training dataset is written | `…/dataset_72_2step_clean_action_7d` |
| `paths.output_dir` | Checkpoints/logs for the run | `groot/outputs/ur10e_gr00t17_7d` |
| `policy.base_model_path` | Base checkpoint to fine-tune | `nvidia/GR00T-N1.7-3B` |
| `policy.repo_id` / `policy.push_to_hub` | Hub upload of the fine-tuned model (`null`/false = local only) | local only |
| `train.steps` / `train.batch_size` | Budget vs VRAM (batch 8 ≈ fits 16 GB with bf16) | 20000 / 8 |
| `train.seed` | Reproducibility | 42 |
| `wandb.enable` / `wandb.project` | Experiment tracking | true / `ur10e-gr00t` |
| `action_projection.*` | 72D→7D index map + gripper open/closed (0.0 / 0.376) | see file |
| `dataset.episodes` | Subset for smoke tests (`null` = all 72) | `null` |

### Run commands

```bash
# 1. Audit the source dataset (read-only, always safe)
.venv/bin/python groot/check_dataset.py

# 2. Build the 7D training dataset (re-run with --overwrite after config edits)
.venv/bin/python groot/project_actions_7d.py --overwrite

# 3. Validate the full stack: 20 steps, no wandb, no Hub push
.venv/bin/python groot/train_groot_ur10e.py --smoke-test

# 4. Full fine-tuning run
.venv/bin/python groot/train_groot_ur10e.py
```

To train on a different checkpoint, horizon, or batch size, change
`policy.base_model_path`, `policy.chunk_size` / `policy.n_action_steps`,
or `train.batch_size` in `train_config.yaml` and re-run steps 3–4 (step 2
only needs re-running if `action_projection` changed). Note
`policy.use_relative_actions` is intentionally `false`: velocity targets are
already motion deltas, so relative mode would difference velocities against
positions. Enable it only when training on absolute position actions.

## 5. Simulation evaluation

Runs the fine-tuned 7D policy in the UR10e collection simulation
(the Isaac Sim project the dataset was generated from) for a configured
number of control steps, using the same scene setup, cameras, timing, and
action application as data collection.

Prerequisites: a trained checkpoint under `outputs/`, a working
`ur10e_basic` simulation project with IsaacLab, and free GPU memory —
the policy server alone needs most of a 16 GB card, so close the Isaac Sim
GUI and other GPU apps first.

All settings live in `groot/eval_config.yaml`:

| Key | Purpose | Default |
| --- | ------- | ------- |
| `policy.checkpoint_dir` | Fine-tuned checkpoint to evaluate | `outputs/ur10e_gr00t17_7d/checkpoints/last/pretrained_model` |
| `execution.total_steps` | Control steps to execute before stopping | 3000 |
| `execution.chunk_samples` | Chunk rows executed per inference (1 = closed-loop) | 1 |
| `sim.task_index` / `sim.episode_index` / `sim.seed` | Scene setup, same meaning as the sim's test mode | 0 / 0 / 42 |
| `sim.plot_gripper` | Save per-step gripper signals next to the sim project | false |
| `server.port` | Local port for the policy service | 5555 |

Run in two terminals (in this order):

```bash
# Terminal 1 — policy service (wait for "Serving on ...")
.venv/bin/python groot/gr00t_policy_server.py

# Terminal 2 — simulation evaluation
/home/rahmlab/projects/IsaacLab/isaaclab.sh -p \
    /home/rahmlab/projects/lerobot/groot/eval_gr00t_sim.py
```

Both commands accept `--config <path>` to use a different config file.
For a quick check, set `execution.total_steps: 20` in a copy of the config
and pass it to both commands. The run ends automatically after
`total_steps` executed steps; console output reports per-inference timing
and progress.

## 5. Remarks

GR00T N1.7 requires Hugging Face authentication: its tokenizer backbone
(`nvidia/Cosmos-Reason2-2B`) is gated and needs accepting terms at
https://huggingface.co/nvidia/Cosmos-Reason2-2B. Without it training fails
on the first batch with `401 GatedRepoError`. The `hf` CLI lives only in
the project venv, so from `<root>` run:

```bash
.venv/bin/hf auth login
```

then re-run the smoke test.
