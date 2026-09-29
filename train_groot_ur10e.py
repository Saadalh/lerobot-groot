#!/usr/bin/env python3
"""Fine-tune GR00T N1.7 on the projected 7D UR10e dataset.

Launcher around the official ``lerobot-train`` pipeline
(``lerobot.scripts.lerobot_train.train``): it reads ``train_config.yaml``,
runs pre-flight checks, builds ``TrainPipelineConfig`` with a ``GrootConfig``
policy, and starts training. Optimizer/scheduler come from the policy's
training preset (AdamW lr=1e-4, HF cosine 5% warmup, grad-clip 1.0).

Usage:
    uv run groot/train_groot_ur10e.py [--config groot/train_config.yaml] [--smoke-test]

``--smoke-test`` overrides steps/batch/log/save-freq to tiny values and
disables wandb + Hub push to validate the whole stack quickly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

GROOT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preflight(cfg: dict) -> None:
    src = Path(cfg["paths"]["source_dataset_root"])
    proj_root = Path(cfg["paths"]["projected_dataset_root"])
    for name, path in [("source_dataset_root", src), ("projected_dataset_root", proj_root)]:
        if not (path / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"{name}={path} is not a LeRobot dataset. "
                + (
                    "Run `uv run groot/project_actions_7d.py --overwrite` first."
                    if name == "projected_dataset_root"
                    else ""
                )
            )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GR00T fine-tuning but is not available.")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_mem:.1f} GB)", flush=True)
    if total_mem < 15 and cfg["train"]["batch_size"] > 4:
        print("WARNING: <15 GB VRAM with batch_size > 4 may OOM — lower train.batch_size.", flush=True)


def build_train_config(cfg: dict, smoke_test: bool = False):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.groot.configuration_groot import GrootConfig

    d, p, t, w = cfg["dataset"], cfg["policy"], cfg["train"], cfg["wandb"]

    dataset_kwargs: dict = dict(
        repo_id=d["repo_id"],
        root=d.get("root") or str(Path(cfg["paths"]["projected_dataset_root"])),
        episodes=d.get("episodes"),
        exclude_episodes=d.get("exclude_episodes") or None,
        eval_split=float(d.get("eval_split", 0.0)),
    )
    if d.get("video_backend"):
        dataset_kwargs["video_backend"] = d["video_backend"]
    dataset_cfg = DatasetConfig(**dataset_kwargs)
    if d.get("image_transforms_enable"):
        dataset_cfg.image_transforms.enable = True

    policy_cfg = GrootConfig(
        base_model_path=p["base_model_path"],
        embodiment_tag=p.get("embodiment_tag", "new_embodiment"),
        chunk_size=int(p.get("chunk_size", 16)),
        n_action_steps=int(p.get("n_action_steps", 16)),
        use_relative_actions=bool(p.get("use_relative_actions", False)),
        relative_exclude_joints=list(p.get("relative_exclude_joints", [])),
        use_bf16=bool(p.get("use_bf16", True)),
        tune_llm=bool(p.get("tune_llm", False)),
        tune_visual=bool(p.get("tune_visual", False)),
        tune_projector=bool(p.get("tune_projector", True)),
        tune_diffusion_model=bool(p.get("tune_diffusion_model", True)),
        tune_vlln=bool(p.get("tune_vlln", True)),
        tune_top_llm_layers=int(p.get("tune_top_llm_layers", 0)),
        device=p.get("device", "cuda"),
        push_to_hub=bool(p.get("push_to_hub", False)),
        repo_id=p.get("repo_id"),
        batch_size=int(t["batch_size"]),
        max_steps=int(t["steps"]),
    )

    wandb_cfg = WandBConfig(
        enable=bool(w.get("enable", True)),
        project=w.get("project", "ur10e-gr00t"),
        disable_artifact=bool(w.get("disable_artifact", True)),
        mode=w.get("mode"),
    )

    steps, batch, log_freq, save_freq = (
        (t["steps"], t["batch_size"], t["log_freq"], t["save_freq"])
        if not smoke_test
        else (20, 2, 1, 10)
    )
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=Path(cfg["paths"]["output_dir"]),
        job_name=t.get("job_name", "ur10e_gr00t17_7d"),
        seed=int(t.get("seed", 42)),
        num_workers=int(t.get("num_workers", 4)),
        batch_size=int(batch),
        steps=int(steps),
        log_freq=int(log_freq),
        save_freq=int(save_freq),
        use_policy_training_preset=bool(t.get("use_policy_training_preset", True)),
        env_eval_freq=int(t.get("env_eval_freq", 0)),
        eval_steps=int(t.get("eval_steps", 0)),
        resume=bool(t.get("resume", False)),
        wandb=wandb_cfg,
    )
    if smoke_test:
        train_cfg.wandb.enable = False
        train_cfg.wandb.mode = "disabled"
        train_cfg.policy.push_to_hub = False
    return train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GROOT_DIR / "train_config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny 20-step run to validate the stack.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preflight(cfg)
    train_cfg = build_train_config(cfg, smoke_test=args.smoke_test)

    print("Resolved training config:", flush=True)
    print(f"  dataset: {train_cfg.dataset.repo_id} @ {train_cfg.dataset.root}", flush=True)
    print(f"  policy: groot {train_cfg.policy.base_model_path} "
          f"(embodiment={train_cfg.policy.embodiment_tag}, "
          f"chunk={train_cfg.policy.chunk_size}, "
          f"relative={train_cfg.policy.use_relative_actions})", flush=True)
    print(f"  steps={train_cfg.steps} batch={train_cfg.batch_size} "
          f"seed={train_cfg.seed} output={train_cfg.output_dir}", flush=True)

    from lerobot.scripts.lerobot_train import train

    train(train_cfg)


if __name__ == "__main__":
    sys.exit(main())
