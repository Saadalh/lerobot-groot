#!/usr/bin/env python3
"""Read-only sanity checks for the UR10e source and projected datasets.

Checks (no files are modified):
  1. v3 layout (data / videos / meta/info.json / stats.json / tasks.parquet).
  2. Episode count, lengths, fps, tasks.
  3. One decoded video frame per camera (validates the mp4 files).
  4. Raw 72D action-space audit: per-group std, constant dims, effort check,
     gripper raw range, gripper-mask fraction.
  5. In-memory preview of the 7D projection (same code as project_actions_7d.py):
     gripper range, velocity stats.
  6. If the projected dataset exists: action shape, stats coverage, gripper range.

Usage:
    uv run groot/check_dataset.py [--config groot/train_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GROOT_DIR))
from gripper_encoding import project_action_72_to_7  # noqa: E402

GROUP_NAMES = ["pos[0:12]", "pos_flag[12:24]", "vel[24:36]", "vel_flag[36:48]",
               "eff[48:60]", "eff_flag[60:72]"]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_layout(root: Path, label: str) -> dict:
    print(f"--- {label}: {root} ---")
    for sub in ["data", "videos", "meta/info.json", "meta/stats.json", "meta/tasks.parquet"]:
        ok = (root / sub).exists()
        print(f"  {'OK ' if ok else 'MISS'} {sub}")
        if not ok:
            raise FileNotFoundError(f"{root / sub} missing")
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GROOT_DIR / "train_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proj = cfg["action_projection"]

    src_root = Path(cfg["paths"]["source_dataset_root"])
    info = check_layout(src_root, "source")
    print(f"  version={info['codebase_version']} robot={info['robot_type']} "
          f"episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")

    tasks = pd.read_parquet(src_root / "meta" / "tasks.parquet")
    print(f"  tasks: {len(tasks)} ({tasks.index.tolist()[:3]} ...)")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(str(src_root))
    print(f"  loaded: {ds.num_episodes} episodes, {ds.num_frames} frames")
    sample = ds[0]
    for key in ["observation.images.side_camera", "observation.images.wrist_camera",
                "observation.state", "observation.end_effector_position", "action"]:
        arr = np.asarray(sample[key])
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.3f} max={arr.max():.3f}")
    print(f"  task[0]: {sample['task']!r}")

    # Raw 72D audit over the first episode file (fast; representative).
    data_file = sorted((src_root / "data").glob("chunk-*/file-*.parquet"))[0]
    df = pd.read_parquet(data_file, columns=["action", "observation.state"])
    A = np.stack(df["action"].to_numpy()).astype(np.float64)
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    print(f"  --- raw 72D audit ({data_file.name}, {len(df)} rows) ---")
    for name, blk in zip(GROUP_NAMES, [A[:, 0:12], A[:, 12:24], A[:, 24:36],
                                       A[:, 36:48], A[:, 48:60], A[:, 60:72]]):
        print(f"  {name}: std_mean={blk.std(axis=0).mean():.4f} "
              f"const_dims={int((blk.std(axis=0) == 0).sum())}/12")
    print(f"  gripper raw pos (idx 6): min={A[:, 6].min():.4f} max={A[:, 6].max():.4f}")
    print(f"  gripper mask (idx 18): commanded fraction={(A[:, 18] > 0.5).mean():.3f}")
    print(f"  finger state (idx 6): min={S[:, 6].min():.4f} max={S[:, 6].max():.4f}")

    # In-memory 7D projection preview.
    P = np.stack([
        project_action_72_to_7(
            A[i], S[i],
            arm_velocity_indices=list(proj["arm_velocity_indices"]),
            gripper_position_index=int(proj["gripper_position_index"]),
            gripper_mask_index=int(proj["gripper_mask_index"]),
            gripper_state_index=int(proj["gripper_state_index"]),
            gripper_open_position=float(proj["gripper_open_position"]),
            gripper_closed_position=float(proj["gripper_closed_position"]),
        )
        for i in range(len(A))
    ])
    print(f"  --- 7D projection preview: vel std={P[:, :6].std(axis=0).round(4).tolist()} "
          f"gripper min/max={P[:, 6].min():.3f}/{P[:, 6].max():.3f} ---")

    # Projected dataset, if present.
    new_root = Path(cfg["paths"]["projected_dataset_root"])
    if (new_root / "meta" / "info.json").exists():
        check_layout(new_root, "projected")
        with open(new_root / "meta" / "stats.json") as f:
            stats = json.load(f)
        print(f"  stats keys: {sorted(stats.keys())}")
        assert "openpi.state" not in stats and "openpi.actions" not in stats, "stale openpi stats present"
        ds7 = LeRobotDataset("local/ur10e_basic_7d", root=str(new_root))
        a7 = np.asarray(ds7[0]["action"])
        print(f"  projected sample: shape={a7.shape} gripper={float(a7[6]):.3f}")
        assert a7.shape == (7,)
    else:
        print("  projected dataset not built yet — run project_actions_7d.py")

    print("All checks passed.")


if __name__ == "__main__":
    main()
