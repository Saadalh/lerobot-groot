#!/usr/bin/env python3
"""Project the 72D source action space to the 7D GR00T training action space.

Reads every setting from ``train_config.yaml`` (same folder) and produces a
new LeRobot v3 dataset tree:

    new_root/
        data/videos...   (see below)
        data/chunk-000/file-*.parquet   (rewritten, 7D `action` column)
        meta/info.json                  (action shape/names updated to 7D)
        meta/stats.json                 (recomputed; stale openpi.* keys dropped)
        meta/tasks.parquet              (copied)
        meta/episodes/...               (copied, per-episode action stats refreshed)
        videos/...                      (copied, byte-identical)

Training action (7D) = 6 joint velocities + 1 encoded gripper position.
Gripper encoding follows ``gripper_encoding.py`` (pi0.5-project fallback:
normalized absolute position in [0, 1]; hold frames reuse the current
gripper position).

The source dataset is never modified. Re-running overwrites `new_root`
(`--overwrite` guardrail: refuse unless the flag or YAML allows it).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gripper_encoding import project_action_72_to_7  # noqa: E402

ACTION_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
ACTION_QUANTILE_SUFFIXES = ["q01", "q10", "q50", "q90", "q99"]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rewrite_data_file(src: Path, dst: Path, proj: dict) -> tuple[int, np.ndarray]:
    """Rewrite one data parquet with the projected 7D action.

    Returns (num_rows, stacked 7D actions) for per-episode stat refresh.
    """
    table = pq.read_table(src)
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    n = table.num_rows
    raw_actions = np.stack(cols["action"]).astype(np.float64)
    raw_states = np.stack(cols["observation.state"]).astype(np.float64)
    if raw_actions.shape[1] != 72:
        raise ValueError(f"{src}: expected 72D action, got {raw_actions.shape}")

    out = np.stack(
        [
            project_action_72_to_7(
                raw_actions[i],
                raw_states[i],
                arm_velocity_indices=list(proj["arm_velocity_indices"]),
                gripper_position_index=int(proj["gripper_position_index"]),
                gripper_mask_index=int(proj["gripper_mask_index"]),
                gripper_state_index=int(proj["gripper_state_index"]),
                gripper_open_position=float(proj["gripper_open_position"]),
                gripper_closed_position=float(proj["gripper_closed_position"]),
            )
            for i in range(n)
        ]
    ).astype(np.float32)

    action_type = pa.list_(pa.float32(), 7)
    new_action = pa.array(out.tolist(), type=action_type)
    idx = table.schema.get_field_index("action")
    table = table.set_column(idx, "action", new_action)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return n, out


def refresh_episode_action_stats(ep_path: Path, actions_by_episode: dict[int, np.ndarray]) -> None:
    """Replace stale 72D per-episode action stats with fresh 7D ones."""
    table = pq.read_table(ep_path)
    names = table.schema.names
    if "stats/action/min" not in names:
        return  # nothing to refresh
    df = table.to_pandas()
    for ep_idx, actions in actions_by_episode.items():
        mask = df["episode_index"].to_numpy() == ep_idx
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            continue
        stats = {
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0),
            "count": np.array([len(actions)]),
        }
        quants = np.quantile(actions, ACTION_QUANTILES, axis=0)
        for suffix, q in zip(ACTION_QUANTILE_SUFFIXES, quants):
            stats[suffix] = q
        for key, values in stats.items():
            col = f"stats/action/{key}"
            if col in df.columns:
                cell = values.astype(float).tolist()
                for r in rows:
                    df.at[df.index[r], col] = cell
    # Preserve original (possibly dictionary-encoded) schema where possible.
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ep_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "train_config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting new_root.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    src_root = Path(cfg["paths"]["source_dataset_root"])
    new_root = Path(cfg["paths"]["projected_dataset_root"])
    proj = cfg["action_projection"]

    if len(proj["output_names"]) != 7:
        raise ValueError("action_projection.output_names must list exactly 7 names")
    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Source dataset not found at {src_root}")
    if new_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{new_root} exists — pass --overwrite to rebuild it")
        shutil.rmtree(new_root)

    # 1. Videos (byte-identical) + tasks table.
    shutil.copytree(src_root / "videos", new_root / "videos")
    (new_root / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy(src_root / "meta" / "tasks.parquet", new_root / "meta" / "tasks.parquet")

    # 2. Data files with projected actions.
    src_files = sorted((src_root / "data").glob("chunk-*/file-*.parquet"))
    if not src_files:
        raise FileNotFoundError(f"No data files under {src_root / 'data'}")
    actions_by_episode: dict[int, np.ndarray] = {}
    total_frames = 0
    for src in src_files:
        rel = src.relative_to(src_root / "data")
        n, out = rewrite_data_file(src, new_root / "data" / rel, proj)
        ep_idx = int(pd.read_parquet(src, columns=["episode_index"])["episode_index"].iloc[0])
        actions_by_episode[ep_idx] = out
        total_frames += n
        print(f"rewrote {rel}: {n} frames", flush=True)
    print(f"total frames: {total_frames} across {len(src_files)} files", flush=True)

    # 3. Episode metadata (copied) + refreshed per-episode action stats.
    shutil.copytree(src_root / "meta" / "episodes", new_root / "meta" / "episodes")
    for ep_path in sorted((new_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        refresh_episode_action_stats(ep_path, actions_by_episode)

    # 4. info.json with the 7D action feature.
    with open(src_root / "meta" / "info.json") as f:
        info = json.load(f)
    info["features"]["action"]["shape"] = [7]
    info["features"]["action"]["names"] = list(proj["output_names"])
    data_mb = sum(p.stat().st_size for p in (new_root / "data").rglob("*.parquet")) / 1e6
    info["data_files_size_in_mb"] = round(data_mb)
    with open(new_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # 5. Recompute dataset stats from the rewritten parquets (absolute space;
    #    matches use_relative_actions=false in the training config).
    from lerobot.datasets import LeRobotDataset, recompute_stats

    dataset = LeRobotDataset("local/ur10e_basic_7d", root=str(new_root))
    recompute_stats(dataset, skip_image_video=True)

    # 6. Drop pi0.5-projection leftovers (they describe the old 8D space).
    stats_path = new_root / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    for stale in ["openpi.state", "openpi.actions"]:
        stats.pop(stale, None)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    # 7. Validate.
    check = LeRobotDataset("local/ur10e_basic_7d", root=str(new_root))
    sample = check[0]
    assert tuple(sample["action"].shape) == (7,), sample["action"].shape
    g = sample["action"][6].item()
    assert 0.0 <= g <= 1.0, f"gripper dim out of range: {g}"
    print(f"OK: {check.num_episodes} episodes, {check.num_frames} frames, "
          f"action {tuple(sample['action'].shape)}, sample gripper={g:.3f}")
    print(f"Projected dataset ready at {new_root}")


if __name__ == "__main__":
    main()
