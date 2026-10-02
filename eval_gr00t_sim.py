#!/usr/bin/env python3
"""Evaluate the fine-tuned GR00T policy in the UR10e collection simulation.

Run under Isaac Python (NOT the project .venv), from any working directory:

    /home/rahmlab/projects/IsaacLab/isaaclab.sh -p \
        /home/rahmlab/projects/lerobot/groot/eval_gr00t_sim.py \
        [--config /home/rahmlab/projects/lerobot/groot/eval_config.yaml]

Structure mirrors src/test_pi05.py in the simulation project: the same
environment, cameras, timing, step_simulation(), and optional gripper-signal
recording. The only difference is the policy call, which goes to
gr00t_policy_server.py over TCP (the two interpreters cannot share a process).

Execution stops after execution.total_steps executed control steps.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from time import perf_counter

GROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GROOT_DIR))

import yaml  # noqa: E402

from eval_protocol import decode_chunk, encode_request, recv_message, send_message  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GROOT_DIR / "eval_config.yaml")
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config)

    sys.path.insert(0, cfg["sim"]["project_root"])
    from src import sim  # noqa: E402

    # Route all tunables through eval_config.yaml: rebuild argv for
    # launch_simulation(), which parses it the same way main.py --test does.
    argv = [sys.argv[0]]
    argv += ["--task-index", str(cfg["sim"]["task_index"])]
    argv += ["--episode-index", str(cfg["sim"]["episode_index"])]
    if cfg["sim"].get("seed") is not None:
        argv += ["--seed", str(cfg["sim"]["seed"])]
    if cfg["sim"].get("tcp_y_sign") is not None:
        argv += ["--tcp-y-sign", str(cfg["sim"]["tcp_y_sign"])]
    if cfg["sim"].get("plot_gripper"):
        argv += ["--plot-gripper"]
    argv += list(cfg["sim"].get("extra_argv", []))
    sys.argv = argv

    total_steps = int(cfg["execution"]["total_steps"])
    chunk_samples = int(cfg["execution"]["chunk_samples"])
    vel_to_droid = 1.0 / (
        float(cfg["execution"]["control_hz"])
        * float(cfg["execution"]["droid_action_scale"])
    )
    if total_steps < 1 or chunk_samples < 1:
        raise ValueError("execution.total_steps and chunk_samples must be >= 1")

    sock = socket.create_connection((cfg["server"]["host"], int(cfg["server"]["port"])))
    print(f"Connected to policy server at {cfg['server']['host']}:{cfg['server']['port']}",
          flush=True)
    executed_steps = 0
    gripper_recorder = None

    def finalize_gripper_debug() -> None:
        if gripper_recorder is None or len(gripper_recorder) == 0:
            return
        from src.gripper_debug import GripperDebugRecorder  # noqa: E402

        launch_args = sim.get_launch_args()
        out_dir = Path(cfg["sim"]["project_root"]) / "gripper_debug"
        png_path, csv_path = gripper_recorder.finalize(
            task_index=launch_args.task_index,
            episode_index=launch_args.episode_index,
            seed=launch_args.seed,
            out_dir=out_dir,
        )
        print(f"Saved gripper debug plot: {png_path}", flush=True)
        print(f"Saved gripper debug CSV: {csv_path}", flush=True)

    def capture_observations() -> bool:
        nonlocal executed_steps, gripper_recorder
        import numpy as np  # noqa: E402

        images = sim.get_images()
        if not images:
            return False
        side, wrist = (np.asarray(images[0]), np.asarray(images[1]))

        # Same raw signals as data collection: full 12-joint vector and the
        # end-effector world position (cf. collect_cube_data.py).
        robot = sim._environment.robot  # noqa: SLF001 - read-only env access
        state = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        eef, _ = robot.end_effector.get_world_pose()
        eef = np.asarray(eef, dtype=np.float32)
        task = sim.get_task_prompt()

        t0 = perf_counter()
        header, payload = encode_request(side, wrist, state, eef, task)
        send_message(sock, header, payload)
        resp_header, resp_payload = recv_message(sock)
        chunk = decode_chunk(resp_header, resp_payload)
        print(f"Model exec {executed_steps + 1}: {(perf_counter() - t0) * 1000:.0f}ms "
              f"chunk={chunk.shape}", flush=True)

        # Convert the 7D policy chunk to the 8D rows step_simulation expects:
        # velocities become per-step position targets via the DROID scale,
        # the synthetic joint stays 0, the gripper command passes through.
        adapted = np.zeros((chunk.shape[0], 8), dtype=np.float64)
        adapted[:, :6] = chunk[:, :6] * vel_to_droid
        adapted[:, 7] = np.clip(chunk[:, 6], 0.0, 1.0)

        if gripper_recorder is None and sim.get_launch_args() is not None:
            launch_args = sim.get_launch_args()
            if launch_args is not None and launch_args.plot_gripper:
                from src.gripper_debug import GripperDebugRecorder  # noqa: E402

                gripper_recorder = GripperDebugRecorder(
                    opened=float(cfg["gripper"]["open_position"]),
                    closed=float(cfg["gripper"]["closed_position"]),
                )
        inference_id = (
            gripper_recorder.new_inference() if gripper_recorder is not None else None
        )
        executed_steps += sim.step_simulation(
            adapted,
            chunk_samples,
            gripper_recorder=gripper_recorder,
            inference_id=inference_id,
        )
        print(f"Executed {executed_steps}/{total_steps} action steps", flush=True)
        if executed_steps >= total_steps:
            finalize_gripper_debug()
            return True
        return False

    try:
        sim.launch_simulation(capture_observations)
    finally:
        finalize_gripper_debug()
        sock.close()


if __name__ == "__main__":
    main()
