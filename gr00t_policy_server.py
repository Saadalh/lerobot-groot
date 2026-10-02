#!/usr/bin/env python3
"""Serve GR00T N1.7 action chunks to the simulation evaluator over TCP.

Run with the project .venv (NOT under Isaac):

    .venv/bin/python groot/gr00t_policy_server.py [--config groot/eval_config.yaml]

Loads the fine-tuned checkpoint once, then serves one (T, 7) chunk per
observation request using the rollout inference path: NumPy observation ->
tensors -> preprocessor -> predict_action_chunk -> postprocessor.

Protocol details live in eval_protocol.py (stdlib sockets, no extra deps).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback
from pathlib import Path

# Reduce CUDA fragmentation: must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

GROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GROOT_DIR))
from eval_protocol import (  # noqa: E402
    decode_request,
    encode_chunk,
    recv_message,
    send_message,
)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GROOT_DIR / "eval_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    checkpoint = Path(cfg["policy"]["checkpoint_dir"])
    device = torch.device(cfg["policy"].get("device", "cuda"))
    for required in ["config.json", "model.safetensors"]:
        if not (checkpoint / required).exists():
            raise FileNotFoundError(f"{checkpoint} is missing {required}")

    print(f"Loading policy from {checkpoint} ...", flush=True)
    policy = GrootPolicy.from_pretrained(str(checkpoint))
    if bool(cfg["policy"].get("bf16_weights", True)):
        # Training checkpoints store fp32 master weights (~12 GB); inference
        # runs the bf16 compute path, so downcast once to fit 16 GB cards.
        policy._groot_model.to(torch.bfloat16)  # noqa: SLF001
    # The fp32 weights load directly onto CUDA; downcasting leaves the old
    # blocks in the caching pool, so release them back to the driver.
    torch.cuda.empty_cache()
    total, by_dtype = 0, {}
    for p in policy.parameters():
        total += p.numel()
        by_dtype[str(p.dtype)] = by_dtype.get(str(p.dtype), 0) + p.numel()
    print(f"policy params: {total / 1e9:.2f}B {by_dtype}", flush=True)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    chunk_len = int(policy.config.n_action_steps)
    print(f"Ready: chunk={chunk_len}, device={device}", flush=True)

    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Serving on {host}:{port} ...", flush=True)
        served = 0
        while True:
            conn, addr = server.accept()
            print(f"Sim connected from {addr}", flush=True)
            with conn:
                while True:
                    try:
                        header, payload = recv_message(conn)
                    except ConnectionError:
                        print("Sim disconnected; waiting for next client.", flush=True)
                        break
                    try:
                        obs = decode_request(header, payload)
                        task = obs.pop("task")
                        assert isinstance(task, str)
                        observation = prepare_observation_for_inference(
                            {k: v for k, v in obs.items()}, device, task
                        )
                        with torch.inference_mode():
                            observation = preprocessor(observation)
                            chunk = policy.predict_action_chunk(observation)
                            chunk = postprocessor(chunk)
                        chunk_np = chunk.squeeze(0).detach().to("cpu").float().numpy()
                        send_message(conn, *encode_chunk(chunk_np))
                        served += 1
                        if served == 1 or served % 100 == 0:
                            print(f"served chunk #{served} shape={chunk_np.shape}", flush=True)
                    except Exception:  # noqa: BLE001 - report back, keep serving
                        send_message(conn, {"error": traceback.format_exc(limit=3)})


if __name__ == "__main__":
    main()
