#!/usr/bin/env python3
"""Length-prefixed socket protocol between the sim client and the policy server.

Stdlib-only (socket/struct/json) plus NumPy arrays — importable from both the
Isaac Python (sim side) and the project .venv (policy side).

Request (client -> server):
    [8-byte big-endian header length][JSON header][raw buffers]
    header = {"images": {"side_camera": {"shape": [H, W, 3], "dtype": "uint8",
                                         "nbytes": N}, ...},
              "state": {"shape": [12], "dtype": "float32", "nbytes": N},
              "eef": {"shape": [3], "dtype": "float32", "nbytes": N},
              "task": str}
    buffers follow in order: side_camera, wrist_camera, state, eef.

Response (server -> client):
    [8-byte big-endian header length][JSON header][raw float32 chunk]
    header = {"action_chunk": {"shape": [T, 7], "dtype": "float32", "nbytes": N}}
    or {"error": str} with no payload.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

HEADER_FMT = ">Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

IMAGE_KEYS = ("side_camera", "wrist_camera")


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        piece = sock.recv(n - len(buf))
        if not piece:
            raise ConnectionError("Socket closed mid-message")
        buf.extend(piece)
    return bytes(buf)


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(HEADER_FMT, len(raw)) + raw + payload)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    (header_len,) = struct.unpack(HEADER_FMT, _recvall(sock, HEADER_SIZE))
    header = json.loads(_recvall(sock, header_len).decode("utf-8"))
    payload = b""
    if "error" not in header:
        total = 0
        for section in ("images", "state", "eef", "action_chunk"):
            entries = header.get(section)
            if entries is None:
                continue
            if isinstance(entries, dict) and "nbytes" in entries:
                total += int(entries["nbytes"])
            elif isinstance(entries, dict):
                total += sum(int(v["nbytes"]) for v in entries.values())
        payload = _recvall(sock, total) if total else b""
    return header, payload


def encode_request(
    side_camera: np.ndarray,
    wrist_camera: np.ndarray,
    state: np.ndarray,
    eef: np.ndarray,
    task: str,
) -> tuple[dict, bytes]:
    """Pack one observation. Images: uint8 HWC; state/eef: float32."""
    arrays = {
        "side_camera": np.ascontiguousarray(side_camera, dtype=np.uint8),
        "wrist_camera": np.ascontiguousarray(wrist_camera, dtype=np.uint8),
        "state": np.ascontiguousarray(state, dtype=np.float32),
        "eef": np.ascontiguousarray(eef, dtype=np.float32),
    }
    header: dict = {
        "images": {
            key: {
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
                "nbytes": arrays[key].nbytes,
            }
            for key in IMAGE_KEYS
        },
        "state": {
            "shape": list(arrays["state"].shape),
            "dtype": "float32",
            "nbytes": arrays["state"].nbytes,
        },
        "eef": {
            "shape": list(arrays["eef"].shape),
            "dtype": "float32",
            "nbytes": arrays["eef"].nbytes,
        },
        "task": task,
    }
    payload = b"".join(
        arrays[key].tobytes() for key in ("side_camera", "wrist_camera", "state", "eef")
    )
    return header, payload


def decode_request(header: dict, payload: bytes) -> dict[str, np.ndarray | str]:
    sizes = [header["images"][k]["nbytes"] for k in IMAGE_KEYS]
    sizes += [header["state"]["nbytes"], header["eef"]["nbytes"]]
    parts, offset = [], 0
    for n in sizes:
        parts.append(payload[offset : offset + n])
        offset += n
    (side_raw, wrist_raw, state_raw, eef_raw) = parts
    # .copy(): payload bytes are immutable; torch requires writable arrays.
    return {
        "observation.images.side_camera": np.frombuffer(
            side_raw, dtype=np.uint8
        ).reshape(header["images"]["side_camera"]["shape"]).copy(),
        "observation.images.wrist_camera": np.frombuffer(
            wrist_raw, dtype=np.uint8
        ).reshape(header["images"]["wrist_camera"]["shape"]).copy(),
        "observation.state": np.frombuffer(state_raw, dtype=np.float32).reshape(
            header["state"]["shape"]
        ).copy(),
        "observation.end_effector_position": np.frombuffer(
            eef_raw, dtype=np.float32
        ).reshape(header["eef"]["shape"]).copy(),
        "task": str(header["task"]),
    }


def encode_chunk(chunk: np.ndarray) -> tuple[dict, bytes]:
    """Pack a (T, 7) float32 action chunk for the response."""
    arr = np.ascontiguousarray(chunk, dtype=np.float32)
    header = {
        "action_chunk": {
            "shape": list(arr.shape),
            "dtype": "float32",
            "nbytes": arr.nbytes,
        }
    }
    return header, arr.tobytes()


def decode_chunk(header: dict, payload: bytes) -> np.ndarray:
    if "error" in header:
        raise RuntimeError(f"Policy server error: {header['error']}")
    meta = header["action_chunk"]
    return np.frombuffer(payload, dtype=np.float32).reshape(meta["shape"]).copy()
