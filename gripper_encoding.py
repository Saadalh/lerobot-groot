#!/usr/bin/env python3
"""Shared gripper encoding for the UR10e GR00T fine-tuning scripts.

Encoding decision
-----------------
GR00T N1.7 ships no gripper specification for the ``new_embodiment`` tag: for a
Hugging Face Hub base model the processor takes the generic path
(``_load_n1_7_checkpoint_processor_assets`` returns ``None`` for non-local
checkpoints) and builds normalization stats from the training dataset. The
checkpoint therefore contributes no gripper encoding.

Fallback (per request): the encoding used when fine-tuning pi0.5-DROID in
``isaac_simulations/ur10e_basic`` (see its ``src/droid_projection.py`` and
``config/train_config.yaml``):

    encoded = clip((value - OPEN) / (CLOSED - OPEN), 0.0, 1.0)

with ``OPEN = 0.0`` and ``CLOSED = 0.376`` (physical finger-joint radians).
Decoding inverts it: ``position = OPEN + clip(action, 0, 1) * (CLOSED - OPEN)``.

Action convention used here (7D): the first six dims are raw joint velocities
for the six physical UR10e arm joints; dim 6 is the encoded absolute gripper
position. Frames whose gripper mask is false reuse the current (hold) gripper
position, mirroring ``project_controls`` in the pi0.5 project.
"""

from __future__ import annotations

import numpy as np

# 72D raw-action schema of the source dataset (see meta/info.json).
POS_DIM = 12
FLAG_OFFSET = 12  # position/velocity/effort masks live 12 columns after values.
VEL_DIM_START = 24
EFF_DIM_START = 48


def normalize_gripper_position(value: float, opened: float, closed: float) -> float:
    """Map an absolute finger-joint position into normalized [0, 1] space."""
    opened = float(opened)
    closed = float(closed)
    if closed <= opened:
        raise ValueError("gripper_closed_position must be greater than gripper_open_position")
    return float(np.clip((float(value) - opened) / (closed - opened), 0.0, 1.0))


def denormalize_gripper_position(action: float, opened: float, closed: float) -> float:
    """Map a normalized gripper command back to an absolute joint position."""
    opened = float(opened)
    closed = float(closed)
    if closed <= opened:
        raise ValueError("gripper_closed_position must be greater than gripper_open_position")
    return opened + float(np.clip(float(action), 0.0, 1.0)) * (closed - opened)


def project_action_72_to_7(
    raw_action: np.ndarray,
    raw_state: np.ndarray,
    *,
    arm_velocity_indices: list[int],
    gripper_position_index: int,
    gripper_mask_index: int,
    gripper_state_index: int,
    gripper_open_position: float,
    gripper_closed_position: float,
) -> np.ndarray:
    """Project one 72D raw action (+12D raw state) to the 7D training action.

    Args:
        raw_action: shape (72,) — position block, flags, velocity block, ...
        raw_state: shape (12,) — current joint positions (index 6 = finger joint).
        arm_velocity_indices: six indices into ``raw_action`` (the velocity block).
        gripper_position_index: index of the absolute finger-joint command.
        gripper_mask_index: flag index; command used only when value > 0.5.
        gripper_state_index: state index of the current finger-joint position.
        gripper_open_position / gripper_closed_position: encoding range.

    Returns:
        shape (7,) float32: [6 joint velocities, encoded gripper position].
    """
    raw_action = np.asarray(raw_action, dtype=np.float64)
    raw_state = np.asarray(raw_state, dtype=np.float64)
    if raw_action.shape != (72,):
        raise ValueError(f"Expected raw action shape (72,), got {raw_action.shape}")
    if raw_state.shape[0] < 7:
        raise ValueError(f"Expected raw state with >= 7 dims, got {raw_state.shape}")

    arm_vel = raw_action[np.asarray(arm_velocity_indices)].astype(np.float64)
    commanded = bool(raw_action[int(gripper_mask_index)] > 0.5)
    if commanded:
        gripper_raw = float(raw_action[int(gripper_position_index)])
    else:
        gripper_raw = float(raw_state[int(gripper_state_index)])
    gripper = normalize_gripper_position(
        gripper_raw, gripper_open_position, gripper_closed_position
    )
    return np.concatenate([arm_vel, [gripper]]).astype(np.float32)
