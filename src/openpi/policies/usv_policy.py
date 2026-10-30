"""USV v1: raw 33D state, one to three RGB views, and two thrust commands.

Previous-command state fields are already divided by 15 N by the recorder.
Fixed engineering scales deliberately preserve presence bits and missing slots.
"""

import dataclasses

import numpy as np

import openpi.transforms as transforms

STATE_DIM = 33
ACTION_DIM = 2
THRUST_LIMIT_N = 15.0
POSITION_SCALE_M = 32.0
CAMERAS = {
    "front": "base_0_rgb",
    "left": "left_wrist_0_rgb",
    "right": "right_wrist_0_rgb",
}


def parse_image(value):
    """LeRobot reads CHW float [0,1]; live clients send HWC uint8."""
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError("Images must have three dimensions.")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] != 3:
        raise ValueError("Images require three RGB channels.")
    if np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("Floating images must be finite in [0,1].")
        value = np.rint(value * 255).astype(np.uint8)
    if value.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB or normalized floating image.")
    return value


def normalize_state(state):
    state = np.asarray(state, dtype=np.float32).copy()
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("USV state must be a finite (33,) vector.")
    slots = state[9:].reshape(8, 3)
    if not np.isin(slots[:, 2], [0, 1]).all():
        raise ValueError("Buoy presence must be exactly 0 or 1.")
    if np.any(slots[slots[:, 2] == 0, :2] != 0):
        raise ValueError("Absent buoy coordinates must be zero.")
    scales = np.array([2, 2, 1, np.pi, np.pi, 1, 1, 1, 1] + [POSITION_SCALE_M, POSITION_SCALE_M, 1] * 8)
    state /= scales
    if np.any(np.abs(state) > 1):
        raise ValueError("USV state exceeds engineering scales; do not silently saturate positions.")
    return state.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class UsvInputs(transforms.DataTransformFn):
    state_version: int = 1

    def __call__(self, data):
        raw_state = np.asarray(data["state"])
        if self.state_version == 1:
            state = normalize_state(raw_state)
        elif self.state_version == 2:
            from usv_vla.gps_state import normalize_extra
            if raw_state.shape != (39,):
                raise ValueError("USV GPS state must have 39 dimensions")
            state = np.concatenate((normalize_state(raw_state[:33]), normalize_extra(raw_state[33:])))
        elif self.state_version == 3:
            from usv_vla.gps_state import gps_position_state
            if 'gps_e7' in data:
                gps = np.asarray(data['gps_e7'])
                if gps.shape != (2,) or not np.issubdtype(gps.dtype, np.integer):
                    raise ValueError('GPS input requires two integer coordinates at 1e-7 degrees')
                state = gps_position_state(normalize_state(raw_state), gps.astype(np.float64) / 10_000_000)
            elif raw_state.shape == (35,) and raw_state.dtype == np.float64:
                state = gps_position_state(normalize_state(raw_state[:33]), raw_state[33:])
            else:
                raise ValueError('GPS state requires 33 values plus integer GPS or float64 state[35]')
        else:
            raise ValueError("Unknown USV state version")
        incoming = data["images"]
        if "front" not in incoming or set(incoming) - set(CAMERAS):
            raise ValueError("USV images require front, with optional left/right views.")
        front = parse_image(incoming["front"])
        images, masks = {}, {}
        for source, dest in CAMERAS.items():
            value = parse_image(incoming[source]) if source in incoming else np.zeros_like(front)
            images[dest], masks[dest] = value, np.bool_(source in incoming)
        result = {"state": state, "image": images, "image_mask": masks}
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError("Actions must contain finite [left_N, right_N].")
            if np.any(np.abs(actions) > THRUST_LIMIT_N):
                raise ValueError("Recorded action exceeds executed thrust limits.")
            result["actions"] = actions / THRUST_LIMIT_N
        return result


@dataclasses.dataclass(frozen=True)
class UsvOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError("Refusing non-finite or non-2D motor output.")
        return {"actions": np.clip(actions, -1, 1) * THRUST_LIMIT_N}
