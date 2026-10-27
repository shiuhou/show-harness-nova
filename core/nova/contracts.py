"""Immutable observation and reviewed semantic translation contract (SI units)."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from inspect_robots_dobot.camera import Frame
from inspect_robots_dobot.types import RobotSnapshot

MOVE_TOKENS = ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN")
M1_TOKENS = (*MOVE_TOKENS, "DONE")
RGB = npt.NDArray[np.uint8]
Vector = tuple[float, float, float]

EEF_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["EEF_MOVE", "GRASP", "RELEASE", "DONE"]},
        "delta_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "reasoning": {"type": "string"},
    },
    "required": ["decision", "delta_mm", "reasoning"],
    "additionalProperties": False,
}


def eef_delta(payload: Any, maximum_m: float) -> Vector:
    """Reject malformed/oversized output; never clamp or split model actions."""
    if not isinstance(payload, dict) or set(payload) != {"decision", "delta_mm", "reasoning"}:
        raise ValueError("invalid EEF response fields")
    token, values = payload["decision"], payload["delta_mm"]
    if token not in ("EEF_MOVE", "GRASP", "RELEASE", "DONE") or not isinstance(
        payload["reasoning"], str
    ):
        raise ValueError("invalid EEF decision")
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
    ):
        raise ValueError("EEF delta requires three finite numbers in millimetres")
    delta = (values[0] / 1000, values[1] / 1000, values[2] / 1000)
    length = math.hypot(*delta)
    if not math.isfinite(maximum_m) or maximum_m <= 0 or length > maximum_m:
        raise ValueError("EEF translation exceeds displacement budget")
    if (token == "EEF_MOVE" and length == 0) or (token != "EEF_MOVE" and length != 0):
        raise ValueError("EEF_MOVE requires displacement; other actions require zero delta")
    return delta


def strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def png_bytes(rgb: RGB) -> bytes:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("expected nonempty HWC uint8 RGB")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class ImageObservation:
    camera: str
    png: bytes
    captured_at: float
    sequence: int
    generation: int
    timestamp_source: str

    @classmethod
    def from_frame(cls, camera: str, frame: Frame) -> ImageObservation:
        return cls(
            camera,
            png_bytes(frame.rgb),
            frame.timestamp,
            frame.sequence,
            frame.generation,
            frame.timestamp_source,
        )

    def rgb(self) -> RGB:
        return np.asarray(Image.open(io.BytesIO(self.png)).convert("RGB"), dtype=np.uint8)

    def metadata(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "sha256": hashlib.sha256(self.png).hexdigest(),
            "captured_at": self.captured_at,
            "sequence": self.sequence,
            "generation": self.generation,
            "timestamp_source": self.timestamp_source,
        }


@dataclass(frozen=True)
class Observation:
    episode_id: str
    observation_id: str
    scene_epoch: int
    calibration_id: str
    images: tuple[ImageObservation, ...]
    measured: RobotSnapshot

    def metadata(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "episode_id": self.episode_id,
            "observation_id": self.observation_id,
            "scene_epoch": self.scene_epoch,
            "calibration_id": self.calibration_id,
            "images": [im.metadata() for im in self.images],
            "measured": asdict(self.measured),
            "holding": "unknown",
            "gripper": "disabled",
        }


@dataclass(frozen=True)
class M1Config:
    calibration_id: str
    mapping_reference: str
    direction_context: str
    step_m: float
    observation_lifetime_s: float
    max_camera_skew_s: float
    directions: tuple[tuple[str, Vector], ...]

    def __post_init__(self) -> None:
        for value in (self.calibration_id, self.mapping_reference, self.direction_context):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "explicit calibration, mapping reference and visual directions required"
                )
        for limit in (self.step_m, self.observation_lifetime_s, self.max_camera_skew_s):
            if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
                raise ValueError("M1 limits must be finite positive numbers")
        if self.step_m > 0.020 or self.observation_lifetime_s > 30:
            raise ValueError("M1 engineering caps: 20mm translation, 30s observation lifetime")
        # The first physical chain test uses the existing measured-start +Z
        # gate. It needs no invented mapping for the other five directions.
        if self.base_z_trial:
            return
        if len(self.directions) != 6 or set(dict(self.directions)) != set(MOVE_TOKENS):
            raise ValueError("explicit mapping of all six unique MV tokens required")
        vectors = dict(self.directions)
        for vector in vectors.values():
            if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
                raise ValueError("direction must be a finite three-vector")
            if not math.isclose(sum(v * v for v in vector), 1.0, abs_tol=1e-9):
                raise ValueError("direction must have unit length; step is configured separately")
        for first, second in (("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"), ("MV_UP", "MV_DOWN")):
            if any(abs(a + b) > 1e-9 for a, b in zip(vectors[first], vectors[second])):
                raise ValueError("opposite directions must be inverse vectors")
        axes = [vectors[t] for t in ("MV_FWD", "MV_LEFT", "MV_UP")]
        for i, axis in enumerate(axes):
            for other in axes[i + 1 :]:
                if abs(sum(a * b for a, b in zip(axis, other))) > 1e-9:
                    raise ValueError("direction axes must be orthogonal")

    @property
    def base_z_trial(self) -> bool:
        return self.directions == (("MV_UP", (0.0, 0.0, 1.0)),)

    @property
    def allowed_tokens(self) -> tuple[str, ...]:
        return ("MV_UP", "DONE") if self.base_z_trial else M1_TOKENS

    @classmethod
    def load(cls, path: Path) -> M1Config:
        obj = strict_json(path.read_text())
        obj["directions"] = tuple(
            (token, tuple(vector)) for token, vector in obj["directions"].items()
        )
        return cls(**obj)

    def validate_observation(self, obs: Observation, now: float) -> float:
        if (
            obs.calibration_id != self.calibration_id
            or not obs.episode_id
            or not obs.observation_id
        ):
            raise ValueError("observation identity/calibration mismatch")
        if type(obs.scene_epoch) is not int or obs.scene_epoch < 0:
            raise ValueError("invalid scene epoch")
        names = tuple(im.camera for im in obs.images)
        if names not in (
            ("front_rgb",),
            ("wrist_rgb",),
            ("front_rgb", "wrist_rgb"),
            ("front_rgb", "right_rgb", "wrist_rgb"),
        ):
            raise ValueError(
                "M1 needs front, wrist, front/wrist or front/right/wrist RGB in declared order"
            )
        times = [im.captured_at for im in obs.images] + [obs.measured.observed_at]
        if not math.isfinite(now) or not all(math.isfinite(t) and 0 <= t <= now for t in times):
            raise ValueError("invalid observation timestamp")
        if max(times) - min(times) > self.max_camera_skew_s:
            raise ValueError("image/state host timestamp skew exceeds reviewed bound")
        expires = min(times) + self.observation_lifetime_s
        if now >= expires:
            raise ValueError("observation expired")
        for im in obs.images:
            if im.sequence < 0 or im.generation < 0 or not im.timestamp_source:
                raise ValueError("invalid image provenance")
            png_bytes(im.rgb())
        return expires
