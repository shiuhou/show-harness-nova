"""M1 deterministic translation into the existing inspect motion validation path."""

from __future__ import annotations

import math

from inspect_robots_dobot.live_profile import LiveMotionProfile
from inspect_robots_dobot.config import SafetyProfile
from inspect_robots_dobot.local_envelope import LocalEnvelopeSafety
from inspect_robots_dobot.motion import (
    CartesianMotionPlan,
    build_micro_move_plan,
    build_motion_plan,
)
from inspect_robots_dobot.types import RobotSnapshot
from inspect_robots_dobot.units import orientation_distance

from core.nova.contracts import M1Config, MOVE_TOKENS


def eef_translation_plan(
    action_id: str,
    delta: tuple[float, float, float],
    observed: RobotSnapshot,
    start: RobotSnapshot,
    config: M1Config,
    profile: LiveMotionProfile,
    now: float,
) -> CartesianMotionPlan:
    """Bind a model delta to the observed TCP and retain the existing gate."""
    if not isinstance(profile.safety, SafetyProfile) or isinstance(
        profile.safety, LocalEnvelopeSafety
    ):
        raise ValueError("EEF mode requires a general reviewed translation profile")
    if not all(math.isfinite(v) for v in delta) or not (
        profile.start_position_tolerance
        < math.hypot(*delta)
        <= min(config.step_m, profile.safety.max_translation_step)
    ):
        raise ValueError("EEF displacement must exceed start tolerance and fit the step budget")
    if (
        start.command_id != observed.command_id
        or math.dist(start.pose.xyz, observed.pose.xyz) > profile.start_position_tolerance
    ):
        raise ValueError("TCP changed since EEF observation")
    # Preserve the observed orientation; reject any intervening orientation change.
    if orientation_distance(start.pose, observed.pose) > profile.start_orientation_tolerance:
        raise ValueError("TCP orientation changed since EEF observation")
    xyz = tuple(a + b for a, b in zip(observed.pose.xyz, delta))
    return build_motion_plan(
        action_id,
        ((*xyz, 0.0, 0.0, 0.0, 0.0),),
        start,
        start.pose,
        profile.safety,
        now,
        profile.keepouts,
    )


def translation_plan(
    action_id: str,
    token: str,
    start: RobotSnapshot,
    config: M1Config,
    profile: LiveMotionProfile,
    now: float,
) -> CartesianMotionPlan:
    if token not in MOVE_TOKENS:
        raise ValueError("M1 supports one translation only; no gripper or rotation")
    if not isinstance(profile.safety, SafetyProfile):
        raise ValueError("M1 requires reviewed general SafetyProfile, not a local +Z envelope")
    if isinstance(profile.safety, LocalEnvelopeSafety):
        envelope = profile.safety.envelope
        if (
            not config.base_z_trial
            or token != "MV_UP"
            or config.step_m != envelope.settings.distance_m
        ):
            raise ValueError("local M1 trial permits only the reviewed base +Z step")
        return build_micro_move_plan(
            envelope.chunk_id, start, "+Z", profile.safety, now, profile.keepouts
        )
    if not profile.start_position_tolerance < config.step_m <= profile.safety.max_translation_step:
        raise ValueError(
            "configured step must exceed start tolerance and fit gate displacement budget"
        )
    vector = dict(config.directions)[token]
    xyz = tuple(a + config.step_m * b for a, b in zip(start.pose.xyz, vector))
    # Agent orientation is RELATIVE to this reference. Zeros preserve measured
    # native rx/ry/rz; gripper value is unused by this translation-only gate.
    waypoint = (xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, 0.0)
    return build_motion_plan(
        action_id, (waypoint,), start, start.pose, profile.safety, now, profile.keepouts
    )
