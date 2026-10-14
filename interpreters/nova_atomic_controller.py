"""M1 deterministic translation into the existing inspect motion validation path."""

from __future__ import annotations

from inspect_robots_dobot.live_profile import LiveMotionProfile
from inspect_robots_dobot.config import SafetyProfile
from inspect_robots_dobot.motion import CartesianMotionPlan, build_motion_plan
from inspect_robots_dobot.types import RobotSnapshot

from core.nova.contracts import M1Config, MOVE_TOKENS


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
