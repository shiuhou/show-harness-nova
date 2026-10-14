"""Explicit offline, recorded API-only, and operator-authorized M1 entry points."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from inspect_robots_dobot.camera import LatestFrameReader
from inspect_robots_dobot.camera_v4l2 import V4L2MjpegFrameSource
from inspect_robots_dobot.clock import FakeClock
from inspect_robots_dobot.config import (
    ConnectionConfig,
    PhysicalCameraConfig,
    SafetyProfile,
    load_config,
)
from inspect_robots_dobot.live_driver import LiveDobotMotionDriver
from inspect_robots_dobot.feedback_client import DobotFeedbackClient
from inspect_robots_dobot.live_mock import MockMotionSocket
from inspect_robots_dobot.live_profile import (
    LiveMotionProfile,
    OperatorReadiness,
    load_live_profile,
)
from inspect_robots_dobot.live_trial import collect_readiness
from inspect_robots_dobot.types import PoseSI

from core.nova.argon import ARGON_MODEL, ArgonClient, local_settings, sdk_client
from core.nova.contracts import ImageObservation, M1Config, Observation, RGB, strict_json
from core.nova.runner import M1Runner
from core.nova.supervision import M1Supervisor
from core.record.episode_logger import M1Evidence
from core.vlm.roles import ControllerAgent
from core.nova.runner import CONTROLLER_PROMPT
from core.nova.contracts import M1_TOKENS


def fake_config() -> M1Config:
    return M1Config(
        "FICTIONAL_CALIBRATION",
        "OFFLINE_ONLY",
        "FICTIONAL: forward +X, left +Y, up +Z.",
        0.005,
        10.0,
        0.1,
        tuple(
            zip(
                M1_TOKENS[:6],
                (
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                ),
            )
        ),
    )


def fake_gate(evidence: M1Evidence) -> tuple[LiveDobotMotionDriver, MockMotionSocket]:
    safety = SafetyProfile(
        (0.1, -0.2, 0.08),
        (0.5, 0.2, 0.5),
        0.08,
        0,
        0,
        0.02,
        0.1,
        5,
        5,
        0.0001,
        0.001,
        0.5,
        0.25,
        0.01,
    )
    profile = LiveMotionProfile(
        safety,
        0.0005,
        0.005,
        0.2,
        3,
        0.01,
        0.04,
        0.1,
        0.0002,
        0.001,
        1.0,
        "FICTIONAL_MODEL",
        "FICTIONAL_FIRMWARE",
        "OFFLINE_ONLY",
        "OFFLINE_ONLY",
    )
    connection = ConnectionConfig(
        host="192.0.2.1",
        controller_firmware="FICTIONAL_FIRMWARE",
        protocol_compatibility_confirmed=True,
        tcp_control_owned=True,
        user_frame=0,
        tool_frame=0,
    )
    peer = MockMotionSocket(PoseSI(0.3, 0.0, 0.2, 0.1, -0.2, 0.3))
    gate = LiveDobotMotionDriver(
        connection,
        profile,
        allow_motion=True,
        socket_factory=lambda host, port, timeout: peer,
        clock=FakeClock(1.0),
        transport_is_mock=True,
        event_sink=lambda event: evidence.record("gate_event", event=event),
    )
    gate.connect(allow_connection=True)
    gate.confirm_readiness(OperatorReadiness("OFFLINE", True, True, True, True, True, True, True))
    return gate, peer


def synthetic_observation(
    gate: LiveDobotMotionDriver, config: M1Config, measured: Any = None
) -> Observation:
    from core.nova.contracts import png_bytes

    sample = measured if measured is not None else gate.measure_start()
    # Fake clock advance gives the AFTER frame a strict post-settle host barrier.
    gate.clock.sleep(0.001)
    image = ImageObservation(
        "front_rgb",
        png_bytes(np.zeros((32, 32, 3), dtype=np.uint8)),
        gate.clock.monotonic(),
        int(gate.clock.monotonic() * 1000000),
        0,
        "synthetic",
    )
    return Observation("OFFLINE_EPISODE", uuid4().hex, 0, config.calibration_id, (image,), sample)


class FixtureInteractions:
    """Offline response fixture; cannot open sockets or control a robot."""

    def __init__(self, text: str) -> None:
        self.text = text

    def create(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            status="completed", model=ARGON_MODEL, output_text=self.text, usage=None
        )


def offline(evidence: M1Evidence, response: str) -> dict[str, Any]:
    config = fake_config()
    gate, peer = fake_gate(evidence)
    client = ArgonClient(SimpleNamespace(interactions=FixtureInteractions(response)), timeout_s=1)
    runner = M1Runner(gate, config, client, evidence)
    try:
        proposal = runner.propose(
            synthetic_observation(gate, config), "Test one supervised small translation"
        )
        if proposal.plan is None:
            return {
                "outcome": "end_requested",
                "physical_send_attempted": False,
                "task_success": "NOT_SCORED",
            }
        peer.expected_motion = proposal.plan.request.serialize().encode()
        peer.target = proposal.plan.final_pose_si
        result = runner.execute(
            proposal,
            confirmation="MOVE ONCE",
            validate_context=lambda obs: None,
            observe_after=lambda: synthetic_observation(
                gate, config, gate.result.final_sample if gate.result else None
            ),
        )
        return asdict(result)
    finally:
        gate.close()


def recorded(evidence: M1Evidence, args: argparse.Namespace) -> dict[str, Any]:
    from core.nova.contracts import png_bytes

    images: list[RGB] = []
    for slot, path in (("front_rgb", args.front), ("wrist_rgb", args.wrist)):
        if path is not None:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            evidence.image("before", slot, png_bytes(rgb))
            images.append(rgb)
    settings = local_settings(args.env)
    with sdk_client(settings, args.timeout) as sdk:
        client = ArgonClient(
            sdk,
            timeout_s=args.timeout,
            api=args.api,
            request_sink=lambda request: evidence.record("model_request", **request),
            response_sink=lambda response: evidence.record("model_response", **response),
        )
        controller = ControllerAgent(
            client,
            CONTROLLER_PROMPT,
            "API-only recorded image validation. No physical commands. Choose DONE if uncertain.",
            strict_decisions=True,
            allowed_tokens=M1_TOKENS,
        )
        response = controller.decide(
            args.task,
            {"description": "recorded API-only; no motion authority"},
            "none",
            "",
            "DISABLED",
            images[0],
            wrist_image=images[1] if len(images) == 2 else None,
        )
        evidence.record(
            "decision",
            raw_text=response.raw_text,
            payload=response.payload,
            prompt=controller.last_prompt,
            source="model",
        )
        return {
            "outcome": "recorded_api_response",
            "token": response.token,
            "model": ARGON_MODEL,
            "physical_send_attempted": False,
            "task_success": "NOT_SCORED",
        }


def live(evidence: M1Evidence, args: argparse.Namespace) -> dict[str, Any]:
    if (
        not args.allow_connection
        or not args.allow_motion
        or not sys.stdin.isatty()
        or not sys.stdout.isatty()
    ):
        raise ValueError(
            "live requires explicit connection/motion flags and an interactive operator"
        )
    config = M1Config.load(args.semantic_config)
    rig = load_config(args.rig_config)
    if rig.connection is None or rig.safety is None or rig.local_micro_move is not None:
        raise ValueError("M1 requires a reviewed connection and general workspace profile")
    safety = rig.safety
    profile = load_live_profile(args.monitoring, safety)
    profile.require_live_descriptions()
    if not profile.firmware or profile.firmware != rig.connection.controller_firmware:
        raise ValueError("reviewed known firmware must match connection and monitoring profile")
    # Verify all semantic/gate configuration before opening any resource.
    if not profile.start_position_tolerance < config.step_m <= rig.safety.max_translation_step:
        raise ValueError("semantic step incompatible with gate budget/tolerance")
    camera_obj = strict_json(args.cameras.read_text())
    specs = tuple(PhysicalCameraConfig(**entry) for entry in camera_obj)
    if tuple(spec.name for spec in specs) not in (("front_rgb",), ("front_rgb", "wrist_rgb")):
        raise ValueError("explicit front, optionally wrist, camera configuration required")
    settings = local_settings(args.env)
    print("Live M1: one fixed-orientation translation; gripper disabled. No enable/home/recovery.")
    print(json.dumps({"semantic": asdict(config), "cameras": camera_obj}, indent=2))
    readiness, attestations = collect_readiness(rig, profile)
    evidence.record(
        "readiness",
        readiness=asdict(readiness),
        attestations=attestations,
        rig=asdict(rig),
        monitoring=asdict(profile),
    )
    readers: dict[str, LatestFrameReader] = {}
    gate = LiveDobotMotionDriver(
        rig.connection,
        profile,
        allow_motion=True,
        event_sink=lambda event: evidence.record("gate_event", event=event),
    )
    runner: M1Runner | None = None
    supervisor: M1Supervisor | None = None
    episode = uuid4().hex
    initial_generations: dict[str, int] = {}
    try:
        with sdk_client(settings, args.timeout) as sdk:
            client = ArgonClient(
                sdk,
                timeout_s=args.timeout,
                api=args.api,
                request_sink=lambda request: evidence.record("model_request", **request),
                response_sink=lambda response: evidence.record("model_response", **response),
            )
            runner = M1Runner(gate, config, client, evidence)
            gate.connect(allow_connection=True)
            gate.confirm_readiness(readiness)
            for spec in specs:
                reader = LatestFrameReader(
                    V4L2MjpegFrameSource(spec),
                    spec.frame_config,
                    startup_timeout=spec.startup_timeout,
                )
                readers[spec.name] = reader
                reader.start()
                initial_generations[spec.name] = reader.latest().generation

            def check_background_context() -> None:
                if (
                    M1Config.load(args.semantic_config) != config
                    or load_config(args.rig_config) != rig
                    or load_live_profile(args.monitoring, safety) != profile
                    or strict_json(args.cameras.read_text()) != camera_obj
                ):
                    raise ValueError("reviewed configuration changed; discard pending decision")
                for name, reader in readers.items():
                    if reader.latest().generation != initial_generations[name]:
                        raise ValueError("camera fault or generation changed")

            # Dedicated receive-only connection. Short bounded reads also bound
            # camera/configuration checks while inference or operator input waits.
            supervisor = M1Supervisor(
                DobotFeedbackClient(
                    replace(
                        rig.connection,
                        timeout=min(rig.connection.timeout, rig.connection.feedback_max_age),
                    ),
                    clock=gate.clock,
                ),
                gate,
                gate.measure_start().command_id,
                evidence,
                cancel=runner.cancel,
                check_context=check_background_context,
            )
            runner.monitor_health = supervisor.check
            supervisor.start()

            def capture(after: float | None = None) -> Observation:
                sample = (
                    gate.result.final_sample
                    if after is not None and gate.result
                    else gate.measure_start()
                )
                if sample is None:
                    raise ValueError("no measured state for observation")
                images = tuple(
                    ImageObservation.from_frame(name, reader.latest(after=after))
                    for name, reader in readers.items()
                )
                # After images need a current state timestamp only for skew validation;
                # retain the real measured settle sample without timestamp rebasing.
                return Observation(episode, uuid4().hex, 0, config.calibration_id, images, sample)

            def validate_context(obs: Observation) -> None:
                assert supervisor is not None
                supervisor.check()
                for im in obs.images:
                    fresh = readers[im.camera].latest()
                    if fresh.generation != initial_generations[im.camera]:
                        raise ValueError("camera generation changed; discard pending decision")
                if (
                    input(
                        "Scene unchanged, sole writer, no takeover or camera fault? Type SCENE UNCHANGED: "
                    )
                    != "SCENE UNCHANGED"
                ):
                    raise ValueError("scene/takeover invalidated decision")

            proposal = runner.propose(capture(), args.task)
            if proposal.plan is None:
                return {"outcome": "end_requested", "task_success": "NOT_SCORED"}
            print(
                json.dumps(
                    {
                        "action_id": proposal.action_id,
                        "token": proposal.token,
                        "raw_response": proposal.raw_text,
                        "expires_at": proposal.expires_at,
                        "command": proposal.plan.request.serialize(),
                    },
                    indent=2,
                )
            )
            # Human prompts happen before the fresh post-confirmation measurement.
            validate_context(proposal.observation)
            confirmation = input("Type MOVE ONCE to execute exactly this command: ")

            def check_without_prompt(obs: Observation) -> None:
                assert supervisor is not None
                supervisor.check()
                for im in obs.images:
                    if readers[im.camera].latest().generation != im.generation:
                        raise ValueError("camera generation changed")

            result = runner.execute(
                proposal,
                confirmation=confirmation,
                validate_context=check_without_prompt,
                observe_after=lambda: capture(
                    gate.result.final_sample.observed_at
                    if gate.result and gate.result.final_sample
                    else time.monotonic()
                ),
            )
            return asdict(result)
    except KeyboardInterrupt:
        if runner is not None:
            runner.cancel()
        raise
    finally:
        gate.close()
        try:
            if supervisor is not None:
                supervisor.close()
        finally:
            for reader in readers.values():
                reader.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    off = sub.add_parser("offline")
    off.add_argument(
        "--response", default='{"decision":"MV_UP","reasoning":"Synthetic offline proposal."}'
    )
    rec = sub.add_parser("recorded")
    rec.add_argument("--front", type=Path, required=True)
    rec.add_argument("--wrist", type=Path)
    physical = sub.add_parser("live")
    for name in ("semantic-config", "rig-config", "monitoring", "cameras"):
        physical.add_argument("--" + name, type=Path, required=True)
    physical.add_argument("--allow-connection", action="store_true")
    physical.add_argument("--allow-motion", action="store_true")
    for p in (off, rec, physical):
        p.add_argument("--evidence-dir", type=Path, required=True)
    for p in (rec, physical):
        p.add_argument("--env", type=Path, default=Path("configs/nova.env"))
        p.add_argument("--timeout", type=float, default=12.0)
        p.add_argument(
            "--api", choices=("interactions", "generate-content"), default="interactions"
        )
        p.add_argument(
            "--task", default="Choose one safe supervised small translation or request DONE."
        )
    args = parser.parse_args(argv)
    evidence: M1Evidence | None = None
    try:
        evidence = M1Evidence(args.evidence_dir)
        evidence.record(
            "start",
            mode=args.mode,
            model=ARGON_MODEL,
            milestone="M1",
            inspect_revision="cc84d90c9d365cd700fbf610d6ca67c64b1ea3f8",
            upstream_revision="101bd59d3413d32a85d3afa1164f62ddf31f99da",
        )
        result = (
            offline(evidence, args.response)
            if args.mode == "offline"
            else (recorded(evidence, args) if args.mode == "recorded" else live(evidence, args))
        )
        evidence.record("summary", **result)
        print(json.dumps(result, allow_nan=False, default=str))
        return 1 if result.get("failure_cause") else 0
    except (Exception, KeyboardInterrupt) as exc:
        # Never echo SDK errors/headers/credential file contents.
        if evidence is not None:
            evidence.record("entry_failed", error_type=type(exc).__name__)
        print(
            f"M1 failed ({type(exc).__name__}); inspect local evidence. No retry.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
