"""One model decision, one human confirmation, one existing gate execution."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable
from uuid import uuid4

from inspect_robots_dobot.live_driver import LiveDobotMotionDriver, LiveMotionResult
from inspect_robots_dobot.local_envelope import LocalEnvelopeSafety
from inspect_robots_dobot.motion import CartesianMotionPlan
from inspect_robots_dobot.units import orientation_distance

from core.record.episode_logger import M1Evidence
from core.vlm.roles import ControllerAgent
from core.nova.contracts import EEF_SCHEMA, M1Config, MOVE_TOKENS, Observation, RGB, eef_delta
from core.vlm.vlm_client import VLMResponse
from interpreters.nova_atomic_controller import eef_translation_plan, translation_plan

CONTROLLER_PROMPT = """Task: {task}
Observation/action binding: {description}
Gripper is disabled; holding is unknown. Orientation stays fixed.
This is a single supervised translation test. Choose one safe small translation
using the reviewed direction mapping in the context, or DONE to request ending.
DONE is not proof of task success. Do not propose grasp, release or rotation.
{output_contract}
"""

TASK_CONTROLLER_PROMPT = """Task: {task}
Observation binding: {description}
The black cube is the target. Use the explicit base-axis mapping in the context.
Each MV_* is exactly one 5 mm Cartesian translation at V5/A5 with fixed orientation.
GRASP closes the verified gripper. RELEASE opens it. Choose DONE only when the task
is complete or the view is unsafe/uncertain. Never invent a trajectory or repeat an action.
Current gripper state: {gripper_state}. Recent actions: {recent_moves}.
{output_contract}
"""

EEF_INSTRUCTIONS = (
    "Pick up the black cube using the camera observations. Adjust the approach from each new view. "
    "The maximum step is a ceiling: use smaller steps as the fingers approach the cube. "
    "Compare the previous wrist image and measured action with the current images before moving again. "
    "If alignment worsened, reassess the base-axis direction instead of repeating the previous delta. "
    "Close the gripper only when the cube is between the fingers, then lift it. "
    "Choose DONE when complete or when a safe action cannot be determined."
)


def _controller_images(
    observation: Observation,
) -> tuple[RGB, RGB | None, str, str, tuple[tuple[str, RGB], ...]]:
    """Select controller inputs by camera name, preserving explicit labels."""
    by_camera = {image.camera: image for image in observation.images}
    if "front_rgb" in by_camera:
        primary = by_camera["front_rgb"]
        wrist = by_camera.get("wrist_rgb")
        return (
            primary.rgb(),
            None if wrist is None else wrist.rgb(),
            "Image A: front RGB",
            "Image B: wrist RGB",
            ()
            if "right_rgb" not in by_camera
            else (("Image C: right RGB", by_camera["right_rgb"].rgb()),),
        )
    wrist = by_camera.get("wrist_rgb")
    if wrist is None:
        raise ValueError("observation has no supported controller image")
    return wrist.rgb(), None, "Image A: wrist RGB", "Image B: wrist RGB", ()


@dataclass(frozen=True)
class Proposal:
    action_id: str
    observation: Observation
    token: str
    raw_text: str
    expires_at: float
    plan: CartesianMotionPlan | None


class _M1ProposalRunner:
    def __init__(
        self,
        gate: LiveDobotMotionDriver,
        config: M1Config,
        client: Any,
        evidence: M1Evidence,
        *,
        monitor_health: Callable[[], None] | None = None,
    ) -> None:
        self.gate, self.config, self.evidence = gate, config, evidence
        self.monitor_health = monitor_health
        self.controller = ControllerAgent(
            client,
            CONTROLLER_PROMPT,
            config.direction_context,
            strict_decisions=True,
            allowed_tokens=config.allowed_tokens,
        )
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._started = False
        self._terminal = False
        self._proposal: Proposal | None = None

    def cancel(self, reason: str = "operator cancellation") -> None:
        # Local gate cancellation remains available while the API is waiting.
        self._cancelled.set()
        self.gate.request_stop()
        self.evidence.record("invalidated", reason=reason)

    def _valid(self, obs: Observation, expires: float) -> None:
        if self.monitor_health is not None:
            self.monitor_health()
        if self._cancelled.is_set() or self.gate.connections_closed:
            raise ValueError("decision invalidated by cancellation/disconnection")
        now = self.gate.clock.monotonic()
        if not math.isfinite(expires) or now >= expires:
            raise ValueError("decision expired")
        self.config.validate_observation(obs, now)

    def propose(self, observation: Observation, task: str) -> Proposal:
        if not self._lock.acquire(blocking=False):
            raise ValueError("one pending action only")
        try:
            if self._started or self._terminal:
                raise ValueError("M1 runner is one-use; no decision retry")
            self._started = True
            expires = self.config.validate_observation(observation, self.gate.clock.monotonic())
            self._valid(observation, expires)
            action_id = uuid4().hex
            for im in observation.images:
                self.evidence.image("before", im.camera, im.png)
            self.evidence.record(
                "observation",
                **observation.metadata(),
                action_id=action_id,
                expires_at=expires,
                config=asdict(self.config),
            )
            image, wrist_image, image_label, wrist_label, additional_images = _controller_images(
                observation
            )
            response = self.controller.decide(
                task,
                {
                    "description": f"action={action_id}; observation={observation.observation_id}; "
                    f"episode={observation.episode_id}; scene_epoch={observation.scene_epoch}; "
                    f"calibration={observation.calibration_id}"
                },
                "none",
                "",
                "DISABLED",
                image,
                wrist_image=wrist_image,
                agentview_label=image_label,
                wrist_label=wrist_label,
                additional_images=additional_images,
            )
            self.evidence.record(
                "decision",
                action_id=action_id,
                source="model",
                raw_text=response.raw_text,
                payload=response.payload,
                prompt=self.controller.last_prompt,
            )
            self._valid(observation, expires)
            plan = None
            if response.token != "DONE":
                # Use a NEW measured start after inference. Old timestamps are never
                # rebased. Exact native command equality is checked after confirmation.
                start = self.gate.measure_start()
                if (
                    start.command_id != observation.measured.command_id
                    or start.pose != observation.measured.pose
                ):
                    raise ValueError("robot moved or queue changed while model was deciding")
                plan = translation_plan(
                    action_id,
                    response.token,
                    start,
                    self.config,
                    self.gate.profile,
                    self.gate.clock.monotonic(),
                )
            proposal = Proposal(
                action_id, observation, response.token, response.raw_text, expires, plan
            )
            self._proposal = proposal
            self.evidence.record(
                "proposal",
                action_id=action_id,
                token=response.token,
                request=plan.request.serialize() if plan else None,
            )
            if response.token == "DONE":
                self._terminal = True
                self.evidence.record("terminal", outcome="end_requested", task_success="NOT_SCORED")
                self.gate.close()
            return proposal
        except BaseException as exc:
            self._terminal = True
            self.gate.close()
            self.evidence.record(
                "terminal",
                outcome="rejected",
                error_type=type(exc).__name__,
                task_success="NOT_SCORED",
            )
            raise
        finally:
            self._lock.release()


class GraspRunner:
    """Argon -> Nova task loop with explicit per-action evidence.

    The model chooses each atomic translation or gripper action. The runner keeps
    one inspected Dashboard session, one Feetech session and optional action limits;
    it never synthesizes a pick trajectory or retries a failed action.
    """

    def __init__(
        self,
        gate: LiveDobotMotionDriver,
        config: M1Config,
        client: Any,
        evidence: M1Evidence,
        gripper: Any,
        *,
        max_actions: int | None = None,
        monitor_health: Callable[[], None],
        action_mode: str = "tokens",
        observation_timeout: float | None = None,
        prompt: str = EEF_INSTRUCTIONS,
    ) -> None:
        if action_mode not in ("tokens", "eef"):
            raise ValueError("unsupported grasp action mode")
        self.action_mode, self.client = action_mode, client
        if observation_timeout is not None and (
            not math.isfinite(observation_timeout) or observation_timeout <= 0
        ):
            raise ValueError("observation timeout must be positive or absent")
        self.observation_timeout, self.prompt = observation_timeout, prompt
        self.gate, self.config, self.evidence, self.gripper = gate, config, evidence, gripper
        self.monitor_health = monitor_health
        self.controller = ControllerAgent(
            client,
            TASK_CONTROLLER_PROMPT,
            config.direction_context,
            strict_decisions=True,
            allowed_tokens=(*MOVE_TOKENS, "GRASP", "RELEASE", "DONE"),
        )
        self.max_actions = max_actions
        self._actions = 0
        self._holding = False
        self._recent: deque[str] = deque(maxlen=6)
        self._previous_observation: Observation | None = None
        self._last_action_feedback: dict[str, Any] | None = None

    def _eef_decision(
        self, observation: Observation, task: str, gripper_state: str
    ) -> tuple[VLMResponse, str]:
        """Compare one completed action with current views; never infer a camera calibration."""
        image, wrist_image, image_label, wrist_label, additional = _controller_images(observation)
        additional = tuple((label + "; CURRENT", rgb) for label, rgb in additional)
        previous = self._previous_observation
        feedback = self._last_action_feedback
        previous_wrist = None
        if (previous is None) != (feedback is None):
            raise ValueError("incomplete visual feedback context")
        if previous is not None:
            if (
                feedback is None
                or feedback["before_observation_id"] != previous.observation_id
                or feedback["after_observation_id"] != observation.observation_id
                or previous.episode_id != observation.episode_id
                or previous.calibration_id != observation.calibration_id
                or previous.scene_epoch != observation.scene_epoch
                or tuple((im.camera, im.generation) for im in previous.images)
                != tuple((im.camera, im.generation) for im in observation.images)
            ):
                raise ValueError("visual feedback context changed")
            previous_wrist = next((im for im in previous.images if im.camera == "wrist_rgb"), None)
        if previous_wrist is not None:
            additional = (
                *additional,
                (
                    "Image D: PREVIOUS wrist RGB, BEFORE the last executed action; historical comparison only",
                    previous_wrist.rgb(),
                ),
            )
        self.evidence.record(
            "task_feedback",
            action_index=self._actions,
            current_observation_id=observation.observation_id,
            previous_action=feedback,
            previous_wrist_image=None
            if previous_wrist is None
            else {
                **previous_wrist.metadata(),
                "file": f"step_{self._actions - 1:03d}_wrist_rgb.png",
            },
        )
        pose = observation.measured.pose
        lower = self.gate.profile.start_position_tolerance * 1000
        upper = min(self.config.step_m, self.gate.profile.safety.max_translation_step) * 1000
        prompt = (
            f"Task: {task}\nOperator instructions: {self.prompt}\n"
            f"Observation={observation.observation_id}; action_index={self._actions}.\n"
            f"Measured tool0 grasp-center TCP XYZ in user0/base mm: {[v * 1000 for v in pose.xyz]}.\n"
            f"Measured native orientation rx/ry/rz degrees: {[math.degrees(v) for v in (pose.rx, pose.ry, pose.rz)]}.\n"
            f"Gripper={gripper_state}. Last completed action feedback: {json.dumps(feedback)}.\n"
            "Images A/B/C, when present, are CURRENT observations from different cameras. "
            "Image D, when present, is the PREVIOUS wrist view before the recorded action. "
            "Compare D with the CURRENT wrist view to judge that action's visual effect. "
            "Wrist-camera translation changes the stationary cube's projection, including scale when moving in depth; "
            "do not treat cube pixel displacement as a base-frame gripper displacement. "
            "Camera projections are uncalibrated. Screen-left/right in different cameras are not common "
            "directions and do not imply a base +Y/-Y sign. Use observed action effects; "
            "do not invent a camera-to-base mapping or make exploratory moves to establish one.\n"
            "Choose exactly one EEF_MOVE, GRASP, RELEASE or DONE. EEF_MOVE supplies delta_mm=[dx,dy,dz] "
            "in user0/base axes relative to this observed TCP, not camera/tool axes or an absolute target. "
            f"Vector length must be >{lower:g} and <={upper:g} mm; this is a maximum, not a target length. "
            "Choose the smallest useful displacement: shorten steps near the cube/fingers or after overshoot. "
            "When the previous move worsened alignment, reassess its sign and reduce the step; "
            "never copy the last delta just because it was executed successfully. "
            "Do not continue descending until lateral alignment and clearance are established. "
            f"Motion uses V{self.gate.profile.safety.speed_percent}/A{self.gate.profile.safety.acceleration_percent}; orientation is fixed. "
            "Other actions require delta_mm=[0,0,0]. GRASP closes; RELEASE opens. "
            "GRASP only when current views show the cube between the fingers at grasp height. "
            "Choose DONE if direction, progress or clearance cannot be determined safely; DONE does not prove success. "
            "Return JSON with decision, delta_mm and brief reasoning identifying the visual change "
            "(or lack of a comparison), the selected direction and why this step size is appropriate."
        )
        response = self.client.complete_json(
            prompt,
            image,
            wrist_image=wrist_image,
            schema=EEF_SCHEMA,
            agentview_label=image_label + "; CURRENT",
            wrist_label=wrist_label + "; CURRENT",
            additional_images=additional,
        )
        return response, prompt

    def _state(self) -> tuple[Any, str]:
        status = self.gripper.read_status()
        return status, (
            "CLOSED"
            if abs(status.position - self.gripper.closed_position)
            < abs(status.position - self.gripper.open_position)
            else "OPEN"
        )

    def step(
        self,
        observation: Observation,
        task: str,
        observe_after: Callable[[float | None], Observation],
    ) -> Observation | None:
        self.monitor_health()
        if self.max_actions is not None and self._actions >= self.max_actions:
            raise RuntimeError("bounded grasp action budget exhausted")
        status, gripper_state = self._state()
        now = self.gate.clock.monotonic()
        self.config.validate_observation(observation, now)
        # Acquisition freshness is checked above; model waiting has no age cutoff
        # by default. Preserve source timestamps and never rebase an old target.
        expires = (
            None
            if self.observation_timeout is None
            else min(
                *(im.captured_at for im in observation.images), observation.measured.observed_at
            )
            + self.observation_timeout
        )
        stage = f"step_{self._actions:03d}"
        for observed_image in observation.images:
            self.evidence.image(stage, observed_image.camera, observed_image.png)
        task_metadata = observation.metadata()
        task_metadata["holding"] = self._holding
        task_metadata["gripper"] = gripper_state
        self.evidence.record(
            "task_observation",
            action_index=self._actions,
            **task_metadata,
            gripper_status=asdict(status),
            expires_at=expires,
        )
        delta = None
        if self.action_mode == "eef":
            response, prompt = self._eef_decision(observation, task, gripper_state)
            delta = eef_delta(
                response.payload["json"],
                min(self.config.step_m, self.gate.profile.safety.max_translation_step),
            )
        else:
            image, wrist_image, image_label, wrist_label, additional_images = _controller_images(
                observation
            )
            response = self.controller.decide(
                task + "\nOperator instructions: " + self.prompt,
                {
                    "description": f"action_index={self._actions}; observation={observation.observation_id}"
                },
                ",".join(self._recent) or "none",
                "",
                gripper_state,
                image,
                wrist_image=wrist_image,
                agentview_label=image_label,
                wrist_label=wrist_label,
                additional_images=additional_images,
            )
            prompt = self.controller.last_prompt
        self.evidence.record(
            "task_decision",
            action_index=self._actions,
            token=response.token,
            raw_text=response.raw_text,
            payload=response.payload,
            prompt=prompt,
        )
        if expires is not None and self.gate.clock.monotonic() >= expires:
            raise RuntimeError("task decision expired")
        self.monitor_health()
        start = self.gate.measure_start()
        if (
            start.command_id != observation.measured.command_id
            or math.dist(start.pose.xyz, observation.measured.pose.xyz)
            > self.gate.profile.start_position_tolerance
            or orientation_distance(start.pose, observation.measured.pose)
            > self.gate.profile.start_orientation_tolerance
        ):
            raise RuntimeError("robot pose or queue changed during model observation")
        current_status, current_gripper = self._state()
        if current_gripper != gripper_state or current_status.moving:
            raise RuntimeError("gripper changed during model observation")
        self.monitor_health()
        token = response.token
        self._actions += 1
        if token == "DONE":
            self.evidence.record("task_terminal", outcome="done", task_success="NOT_SCORED")
            self.gate.close()
            return None
        if token == "GRASP":
            self.gripper.enable_torque()
            self.monitor_health()
            self.evidence.record(
                "gripper_command", token=token, position=self.gripper.closed_position
            )
            self.monitor_health()
            self.gripper.command_position(self.gripper.closed_position)
            self._holding = True
            self._recent.append(token)
            after = observe_after(None)
        elif token == "RELEASE":
            self.gripper.enable_torque()
            self.monitor_health()
            self.evidence.record(
                "gripper_command", token=token, position=self.gripper.open_position
            )
            self.monitor_health()
            self.gripper.command_position(self.gripper.open_position)
            self._holding = False
            self._recent.append(token)
            after = observe_after(None)
        else:
            if delta is not None:
                plan = eef_translation_plan(
                    uuid4().hex,
                    delta,
                    observation.measured,
                    start,
                    self.config,
                    self.gate.profile,
                    self.gate.clock.monotonic(),
                )
            else:
                plan = translation_plan(
                    uuid4().hex,
                    token,
                    start,
                    self.config,
                    self.gate.profile,
                    self.gate.clock.monotonic(),
                )
            self.evidence.record(
                "task_motion_plan",
                action_index=self._actions - 1,
                action_mode=self.action_mode,
                delta_m=delta,
                plan=asdict(plan),
                command=plan.request.serialize(),
            )
            authority = self.gate.arm(plan, confirmation="MOVE ONCE", expires_at=expires)
            result = self.gate.execute(plan, authority)
            self.evidence.record(
                "task_execution", action_index=self._actions - 1, result=asdict(result)
            )
            if result.failure_cause is not None:
                raise RuntimeError(result.failure_cause)
            if delta is None:
                self._recent.append(token)
            after = observe_after(result.final_sample.observed_at if result.final_sample else None)
        self.evidence.record(
            "task_post_observation", action_index=self._actions - 1, **after.metadata()
        )
        if self.action_mode == "eef":
            self._last_action_feedback = {
                "action_index": self._actions - 1,
                "decision": token,
                "requested_delta_mm": None if delta is None else [v * 1000 for v in delta],
                "measured_delta_mm": [
                    round((b - a) * 1000, 6)
                    for a, b in zip(observation.measured.pose.xyz, after.measured.pose.xyz)
                ],
                "before_observation_id": observation.observation_id,
                "after_observation_id": after.observation_id,
            }
            self._previous_observation = observation
        return after


class M1Runner(_M1ProposalRunner):
    def execute(
        self,
        proposal: Proposal,
        *,
        confirmation: str,
        validate_context: Callable[[Observation], None],
        observe_after: Callable[[], Observation],
    ) -> LiveMotionResult:
        if not self._lock.acquire(blocking=False):
            raise ValueError("one pending action only")
        try:
            if self._terminal or proposal is not self._proposal or proposal.plan is None:
                raise ValueError("duplicate, foreign or terminal action")
            # Consume before any potentially failing I/O. No retry of this action.
            self._terminal = True
            if confirmation != "MOVE ONCE":
                raise ValueError("explicit MOVE ONCE confirmation required")
            self._valid(proposal.observation, proposal.expires_at)
            validate_context(proposal.observation)
            if isinstance(self.gate.profile.safety, LocalEnvelopeSafety):
                # Binding is one-use. The gate itself samples again at arm and
                # dispatch and checks the unchanged native target and queue.
                # Never regenerate its envelope after model/human review.
                plan = proposal.plan
            else:
                start = self.gate.measure_start()
                if start.command_id != proposal.observation.measured.command_id:
                    raise ValueError("controller queue changed since observation")
                plan = translation_plan(
                    proposal.action_id,
                    proposal.token,
                    start,
                    self.config,
                    self.gate.profile,
                    self.gate.clock.monotonic(),
                )
                if plan.request.serialize() != proposal.plan.request.serialize():
                    raise ValueError("measured pose changed from operator-reviewed command")
            self.evidence.record(
                "operator_confirmation",
                action_id=proposal.action_id,
                confirmation=confirmation,
                plan=asdict(plan),
            )
            # Context can expire during durable I/O. Gate rechecks its own fresh
            # measurement/authority just before writing; no hidden target regeneration.
            validate_context(proposal.observation)
            self._valid(proposal.observation, proposal.expires_at)
            authority = self.gate.arm(
                plan,
                confirmation=confirmation,
                expires_at=min(
                    proposal.expires_at,
                    self.gate.clock.monotonic() + self.gate.profile.authority_lifetime,
                ),
            )
            result = self.gate.execute(plan, authority)
            self.evidence.record("execution", action_id=proposal.action_id, result=asdict(result))
            if result.failure_cause is None:
                after = observe_after()
                if (after.episode_id, after.scene_epoch, after.calibration_id) != (
                    proposal.observation.episode_id,
                    proposal.observation.scene_epoch,
                    proposal.observation.calibration_id,
                ):
                    raise ValueError("post-observation identity changed")
                if after.observation_id == proposal.observation.observation_id:
                    raise ValueError("post-observation must have a new identity")
                before_images = {im.camera: im for im in proposal.observation.images}
                if set(before_images) != {im.camera for im in after.images}:
                    raise ValueError("post-observation camera set changed")
                for im in after.images:
                    previous = before_images[im.camera]
                    if (
                        im.generation != previous.generation
                        or im.sequence <= previous.sequence
                        or result.final_sample is None
                        or im.captured_at <= result.final_sample.observed_at
                    ):
                        raise ValueError("post-image is not a fresh frame after measured settling")
                    self.evidence.image("after", im.camera, im.png)
                self.config.validate_observation(after, self.gate.clock.monotonic())
                self.evidence.record("post_observation", **after.metadata())
            self.evidence.record(
                "terminal",
                outcome="execution_unknown"
                if result.acceptance_ambiguous
                else "gate_completed"
                if result.failure_cause is None
                else "gate_failed",
                task_success="NOT_SCORED",
            )
            return result
        except BaseException as exc:
            self.evidence.record(
                "terminal",
                outcome="failed",
                error_type=type(exc).__name__,
                task_success="NOT_SCORED",
            )
            raise
        finally:
            self._terminal = True
            self.gate.close()
            self._lock.release()
