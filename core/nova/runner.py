"""One model decision, one human confirmation, one existing gate execution."""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable
from uuid import uuid4

from inspect_robots_dobot.live_driver import LiveDobotMotionDriver, LiveMotionResult
from inspect_robots_dobot.motion import CartesianMotionPlan

from core.record.episode_logger import M1Evidence
from core.vlm.roles import ControllerAgent
from core.nova.contracts import M1Config, M1_TOKENS, Observation
from interpreters.nova_atomic_controller import translation_plan

CONTROLLER_PROMPT = """Task: {task}
Observation/action binding: {description}
Gripper is disabled; holding is unknown. Orientation stays fixed.
This is a single supervised translation test. Choose one safe small translation
using the reviewed direction mapping in the context, or DONE to request ending.
DONE is not proof of task success. Do not propose grasp, release or rotation.
{output_contract}
"""


@dataclass(frozen=True)
class Proposal:
    action_id: str
    observation: Observation
    token: str
    raw_text: str
    expires_at: float
    plan: CartesianMotionPlan | None


class M1Runner:
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
            allowed_tokens=M1_TOKENS,
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
                observation.images[0].rgb(),
                wrist_image=observation.images[1].rgb() if len(observation.images) == 2 else None,
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
