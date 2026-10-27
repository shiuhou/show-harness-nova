"""Receive-only M1 monitoring independent of model calls and operator prompts."""

from __future__ import annotations

import threading
from typing import Callable

from inspect_robots_dobot.feedback import RawFeedback
from inspect_robots_dobot.feedback_client import DobotFeedbackClient
from inspect_robots_dobot.live_driver import LiveDobotMotionDriver, LiveMotionState
from inspect_robots_dobot.types import RobotMode

from core.record.episode_logger import M1Evidence


class _SupervisionFault(ValueError):
    """Only reviewed, credential-free monitoring reasons cross into evidence."""


class M1Supervisor:
    """One receive thread; cancellation uses the existing gate's sole writer.

    Host receive age and controller-clock progress are separate checks, neither
    proves the absence of buffered packets. Raw pose/joint units and flag
    polarities are not inferred. Dashboard measurements remain authoritative.
    """

    def __init__(
        self,
        feedback: DobotFeedbackClient,
        gate: LiveDobotMotionDriver,
        initial_command_id: int,
        evidence: M1Evidence,
        *,
        cancel: Callable[[str], None],
        check_context: Callable[[], None],
    ) -> None:
        self.feedback, self.gate, self.evidence = feedback, gate, evidence
        self.initial_command_id = initial_command_id
        self.cancel, self.check_context = cancel, check_context
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._failure: str | None = None
        self._sample: RawFeedback | None = None
        self._baseline: tuple[int, ...] | None = None
        self._progress_at = 0.0
        self._samples = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("M1 supervisor cannot restart or reconnect")
        self.evidence.record("supervision_started", source="receive_only_feedback")
        self._thread = threading.Thread(target=self._run, name="nova-m1-feedback", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.feedback.config.timeout + 0.5):
            self._fail("feedback startup deadline exceeded")
        self.check()

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self._failure is not None:
                return
            self._failure = reason
        # Cancellation must precede fallible evidence I/O. This never opens a
        # Dashboard connection or writes Stop from the receiving thread.
        self.gate.request_stop()
        self._ready.set()
        self.cancel(reason)

    @staticmethod
    def _stable_fields(sample: RawFeedback) -> tuple[int, ...]:
        # Preserve the attested initial raw flags rather than guessing polarity.
        return (
            sample.enable_status,
            sample.error_status,
            sample.collision_state,
            sample.robot_type,
        )

    def _accept(self, sample: RawFeedback) -> None:
        now = self.gate.clock.monotonic()
        if not sample.is_fresh(now, self.feedback.config.feedback_max_age):
            raise _SupervisionFault("feedback host receive age exceeded")
        if (sample.user_index, sample.tool_index) != (
            self.gate.connection.user_frame,
            self.gate.connection.tool_frame,
        ):
            raise _SupervisionFault("feedback coordinate frame changed or mismatched")
        idle = self.gate.state in (LiveMotionState.IDLE, LiveMotionState.ARMED_FOR_ONE_PLAN)
        allowed_modes = (
            (RobotMode.ENABLED_IDLE,)
            if idle
            else (
                RobotMode.ENABLED_IDLE,
                RobotMode.RUNNING,
            )
        )
        if sample.robot_mode not in allowed_modes:
            raise _SupervisionFault("feedback robot mode invalidated M1")
        expected_idle = self.gate.expected_idle_command_id or self.initial_command_id
        if idle and sample.command_id != expected_idle:
            raise _SupervisionFault("feedback queue changed before dispatch")
        with self._lock:
            if self._baseline is None:
                self._baseline = self._stable_fields(sample)
                self._progress_at = now
            elif self._stable_fields(sample) != self._baseline:
                raise _SupervisionFault("feedback enable/error/collision/model flags changed")
            previous = self._sample
            if previous is not None:
                if (
                    sample.controller_unix_ms < previous.controller_unix_ms
                    or sample.runtime_ms < previous.runtime_ms
                ):
                    raise _SupervisionFault("feedback controller clock regressed")
                if sample.controller_unix_ms > previous.controller_unix_ms:
                    self._progress_at = now
                elif now - self._progress_at >= self.feedback.config.feedback_max_age:
                    raise _SupervisionFault("feedback controller clock stopped progressing")
            self._sample = sample
            self._samples += 1

    def _run(self) -> None:
        try:
            self.feedback.connect()
            next_context_check = 0.0
            while not self._stop.is_set() and not self.gate.connections_closed:
                self._accept(self.feedback.read_sample())
                now = self.gate.clock.monotonic()
                if now >= next_context_check:
                    self.check_context()
                    next_context_check = now + min(0.05, self.feedback.config.feedback_max_age / 2)
                self._ready.set()
        except Exception as exc:
            if not self._stop.is_set() and not self.gate.connections_closed:
                try:
                    reason = str(exc) if isinstance(exc, _SupervisionFault) else type(exc).__name__
                    self._fail("M1 supervision failed: " + reason)
                    self.evidence.record(
                        "supervision_failed",
                        reason=reason,
                        error_type=type(exc).__name__,
                        samples=self._samples,
                    )
                except Exception:
                    # A failed log must not undo the cancellation already latched.
                    self.gate.request_stop()
        finally:
            self._ready.set()
            try:
                self.feedback.close()
            except Exception:
                self._fail("feedback close failed")

    def check(self) -> None:
        with self._lock:
            failure, sample = self._failure, self._sample
        if failure is not None:
            raise RuntimeError(failure)
        if sample is None or not sample.is_fresh(
            self.gate.clock.monotonic(), self.feedback.config.feedback_max_age
        ):
            self._fail("feedback absent or stale")
            raise RuntimeError("feedback absent or stale")
        try:
            self.check_context()
        except Exception:
            self._fail("M1 camera or configuration invalidated")
            raise

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.feedback.config.timeout + 0.5)
            if self._thread.is_alive():
                self._fail("feedback worker did not terminate within its I/O bound")
                raise RuntimeError("feedback worker did not terminate")
            self.evidence.record("supervision_closed", samples=self._samples, failure=self._failure)
