"""Run the real provider/role/interpreter/one-shot gate with injected byte transport."""

from dataclasses import replace
import json
import threading
import queue
import struct
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from google.genai import types
from inspect_robots_dobot.feedback_client import DobotFeedbackClient

from core.nova.argon import ARGON_MODEL, ArgonClient, local_settings, sdk_client
from core.nova.cli import (
    FixtureInteractions,
    fake_config,
    fake_gate,
    main,
    synthetic_observation,
)
from core.nova.contracts import M1_TOKENS, MOVE_TOKENS
from core.nova.runner import GraspRunner, M1Runner
from core.nova.supervision import M1Supervisor
from core.record.episode_logger import M1Evidence
from core.vlm.vlm_client import VLMResponse
from interpreters.nova_atomic_controller import translation_plan

pytestmark = pytest.mark.usefixtures("socket_disabled")
VALID = '{"decision":"MV_UP","reasoning":"Test translation."}'


def setup(tmp_path, text=VALID, interaction=None):
    evidence = M1Evidence(tmp_path / "episode")
    gate, peer = fake_gate(evidence)
    client = ArgonClient(
        SimpleNamespace(interactions=interaction or FixtureInteractions(text)),
        timeout_s=1,
        response_sink=lambda response: evidence.record("model_response", **response),
    )
    config = fake_config()
    runner = M1Runner(gate, config, client, evidence)
    obs = synthetic_observation(gate, config)
    return runner, gate, peer, obs, evidence


def prepare(runner, gate, peer, obs):
    proposal = runner.propose(obs, "move safely")
    assert proposal.plan is not None
    peer.expected_motion = proposal.plan.request.serialize().encode()
    peer.target = proposal.plan.final_pose_si
    return proposal


def execute(runner, gate, proposal, **kwargs):
    return runner.execute(
        proposal,
        confirmation="MOVE ONCE",
        validate_context=lambda obs: None,
        observe_after=lambda: synthetic_observation(gate, runner.config, gate.result.final_sample),
        **kwargs,
    )


def moves(peer):
    return [cmd for cmd in peer.sent if cmd.startswith(b"MovL(")]


def test_grasp_task_observation_overrides_generic_holding_metadata(tmp_path):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Status:
        model_number: int = 10760
        servo_id: int = 1
        position: int = 2490
        torque_enabled: bool = False
        moving: bool = False
        load: int = 0
        current: int = 0
        voltage_raw: int = 244
        temperature_raw: int = 40

    class FakeGripper:
        open_position = 2490
        closed_position = 1470

        def read_status(self):
            return Status()

    evidence = M1Evidence(tmp_path / "episode")
    gate, _ = fake_gate(evidence)
    client = ArgonClient(
        SimpleNamespace(interactions=FixtureInteractions('{"decision":"DONE","reasoning":"done"}')),
        timeout_s=1,
    )
    config = fake_config()
    runner = GraspRunner(
        gate,
        config,
        client,
        evidence,
        FakeGripper(),
        max_actions=1,
        monitor_health=lambda: None,
    )

    assert runner.step(synthetic_observation(gate, config), "pick", lambda _after: None) is None
    rows = events(evidence)
    observation = next(row for row in rows if row["kind"] == "task_observation")
    assert observation["holding"] is False
    assert observation["gripper"] == "OPEN"


def events(evidence):
    return [
        json.loads(line) for line in (evidence.run_dir / "events.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("token", MOVE_TOKENS)
def test_six_directions_preserve_orientation_and_run_real_gate_once(tmp_path, token):
    runner, gate, peer, obs, evidence = setup(
        tmp_path, json.dumps({"decision": token, "reasoning": "test"})
    )
    proposal = prepare(runner, gate, peer, obs)
    result = execute(runner, gate, proposal)
    assert result.failure_cause is None
    assert result.command_id == 43 and result.motion_write_attempts == 1
    assert result.transport_is_mock and not result.physical_send_attempted
    assert result.final_sample.command_id == 43
    assert result.final_sample.pose.values[3:] == pytest.approx(
        obs.measured.pose.values[3:], abs=1e-14
    )
    vector = dict(runner.config.directions)[token]
    assert result.final_sample.pose.xyz == pytest.approx(
        tuple(a + 0.005 * b for a, b in zip(obs.measured.pose.xyz, vector))
    )
    assert moves(peer) == [proposal.plan.request.serialize().encode()]
    assert peer.closed and gate.connections_closed
    assert {cmd.split(b"(")[0] for cmd in peer.sent} <= {
        b"RobotMode",
        b"GetErrorID",
        b"GetAngle",
        b"GetPose",
        b"GetCurrentCommandID",
        b"MovL",
    }
    rows = events(evidence)
    assert next(i for i, e in enumerate(rows) if e["kind"] == "operator_confirmation") < next(
        i
        for i, e in enumerate(rows)
        if e["kind"] == "gate_event" and e["event"].get("kind") == "authority_consumed"
    )
    assert (evidence.run_dir / "before_front_rgb.png").read_bytes() == obs.images[0].png
    assert (evidence.run_dir / "after_front_rgb.png").is_file()
    with pytest.raises(ValueError):
        execute(runner, gate, proposal)
    assert len(moves(peer)) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "MV_UP",
        '{"decision":"MV_UP",',
        "```json\n" + VALID + "\n```",
        '{"decision":"MV_UP","reasoning":1}',
        '{"decision":"MV_UP"}',
        '{"decision":"GRASP","reasoning":"close"}',
        '{"decision":"RELEASE","reasoning":"open"}',
        '{"decision":"ROTATE_CW","reasoning":"turn"}',
        '{"decision":"MV_UP","reasoning":"test","extra":true}',
        '{"decision":"MV_DOWN","decision":"MV_UP","reasoning":"duplicate"}',
        '{"decision":" MV_UP ","reasoning":"whitespace"}',
        '{"decision":"MV_UP","reasoning":NaN}',
        "I refuse.",
        "",
    ],
)
def test_invalid_model_outputs_have_no_fallback_and_no_motion(tmp_path, raw):
    runner, gate, peer, obs, _ = setup(tmp_path, raw)
    with pytest.raises(RuntimeError):
        runner.propose(obs, "task")
    assert not moves(peer) and peer.closed
    with pytest.raises(ValueError):
        runner.propose(obs, "task")


def test_done_is_end_request_only(tmp_path):
    runner, gate, peer, obs, evidence = setup(
        tmp_path, '{"decision":"DONE","reasoning":"uncertain"}'
    )
    p = runner.propose(obs, "task")
    assert p.plan is None and not moves(peer) and peer.closed
    terminal = [event for event in events(evidence) if event["kind"] == "terminal"]
    assert len(terminal) == 1
    from datetime import datetime

    timestamp = datetime.fromisoformat(terminal[0]["recorded_at"])
    assert timestamp.utcoffset() is not None
    assert {key: value for key, value in terminal[0].items() if key != "recorded_at"} == {
        "kind": "terminal",
        "outcome": "end_requested",
        "task_success": "NOT_SCORED",
    }


def test_strict_role_independently_rejects_duplicate_raw_fields(tmp_path):
    class MalformedClient:
        def complete_json(self, *args, **kwargs):
            return VLMResponse(
                "MV_UP",
                '{"decision":"MV_DOWN","decision":"MV_UP","reasoning":"x"}',
                {"json": {"decision": "MV_UP", "reasoning": "x"}},
            )

    runner, gate, peer, obs, _ = setup(tmp_path)
    runner.controller.client = MalformedClient()
    with pytest.raises(RuntimeError):
        runner.propose(obs, "task")
    assert not moves(peer) and peer.closed


@pytest.mark.parametrize(
    "status,model",
    [("incomplete", ARGON_MODEL), ("failed", ARGON_MODEL), ("completed", "gemini-2.0-flash")],
)
def test_incomplete_and_model_substitution_fail_closed(tmp_path, status, model):
    class Model:
        def create(self, **kwargs):
            return SimpleNamespace(status=status, model=model, output_text=VALID)

    runner, gate, peer, obs, _ = setup(tmp_path, interaction=Model())
    with pytest.raises(RuntimeError):
        runner.propose(obs, "task")
    assert not moves(peer) and peer.closed


@pytest.mark.parametrize("change", ["expired", "calibration", "scene", "camera_set", "skew"])
def test_observation_contract_rejects_before_model_or_motion(tmp_path, change):
    runner, gate, peer, obs, _ = setup(tmp_path)
    if change == "expired":
        gate.clock.sleep(11)
    if change == "calibration":
        obs = replace(obs, calibration_id="new")
    if change == "scene":
        obs = replace(obs, scene_epoch=-1)
    if change == "camera_set":
        obs = replace(obs, images=())
    if change == "skew":
        obs = replace(obs, images=(replace(obs.images[0], captured_at=0),))
    with pytest.raises(ValueError):
        runner.propose(obs, "task")
    assert not moves(peer)


@pytest.mark.parametrize(
    "change", ["expired", "cancel", "context", "pose", "queue", "foreign", "confirmation"]
)
def test_pending_action_invalidation_and_approval(tmp_path, change):
    runner, gate, peer, obs, _ = setup(tmp_path)
    p = prepare(runner, gate, peer, obs)
    if change == "expired":
        gate.clock.sleep(11)
    if change == "cancel":
        runner.cancel("scene reset")
    if change == "pose":
        peer.measured = replace(peer.measured, x=peer.measured.x + 0.001)
    if change == "queue":
        peer._command_id = 44
    if change == "foreign":
        p = replace(p, action_id="foreign")

    def context(obs):
        if change == "context":
            raise ValueError("camera failed / takeover")

    with pytest.raises(ValueError):
        runner.execute(
            p,
            confirmation="NO" if change == "confirmation" else "MOVE ONCE",
            validate_context=context,
            observe_after=lambda: obs,
        )
    assert not moves(peer) and peer.closed


def test_lost_ack_is_terminal_unknown_no_resend(tmp_path):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    p = prepare(runner, gate, peer, obs)
    original = peer.sendall

    def send(data):
        if data.startswith(b"MovL("):
            peer.sent.append(data)
            peer.measured = peer.target
            peer._command_id = 43
        else:
            original(data)

    peer.sendall = send
    result = execute(runner, gate, p)
    assert result.acceptance_ambiguous and result.motion_write_attempts == 1
    assert result.stop_attempts == 1 and peer.closed
    assert events(evidence)[-1]["outcome"] == "execution_unknown"
    with pytest.raises(ValueError):
        execute(runner, gate, p)
    assert len(moves(peer)) == 1


def test_evidence_failure_before_send_prevents_motion(tmp_path, monkeypatch):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    p = prepare(runner, gate, peer, obs)
    record = evidence.record

    def fail(kind, **data):
        if kind == "gate_event" and data["event"].get("kind") == "authority_consumed":
            raise OSError("disk full")
        record(kind, **data)

    monkeypatch.setattr(evidence, "record", fail)
    result = execute(runner, gate, p)
    assert result.failure_cause and result.motion_write_attempts == 0 and not moves(peer)


def test_api_timeout_and_late_response_cannot_become_action(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class Delayed:
        def create(self, **kwargs):
            started.set()
            release.wait(2)
            return SimpleNamespace(status="completed", model=ARGON_MODEL, output_text=VALID)

    runner, gate, peer, obs, _ = setup(tmp_path, interaction=Delayed())
    runner.controller.client.timeout_s = 0.02
    try:
        with pytest.raises(TimeoutError):
            runner.propose(obs, "task")
        assert started.is_set() and not moves(peer) and peer.closed
    finally:
        release.set()
    with pytest.raises(ValueError):
        runner.propose(obs, "task")


def test_cancel_while_api_waits_discards_valid_late_response(tmp_path):
    started = threading.Event()
    release = threading.Event()
    errors = []

    class Delayed:
        def create(self, **kwargs):
            started.set()
            release.wait(2)
            return SimpleNamespace(status="completed", model=ARGON_MODEL, output_text=VALID)

    runner, gate, peer, obs, _ = setup(tmp_path, interaction=Delayed())

    def propose():
        try:
            runner.propose(obs, "task")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=propose)
    thread.start()
    assert started.wait(1)
    with pytest.raises(ValueError):
        runner.propose(obs, "concurrent")
    runner.cancel()
    release.set()
    thread.join(2)
    assert not thread.is_alive() and errors and not moves(peer) and peer.closed


def test_gate_rejects_keepout_and_tiny_step(tmp_path):
    from inspect_robots_dobot.motion import KeepoutBox

    runner, gate, peer, obs, _ = setup(tmp_path)
    profile = replace(
        gate.profile, keepouts=(KeepoutBox((0.29, -0.01, 0.201), (0.31, 0.01, 0.21)),)
    )
    with pytest.raises(Exception):
        translation_plan("a", "MV_UP", obs.measured, runner.config, profile, gate.clock.monotonic())
    with pytest.raises(ValueError):
        translation_plan(
            "a",
            "MV_UP",
            obs.measured,
            replace(runner.config, step_m=0.0001),
            gate.profile,
            gate.clock.monotonic(),
        )
    assert not moves(peer)


def test_stale_post_frame_does_not_claim_success(tmp_path):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    p = prepare(runner, gate, peer, obs)
    with pytest.raises(ValueError):
        runner.execute(
            p,
            confirmation="MOVE ONCE",
            validate_context=lambda obs: None,
            observe_after=lambda: replace(obs, observation_id="after"),
        )
    assert len(moves(peer)) == 1
    assert events(evidence)[-1]["outcome"] == "failed"


def test_logger_refuses_reuse_and_redacts_secrets(tmp_path):
    evidence = M1Evidence(tmp_path / "episode")
    evidence.record("metadata", api_key="DO_NOT_SAVE", nested={"authorization": "DO_NOT_SAVE"})
    assert "DO_NOT_SAVE" not in (evidence.run_dir / "events.jsonl").read_text()
    with pytest.raises(FileExistsError):
        M1Evidence(evidence.run_dir)


def test_local_credentials_are_literal_private_and_argon_only(tmp_path):
    p = tmp_path / "nova.env"
    p.write_text(
        'export GEMINI_API_KEY="test-only"\nGOOGLE_GEMINI_BASE_URL=https://example.invalid\nGEMINI_MODEL=gemini-4-argon\n'
    )
    p.chmod(0o600)
    assert local_settings(p)["GEMINI_MODEL"] == ARGON_MODEL
    p.write_text("GEMINI_API_KEY=test\nGEMINI_MODEL=gemini-2.0-flash\n")
    with pytest.raises(ValueError):
        local_settings(p)
    p.chmod(0o644)
    with pytest.raises(ValueError):
        local_settings(p)


def test_offline_cli_and_live_permission_fail_before_resources(tmp_path, monkeypatch):
    assert main(["offline", "--evidence-dir", str(tmp_path / "offline")]) == 0
    assert (
        main(
            [
                "live",
                "--evidence-dir",
                str(tmp_path / "live"),
                "--semantic-config",
                "absent",
                "--rig-config",
                "absent",
                "--monitoring",
                "absent",
                "--cameras",
                "absent",
            ]
        )
        == 1
    )


@pytest.mark.parametrize("http_status", [200, 408, 409, 429, 500, "connection_error"])
@pytest.mark.parametrize("api", ["interactions", "generate-content"])
def test_official_sdk_wire_contract_has_no_automatic_retries(
    tmp_path, monkeypatch, http_status, api
):
    requests = []

    def send(self, request, **kwargs):
        requests.append(request)
        if http_status == "connection_error":
            raise httpx.ConnectError("test", request=request)
        if http_status != 200:
            return httpx.Response(http_status, json={"error": {"message": "test"}}, request=request)
        body = (
            {
                "status": "completed",
                "model": ARGON_MODEL,
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": VALID}]}],
            }
            if api == "interactions"
            else {
                "modelVersion": ARGON_MODEL,
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": [{"text": VALID}]},
                    }
                ],
            }
        )
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    settings = {"GEMINI_API_KEY": "test-key", "GOOGLE_GEMINI_BASE_URL": "https://example.invalid"}
    with sdk_client(settings, 1) as sdk:
        client = ArgonClient(sdk, timeout_s=1, api=api)
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": list(M1_TOKENS)},
                "reasoning": {"type": "string"},
            },
            "required": ["decision", "reasoning"],
            "additionalProperties": False,
        }
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        if http_status == 200:
            response = client.complete_json("test", rgb, wrist_image=rgb, schema=schema)
            assert response.token == "MV_UP"
            assert response.payload["model_version"] == (
                ARGON_MODEL if api == "generate-content" else "NOT_REPORTED"
            )
        else:
            with pytest.raises(RuntimeError):
                client.complete_json("test", rgb, schema=schema)
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert str(requests[0].url).startswith("https://example.invalid/")
    if api == "generate-content":
        assert str(requests[0].url).endswith("/v1beta/models/gemini-4-argon:generateContent")
        config = body["generationConfig"]
        # Pinned SDK serializes this field as thinking_level on the wire.
        assert config["thinkingConfig"]["thinking_level"] == "MEDIUM"
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"] == schema
        assert body["systemInstruction"]["parts"][0]["text"]
        assert len(body["contents"]) == 1 and body["contents"][0]["role"] == "user"
        parts = body["contents"][0]["parts"]
        assert len([p for p in parts if "inlineData" in p]) == (2 if http_status == 200 else 1)
        assert not body.get("tools") and not body.get("cachedContent")
        return
    assert body["model"] == ARGON_MODEL and body["store"] is False and body["background"] is False
    assert body["response_format"]["schema"] == schema
    assert "previous_interaction_id" not in body
    assert body["system_instruction"] and body["generation_config"]["thinking_level"] == "medium"
    # SDK 2.30.1 normalizes the caller's content list to one user_input step.
    assert len(body["input"]) == 1 and body["input"][0]["type"] == "user_input"
    content = body["input"][0]["content"]
    assert [v["type"] for v in content] == (
        ["text", "image", "text", "image", "text"]
        if http_status == 200
        else ["text", "image", "text"]
    )
    assert all(v["mime_type"] == "image/png" and v["data"] for v in content if v["type"] == "image")


@pytest.mark.parametrize(
    "change", ["truncated", "blocked", "different_model", "tool", "empty", "no_candidate"]
)
def test_generate_content_bad_response_has_no_motion(tmp_path, change):
    body = {
        "modelVersion": ARGON_MODEL,
        "candidates": [
            {"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": VALID}]}}
        ],
    }
    if change == "truncated":
        body["candidates"][0]["finishReason"] = "MAX_TOKENS"
    if change == "blocked":
        body["promptFeedback"] = {"blockReason": "SAFETY"}
    if change == "different_model":
        body["modelVersion"] = "gemini-2.0-flash"
    if change == "tool":
        body["candidates"][0]["content"]["parts"] = [{"functionCall": {"name": "move", "args": {}}}]
    if change == "empty":
        body["candidates"][0]["content"]["parts"] = []
    if change == "no_candidate":
        body["candidates"] = []

    class Models:
        def generate_content(self, **kwargs):
            return types.GenerateContentResponse.model_validate(body)

    runner, gate, peer, obs, _ = setup(tmp_path)
    runner.controller.client = ArgonClient(
        SimpleNamespace(models=Models()), timeout_s=1, api="generate-content"
    )
    with pytest.raises(RuntimeError):
        runner.propose(obs, "task")
    assert not moves(peer) and peer.closed


def test_generate_content_missing_version_is_explicit_and_uses_real_m1_path(tmp_path):
    class Models:
        def generate_content(self, **kwargs):
            return types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"role": "model", "parts": [{"text": VALID}]},
                        }
                    ]
                }
            )

    runner, gate, peer, obs, evidence = setup(tmp_path)
    runner.controller.client = ArgonClient(
        SimpleNamespace(models=Models()), timeout_s=1, api="generate-content"
    )
    proposal = prepare(runner, gate, peer, obs)
    result = execute(runner, gate, proposal)
    assert result.failure_cause is None and len(moves(peer)) == 1
    decision = next(event for event in events(evidence) if event["kind"] == "decision")
    assert decision["payload"]["model_version"] == "NOT_REPORTED"
    assert decision["payload"]["request"]["api"] == "generate-content"


@pytest.mark.parametrize("raw", ['{"decision":"MV_UP",', "X" * 70000])
def test_rejected_final_output_is_bounded_and_audited_without_motion(tmp_path, raw):
    runner, gate, peer, obs, evidence = setup(tmp_path, raw)
    with pytest.raises(RuntimeError):
        runner.propose(obs, "task")
    audit = [e for e in events(evidence) if e["kind"] == "model_response"]
    assert audit[0]["raw_text"] == raw[:65536]
    assert audit[0]["truncated_for_evidence"] == (len(raw) > 65536)
    assert audit[1]["outcome"] == "rejected" and audit[1]["reason"]
    assert not moves(peer) and gate.connections_closed


def test_model_response_evidence_failure_prevents_proposal_and_motion(tmp_path):
    runner, gate, peer, obs, _ = setup(tmp_path)

    def failed_audit(response):
        raise OSError("disk full")

    runner.controller.client.response_sink = failed_audit
    with pytest.raises(OSError):
        runner.propose(obs, "task")
    assert not moves(peer) and gate.connections_closed


@pytest.mark.parametrize("api", ["interactions", "generate-content"])
def test_provider_text_and_token_contracts_share_fixed_model_and_budget(api):
    replies = iter(["The view is clear.", '{"decision":"DONE"}'])
    requests = []

    class SDK:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(status="completed", model=ARGON_MODEL, output_text=next(replies))

        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {"finishReason": "STOP", "content": {"parts": [{"text": next(replies)}]}}
                    ]
                }
            )

    sdk = SDK()
    client = ArgonClient(SimpleNamespace(interactions=sdk, models=sdk), timeout_s=1, api=api)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert client.complete_text("Describe", rgb).raw_text == "The view is clear."
    assert client.complete_token("Decide", ["DONE"], rgb).token == "DONE"
    assert len(requests) == 2 and all(r["model"] == ARGON_MODEL for r in requests)


def feedback_wire(*, mode=5, command=42, user=0, enable=1, runtime=1000, timestamp=None):
    """Fixed V4.6.5 packet fixture; parsing remains in the production receiver."""
    packet = bytearray(1440)
    struct.pack_into("<H", packet, 0, 1440)
    struct.pack_into("<Q", packet, 24, mode)
    struct.pack_into("<Q", packet, 32, 100000 + runtime if timestamp is None else timestamp)
    struct.pack_into("<Q", packet, 40, runtime)
    struct.pack_into("<Q", packet, 48, 0x0123456789ABCDEF)
    struct.pack_into("<Q", packet, 1112, command)
    packet[1012] = user
    packet[1026] = enable
    return bytes(packet)


class ControlledFeedbackSocket:
    def __init__(self):
        self.packets = queue.Queue()
        self.closed = False
        self.sent = []
        self.timeout = 0.2
        self.heartbeat = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, size):
        try:
            return self.packets.get(
                timeout=min(self.timeout, 0.02) if self.heartbeat else self.timeout
            )
        except queue.Empty:
            if self.heartbeat:
                return self.heartbeat()
            raise TimeoutError("fixture read bound") from None

    def sendall(self, data):
        self.sent.append(data)
        raise AssertionError("feedback is receive-only")

    def close(self):
        self.closed = True


def supervised(runner, gate, evidence, health=lambda: None, *, heartbeat=None):
    peer = ControlledFeedbackSocket()
    peer.heartbeat = heartbeat
    calls = []
    cancelled = threading.Event()

    def socket_factory(host, port, timeout):
        calls.append((host, port))
        return peer

    def cancel(reason):
        try:
            runner.cancel(reason)
        finally:
            cancelled.set()

    feedback = DobotFeedbackClient(
        replace(gate.connection, timeout=0.2), socket_factory=socket_factory, clock=gate.clock
    )
    supervisor = M1Supervisor(feedback, gate, 42, evidence, cancel=cancel, check_context=health)
    runner.monitor_health = supervisor.check
    peer.packets.put(feedback_wire())
    supervisor.start()
    return supervisor, peer, calls, cancelled


@pytest.mark.parametrize(
    "timestamp,runtime,reason",
    [
        (101300, 1000, None),
        (101000, 1001, "clock stopped progressing"),
        (101000, 1000, "clock stopped progressing"),
        (100999, 1001, "clock regressed"),
        (101300, 999, "clock regressed"),
    ],
)
def test_supervision_uses_timestamp_progress_and_rejects_either_counter_regression(
    tmp_path, timestamp, runtime, reason
):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    accepted = threading.Event()
    supervisor, receiver, calls, cancelled = supervised(runner, gate, evidence, accepted.set)
    try:
        for step in (1, 2):
            accepted.clear()
            gate.clock.sleep(0.1)
            receiver.packets.put(feedback_wire(timestamp=101000, runtime=1000))
            assert accepted.wait(0.1)
        accepted.clear()
        gate.clock.sleep(0.1)
        receiver.packets.put(feedback_wire(timestamp=timestamp, runtime=runtime))
        if reason is None:
            assert accepted.wait(0.1), "a progressing TimeStamp must accept a static RunTime"
            supervisor.check()
            assert not cancelled.is_set()
        else:
            assert cancelled.wait(0.1)
            with pytest.raises(RuntimeError, match=reason):
                supervisor.check()
    finally:
        supervisor.close()
    assert receiver.closed and not receiver.sent and len(calls) == 1
    assert not moves(peer)


@pytest.mark.parametrize(
    "fault",
    ["mode", "queue", "frame", "flags", "clock_regression", "clock_frozen", "disconnect", "camera"],
)
def test_receive_only_supervision_invalidates_while_model_is_waiting(tmp_path, fault):
    api_started, api_release = threading.Event(), threading.Event()
    errors = []
    unhealthy = threading.Event()

    class Delayed:
        def create(self, **kwargs):
            api_started.set()
            api_release.wait(2)
            return SimpleNamespace(status="completed", model=ARGON_MODEL, output_text=VALID)

    def health():
        if unhealthy.is_set():
            raise ValueError("camera stale")

    runner, gate, peer, obs, evidence = setup(tmp_path, interaction=Delayed())
    supervisor, receiver, calls, cancelled = supervised(runner, gate, evidence, health)

    def propose():
        try:
            runner.propose(obs, "task")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=propose)
    thread.start()
    assert api_started.wait(1)
    wire = feedback_wire(runtime=1001)
    if fault == "mode":
        wire = feedback_wire(mode=11, runtime=1001)
    elif fault == "queue":
        wire = feedback_wire(command=43, runtime=1001)
    elif fault == "frame":
        wire = feedback_wire(user=1, runtime=1001)
    elif fault == "flags":
        wire = feedback_wire(enable=0, runtime=1001)
    elif fault == "clock_regression":
        wire = feedback_wire(runtime=999)
    elif fault == "clock_frozen":
        gate.clock.sleep(0.3)
        wire = feedback_wire(runtime=1000)
    elif fault == "disconnect":
        wire = b""
    elif fault == "camera":
        gate.clock.sleep(0.06)
        unhealthy.set()
    receiver.packets.put(wire)
    try:
        assert cancelled.wait(1)
    finally:
        api_release.set()
        thread.join(2)
        supervisor.close()
    assert not thread.is_alive() and errors and not moves(peer) and peer.closed
    assert receiver.closed and not receiver.sent and len(calls) == 1
    assert any(e["kind"] == "supervision_failed" for e in events(evidence))
    with pytest.raises(RuntimeError):
        supervisor.start()


def test_stale_feedback_blocks_dispatch_even_before_receive_thread_notices(tmp_path):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    supervisor, receiver, _, _ = supervised(runner, gate, evidence)
    proposal = prepare(runner, gate, peer, obs)
    gate.clock.sleep(0.3)
    try:
        with pytest.raises(RuntimeError, match="stale"):
            execute(runner, gate, proposal)
        assert not moves(peer)
    finally:
        supervisor.close()
    assert receiver.closed


def test_supervisor_fault_after_send_requests_stop_through_existing_writer(tmp_path, monkeypatch):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    runtime = 1000

    def heartbeat():
        nonlocal runtime
        runtime += 1
        return feedback_wire(runtime=runtime)

    supervisor, receiver, _, cancelled = supervised(runner, gate, evidence, heartbeat=heartbeat)
    proposal = prepare(runner, gate, peer, obs)
    recv = peer.recv
    send = peer.sendall
    write_threads = []
    injected = False

    def audited_send(data):
        write_threads.append(threading.get_ident())
        send(data)

    def fault_on_ack(size):
        nonlocal injected
        if moves(peer) and not injected:
            injected = True
            receiver.packets.put(feedback_wire(mode=11, runtime=runtime + 1))
            assert cancelled.wait(1)
        return recv(size)

    monkeypatch.setattr(peer, "sendall", audited_send)
    monkeypatch.setattr(peer, "recv", fault_on_ack)
    try:
        result = execute(runner, gate, proposal)
        assert result.failure_cause and result.motion_write_attempts == 1
        assert result.stop_attempts == 1 and len(moves(peer)) == 1
        assert set(write_threads) == {threading.get_ident()}
    finally:
        supervisor.close()
    assert not receiver.sent


def test_healthy_supervisor_allows_exactly_one_real_gate_action(tmp_path):
    runner, gate, peer, obs, evidence = setup(tmp_path)
    runtime = 1000

    def heartbeat():
        nonlocal runtime
        runtime += 1
        return feedback_wire(runtime=runtime, command=peer._command_id)

    supervisor, receiver, calls, cancelled = supervised(runner, gate, evidence, heartbeat=heartbeat)
    try:
        result = execute(runner, gate, prepare(runner, gate, peer, obs))
        assert result.failure_cause is None and result.motion_write_attempts == 1
        assert not cancelled.is_set() and len(moves(peer)) == 1
    finally:
        supervisor.close()
    assert receiver.closed and not receiver.sent and len(calls) == 1
