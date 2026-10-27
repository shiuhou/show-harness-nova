"""Offline workbench lifecycle tests; no HTTP, devices or model endpoints."""

from argparse import Namespace
from types import SimpleNamespace
import threading

import pytest

from core.nova import workbench
from core.nova.cli import LiveTaskControl


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(workbench, "load_config", lambda _: None)
    monkeypatch.setattr(workbench.M1Config, "load", lambda _: None)
    monkeypatch.setattr(
        workbench,
        "grasp_motion_settings",
        lambda *_: (
            SimpleNamespace(
                safety=SimpleNamespace(max_translation_step=0.005), start_position_tolerance=0.0005
            ),
            SimpleNamespace(step_m=0.005),
        ),
    )
    return workbench.Workbench(
        Namespace(
            rig_config=tmp_path / "rig",
            monitoring=tmp_path / "monitor",
            semantic_config=tmp_path / "semantic",
        )
    )


def test_idle_construction_opens_no_devices(app):
    assert not app.active()
    assert app.readers == {}
    assert app.snapshot()["status"] == "idle"


def test_shutdown_cancels_worker_before_closing_all_readers(app):
    order = []
    control = LiveTaskControl()
    app.control = control

    def task():
        assert control.cancelled.wait(2)
        order.append("worker stopped")

    class Reader:
        def close(self):
            assert not app.active()
            order.append("reader closed")

    app.readers = {"front_rgb": Reader()}
    app.worker = threading.Thread(target=task)
    app.worker.start()
    app.close()
    assert order == ["worker stopped", "reader closed"]
    assert app.closing and not app.readers
    with pytest.raises(ValueError, match="关闭"):
        app.start({"task": "test", "prompt": "test"})
    with pytest.raises(ValueError, match="关闭"):
        app.preview(True)


def test_cleanup_attempts_every_reader_even_after_close_failure(app):
    closed = []

    class Reader:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("fake close failure")

    app.readers = {"front": Reader("front"), "wrist": Reader("wrist", True)}
    with pytest.raises(RuntimeError, match="fake close"):
        app.preview(False)
    assert set(closed) == {"front", "wrist"}
    assert not app.readers


def test_cancel_before_gate_binding_stays_latched():
    calls = []
    control = LiveTaskControl()
    control.stop()
    control.bind(SimpleNamespace(request_stop=lambda: calls.append("stop")))
    assert calls == ["stop"]
    with pytest.raises(RuntimeError, match="cancelled"):
        control.check()


def test_grasp_requires_general_profile_and_cannot_widen_budget(monkeypatch, tmp_path):
    from dataclasses import replace
    from core.nova import cli
    from core.record.episode_logger import M1Evidence
    from inspect_robots_dobot.config import DobotConfig

    gate, _ = cli.fake_gate(M1Evidence(tmp_path / "evidence"))
    monkeypatch.setattr(
        cli,
        "load_live_profile",
        lambda _, safety: replace(
            gate.profile,
            safety=safety,
            tool_tcp_description="FICTIONAL",
            payload_description="FICTIONAL",
        ),
    )
    config = cli.fake_config()
    with pytest.raises(ValueError, match="general safety"):
        cli.grasp_motion_settings(DobotConfig(), tmp_path / "monitor", config)
    rig = DobotConfig(safety=gate.profile.safety)
    with pytest.raises(ValueError, match="reviewed limits"):
        cli.grasp_motion_settings(rig, tmp_path / "monitor", config, 6.0)
    profile, narrowed = cli.grasp_motion_settings(rig, tmp_path / "monitor", config, 2.0)
    assert profile.safety.max_translation_step == narrowed.step_m == 0.002
    assert profile.safety.minimum_tcp_z == gate.profile.safety.minimum_tcp_z
    assert narrowed.directions == config.directions
    assert narrowed.calibration_id == config.calibration_id
    with pytest.raises(ValueError, match="gripper-settings"):
        cli.make_gripper(None)
