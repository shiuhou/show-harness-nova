"""Loopback-only manual Nova workbench. Startup opens no hardware."""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import re
import secrets
import signal
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from typing import Any
from types import FrameType

from PIL import Image
from inspect_robots_dobot.config import load_config
from core.nova.cli import LiveTaskControl, grasp_live, grasp_motion_settings
from core.nova.runner import EEF_INSTRUCTIONS
from core.nova.contracts import M1Config
from inspect_robots_dobot.camera import LatestFrameReader
from inspect_robots_dobot.camera_v4l2 import V4L2MjpegFrameSource
from inspect_robots_dobot.config import PhysicalCameraConfig
from core.record.episode_logger import M1Evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "run-evidence"
CAMERAS = ("front_rgb", "right_rgb", "wrist_rgb")


class Workbench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.closing = False
        self.readers: dict[str, LatestFrameReader] = {}
        self.worker: threading.Thread | None = None
        self.control: LiveTaskControl | None = None
        profile, config = grasp_motion_settings(
            load_config(args.rig_config), args.monitoring, M1Config.load(args.semantic_config)
        )
        self.default_step_mm = min(config.step_m, profile.safety.max_translation_step) * 1000
        self.minimum_step_mm = profile.start_position_tolerance * 1000
        self.state: dict[str, Any] = {
            "status": "idle",
            "run": None,
            "phase": "idle",
            "phase_at": time.time(),
        }

    def active(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                **self.state,
                "active": self.active(),
                "preview": bool(self.readers),
                "cameras": list(CAMERAS),
                "defaults": {"prompt": EEF_INSTRUCTIONS, "max_step_mm": self.default_step_mm},
                "limits": {
                    "max_step_mm": self.default_step_mm,
                    "min_step_mm_exclusive": self.minimum_step_mm,
                },
                "settings": {
                    "action_mode": "eef",
                    "thinking_level": "low",
                    "api_timeout": None,
                    "max_actions": None,
                    "task_lifetime": None,
                    "observation_timeout": None,
                },
            }

    def preview(self, enabled: bool) -> None:
        with self.lock:
            if enabled and self.closing:
                raise ValueError("工作台正在关闭")
            if self.active():
                raise ValueError("任务运行中不能切换摄像头")
            if not enabled:
                readers, self.readers = self.readers, {}
                with ExitStack() as cleanup:
                    for reader in readers.values():
                        cleanup.callback(reader.close)
                return
            if self.readers:
                return
            specs = [
                PhysicalCameraConfig(**entry) for entry in json.loads(self.args.cameras.read_text())
            ]
            if tuple(s.name for s in specs) != CAMERAS:
                raise ValueError("任务需要 front/right/wrist 三镜头配置")
            readers = {}
            try:
                for spec in specs:
                    reader = LatestFrameReader(
                        V4L2MjpegFrameSource(spec),
                        spec.frame_config,
                        startup_timeout=spec.startup_timeout,
                    )
                    readers[spec.name] = reader
                    reader.start()
                self.readers = readers
            except Exception:
                with ExitStack() as cleanup:
                    for reader in readers.values():
                        cleanup.callback(reader.close)
                raise

    def frame(self, name: str) -> bytes:
        with self.lock:
            reader = self.readers.get(name)
        if reader is None:
            raise ValueError("摄像头未连接")
        frame = reader.latest()
        out = io.BytesIO()
        Image.fromarray(frame.rgb).save(out, format="JPEG", quality=80)
        return out.getvalue()

    def start(self, body: dict[str, Any]) -> None:
        for key in ("task", "prompt"):
            if not isinstance(body.get(key), str) or not 1 <= len(body[key].strip()) <= 16000:
                raise ValueError("任务和 prompt 需为 1–16000 字符")
        with self.lock:
            if self.closing:
                raise ValueError("工作台正在关闭")
            if self.active():
                raise ValueError("已有任务运行中")
            if not self.readers:
                raise ValueError("请先连接摄像头预览")
            max_step_mm = body.get("max_step_mm", self.default_step_mm)
            if max_step_mm is None:
                raise ValueError("请填写最大步长（mm）")
            grasp_motion_settings(
                load_config(self.args.rig_config),
                self.args.monitoring,
                M1Config.load(self.args.semantic_config),
                max_step_mm,
            )
            self.control = LiveTaskControl()
            run = datetime.now().strftime("argon-workbench-%Y%m%d-%H%M%S-") + uuid4().hex[:8]
            args = argparse.Namespace(
                **vars(self.args),
                task=body["task"].strip(),
                prompt=body["prompt"].strip(),
                allow_connection=True,
                allow_motion=True,
                action_mode="eef",
                max_actions=None,
                task_lifetime=None,
                observation_timeout=None,
                timeout=None,
                thinking_level="low",
                image_max_edge=384,
                api="generate-content",
                max_step_mm=max_step_mm,
            )
            self.state = {
                "status": "running",
                "run": run,
                "phase": "starting",
                "phase_at": time.time(),
                "started_at": time.time(),
                "max_step_mm": max_step_mm,
            }
            self.worker = threading.Thread(
                target=self.execute, args=(run, args, self.control), name="nova-task", daemon=False
            )
            self.worker.start()

    def event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind in (
            "task_observation",
            "model_request",
            "model_response",
            "task_decision",
            "task_motion_plan",
            "task_execution",
            "gripper_command",
            "entry_failed",
            "summary",
            "hardware_check",
            "hardware_check_failed",
        ):
            with self.lock:
                self.state.update(phase=kind, phase_at=time.time())
                if "action_index" in event:
                    self.state["step"] = event["action_index"]
                if kind == "hardware_check_failed":
                    self.state["failure"] = event

    def execute(self, run: str, args: argparse.Namespace, control: LiveTaskControl) -> None:
        evidence = None
        status = "failed"
        try:
            evidence = M1Evidence(EVIDENCE / run, on_event=self.event)
            evidence.record(
                "start",
                mode="workbench",
                task=args.task,
                prompt=args.prompt,
                action_mode="eef",
                thinking_level="low",
                max_step_mm=args.max_step_mm,
            )
            result = grasp_live(evidence, args, shared_readers=self.readers, control=control)
            evidence.record("summary", **result)
            status = "finished" if not result.get("failure_cause") else "failed"
        except Exception as exc:
            if evidence is not None:
                evidence.record(
                    "entry_failed",
                    error_type=type(exc).__name__,
                    operator_cancelled=control.cancelled.is_set(),
                )
            status = "stopped" if control.cancelled.is_set() else "failed"
        finally:
            with self.lock:
                self.state.update(status=status, ended_at=time.time())

    def stop(self) -> None:
        with self.lock:
            if self.control is not None:
                self.control.stop()
            if self.active():
                self.state["status"] = "stopping"

    def close(self) -> None:
        # Latch shutdown before releasing the lock so no HTTP request can start
        # another task while the current worker is joining.
        with self.lock:
            self.closing = True
            self.stop()
            worker = self.worker
        if worker is not None:
            worker.join()
        self.preview(False)


def run_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", name):
        raise ValueError("无效记录目录")
    path = (EVIDENCE / name).resolve()
    if path.parent != EVIDENCE.resolve() or not path.is_dir():
        raise ValueError("记录不存在")
    return path


def event_page(name: str, offset: int) -> dict[str, Any]:
    path = run_path(name) / "events.jsonl"
    if path.is_symlink():
        raise ValueError("不支持符号链接")
    events = []
    with path.open("rb") as stream:
        size = path.stat().st_size
        if offset < 0 or offset > size:
            raise ValueError("无效记录位置")
        stream.seek(offset)
        for _ in range(200):
            position = stream.tell()
            line = stream.readline(2_000_000)
            if len(line) == 2_000_000 and not line.endswith(bytes([10])):
                raise ValueError("单条记录超过网页读取上限，请查看本地证据")
            if not line or not line.endswith(bytes([10])):
                stream.seek(position)
                break
            events.append(json.loads(line))
        return {"events": events, "cursor": stream.tell(), "more": stream.tell() < size}


class WorkbenchServer(ThreadingHTTPServer):
    app: Workbench
    token: str
    allowed_hosts: set[str]


class Handler(BaseHTTPRequestHandler):
    server: WorkbenchServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def send(
        self, status: int, data: bytes | dict[str, Any], content_type: str = "application/json"
    ) -> None:
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(data)

    def trusted(self) -> None:
        host = self.headers.get("Host", "")
        if host not in self.server.allowed_hosts:
            raise ValueError("仅允许本机访问")
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            raise ValueError("拒绝跨站请求")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise ValueError("拒绝跨站请求")

    def do_GET(self) -> None:
        try:
            self.trusted()
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                page = (
                    Path(__file__)
                    .with_suffix(".html")
                    .read_text()
                    .replace("__TOKEN__", self.server.token)
                )
                self.send(200, page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self.send(200, self.server.app.snapshot())
            elif url.path == "/api/runs":
                runs = sorted(
                    (
                        p.name
                        for p in EVIDENCE.iterdir()
                        if p.is_dir() and not p.is_symlink() and (p / "events.jsonl").is_file()
                    ),
                    reverse=True,
                )
                self.send(200, {"runs": runs[:300]})
            elif url.path == "/api/events":
                self.send(200, event_page(query["run"][0], int(query.get("cursor", ["0"])[0])))
            elif url.path == "/api/frame":
                self.send(200, self.server.app.frame(query["camera"][0]), "image/jpeg")
            elif url.path == "/api/image":
                folder = run_path(query["run"][0])
                filename = query["file"][0]
                if not re.fullmatch(r"step_[0-9]+_(front|right|wrist)_rgb[.]png", filename):
                    raise ValueError("无效图像路径")
                path = folder / filename
                if path.resolve().parent != folder or path.is_symlink():
                    raise ValueError("无效图像路径")
                edge = int(query.get("edge", ["0"])[0])
                if edge:
                    if not 64 <= edge <= 2048:
                        raise ValueError("无效上传尺寸")
                    with Image.open(path) as image:
                        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                        out = io.BytesIO()
                        image.save(out, format="PNG")
                    self.send(200, out.getvalue(), "image/png")
                else:
                    self.send(200, path.read_bytes(), "image/png")
            else:
                self.send(404, {"error": "不存在"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send(
                400, {"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
            )

    def do_POST(self) -> None:
        try:
            self.trusted()
            if not secrets.compare_digest(self.headers.get("X-Nova-Token", ""), self.server.token):
                self.send(403, {"error": "请刷新页面后操作"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 150000:
                raise ValueError("请求长度无效")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("无效请求")
            if self.path == "/api/start":
                self.server.app.start(body)
            elif self.path == "/api/stop":
                self.server.app.stop()
            elif self.path == "/api/preview":
                if type(body.get("enabled")) is not bool:
                    raise ValueError("缺少预览状态")
                self.server.app.preview(body["enabled"])
            else:
                self.send(404, {"error": "不存在"})
                return
            self.send(200, self.server.app.snapshot())
        except Exception as exc:
            self.send(
                409, {"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    for flag, default in (
        ("session-readiness", "nova.local.session-readiness.json"),
        ("semantic-config", "nova.local.z5.semantic.json"),
        ("rig-config", "nova.local.z5.rig.json"),
        ("monitoring", "nova.local.z5.monitor.json"),
        ("cameras", "nova.local.z5.three.cameras.json"),
        ("env", "nova.env"),
    ):
        parser.add_argument("--" + flag, type=Path, default=ROOT / "configs" / default)
    parser.add_argument("--gripper-settings", type=Path, required=True)
    parser.add_argument("--allow-unknown-firmware", action="store_true")
    args = parser.parse_args()
    EVIDENCE.mkdir(exist_ok=True)
    with (EVIDENCE / ".nova-workbench.lock").open("a") as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def terminate(signum: int, frame: FrameType | None) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        app = Workbench(args)
        server = WorkbenchServer(("127.0.0.1", args.port), Handler)
        server.app = app
        server.token = secrets.token_urlsafe(32)
        server.allowed_hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
        print(
            f"Nova workbench: http://127.0.0.1:{args.port} (idle; hardware disconnected)",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                app.close()
            finally:
                server.server_close()


if __name__ == "__main__":
    main()
