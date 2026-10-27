# Supervised Nova M1

The M1 path is one Argon decision, one operator confirmation and one
existing inspect motion-gate execution. The controller offers only MV_FWD,
MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN and DONE. Orientation remains fixed;
gripper control, rotation, action chunking and automatic recovery are absent.
DONE requests ending; every result remains task_success=NOT_SCORED.

## Status and sources

- Upstream Show-Harness: `101bd59d3413d32a85d3afa1164f62ddf31f99da`.
- Motion gate: `shiuhou/inspect-robots-dobot` at
  `9c1b77f2d57e60b6aa9f7a1d1442f92ac85c2ba7`; its upstream type dependency is
  pinned in requirements-nova.txt.
- M1 offline integration: PASS. Production provider, strict ControllerAgent,
  interpreter, existing gate, CLI and durable evidence are exercised together
  with injected HTTP/model responses, in-memory transport and synthetic frames.
- API smoke: PASS for text, one synthetic image and two synthetic views using
  the configured gateway's generateContent endpoint and gemini-4-argon alias.
  The gateway returns no model version; underlying weights cannot be verified
  from this response. Interactions returns NotFoundError on that gateway.
- Physical single-translation M1: NOT_RUN. Hardware authorization and reviewed
  rig/calibration profiles are missing. Offline results establish neither
  controller compatibility nor collision safety. Full physical grasp and later acceptance remain incomplete.

The engineering authority is the parent workspace's `.agents/docs/SPEC.md`.
This runbook records the implemented interface, not permission to use devices.

## Research versus physical acceptance

This release includes the reviewed EEF/three-view/workbench software described
below, with 158 passing tests and one skipped upstream test. The Nova/workbench
subset passes 92 socket-blocked tests; strict mypy and Ruff pass. Firmware-exception
handling, shutdown and cleanup have dedicated regressions in the owning repositories.
Historical local operator trials do not validate this revised source; completed
physical grasp and full M1 physical acceptance remain unestablished. Raw local
traces and machine-specific configurations are not included.

## Installation and credentials

Use Python 3.11. Place this repository and the pinned inspect repository in
sibling directories named show-harness and inspect-robots-dobot. In show-harness:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/requirements-nova.txt
cp configs/nova.env.example configs/nova.env
chmod 600 configs/nova.env
```

Fill GEMINI_API_KEY in configs/nova.env with an editor. That file is ignored;
never print, stage or commit it. Existing files must be preserved when setting
up another environment. The parser accepts literal assignments and optional
export prefixes, without evaluating shell expressions. Environment variables
also work; file values take precedence. GEMINI_MODEL must be gemini-4-argon.
Changing it to Flash is rejected rather than silently changing the baseline.

The default API is Interactions. For a gateway verified to support generateContent
instead, explicitly select `--api generate-content`. Endpoint selection is fixed
before the request; failures never cause endpoint/model fallback or retries.
Both paths use official google-genai 2.30.1, a bounded deadline, medium thinking,
one candidate and a strict JSON contract. Interactions uses store=false,
stream=false and background=false. generateContent sends a stateless contents
request and exposes no store=false control; gateway retention is not asserted.

## Offline and API-only entry points

```bash
# Synthetic frames/model, in-memory transport, real inspect validation/gate.
# Each run requires a fresh evidence directory.
.venv/bin/python -m core.nova.cli offline --evidence-dir run-evidence/m1-demo

# Explicit cloud call with recorded images; no robot or camera connection.
.venv/bin/python -m core.nova.cli recorded --api generate-content \
  --front /path/to/front.png --wrist /path/to/wrist.png \
  --evidence-dir run-evidence/api-demo
```

Omit --wrist for a single image. API smoke with synthetic images validates the
transport and response contract, not visual grounding or manipulation ability.
M1 uses ControllerAgent only. Planner/recovery roles and their physical behavior
are outside this milestone; the provider also implements complete_text and
complete_token for the existing client interface.

## Live prerequisites and failure semantics

The live entry point requires an interactive operator, explicit --allow-connection
and --allow-motion flags, and four reviewed local inputs: --semantic-config,
--rig-config, --monitoring and --cameras. These flags do not replace current-session
authorization. Do not run live without it. No live profile is supplied with guessed
hardware values. Store profiles as ignored configs/nova.local*.json.

- Semantic input follows M1Config: calibration_id, mapping_reference,
  direction_context, step_m, observation_lifetime_s, max_camera_skew_s and six
  explicitly reviewed direction vectors. Opposite vectors must be inverse;
  the three axes must be orthogonal. Engineering caps are 20mm and 30s, not
  robot safety guarantees. The gate may impose tighter limits.
- Rig and monitoring use the inspect component's load_config/load_live_profile
  contracts and motion-execution documentation. M1 requires a general
  SafetyProfile, known matching firmware and an attested operator readiness
  check; its fixed +Z local micro-move envelope cannot represent all six actions.
- Cameras are an ordered JSON array of inspect PhysicalCameraConfig objects:
  front_rgb, optionally wrist_rgb. V4L2 access happens only in the live path.
  Calibration and direction_context must describe the actual camera/base frame
  relationship. Fictional offline values are not calibration data.

The operator reviews the exact command and confirms SCENE UNCHANGED and MOVE
ONCE. The runner binds it to episode/observation/scene/calibration, source-image
timestamps and expiry. It rechecks camera generation and measured pose/queue
before arming. M1 conservatively rejects changed poses rather than regenerating
an approved target. Real measurement jitter and timing remain unvalidated.

A dedicated receive-only inspect feedback client runs independently of model
calls and operator prompts. It checks host receive age, controller runtime
progress/regression, RobotMode, queue changes before dispatch, selected frames
and changes from the attested initial raw enable/error/collision/model fields.
It also checks camera health/generation and all four reviewed configuration files.
Raw flag polarity, pose units and absolute controller-to-host clock alignment are
not inferred. Receiving a packet does not prove that the OS buffer is drained;
fresh Dashboard measurements remain required. The first feedback sample must
match the operator-attested idle state. This conservative monitor is an offline
tested candidate, not a validated physical safety system.

Faults latch cancellation before fallible logging. The receiving thread never
writes robot commands; the existing gate worker owns any bounded Stop and
cleanup. The receiver has bounded I/O, closes without reconnect and records a
sample count rather than unbounded per-packet logs.

Malformed, refused, truncated, cancelled, late, expired or duplicate decisions
cannot become movement. Evidence errors before dispatch prevent the motion write.
Lost acknowledgement is terminal execution_unknown: no resend, reconnect,
ClearError, EnableRobot, home or automatic RELEASE. Gate stopping is distinct
from releasing an object. No physical safety claim follows from a fake stop.

After successful settling, the runner requires a new image sequence after the
measured completion barrier, with unchanged episode/calibration/camera generation.
This records an observation; it does not independently establish task success.
Continuous sessions and gripper behavior require M2 design and M1 physical
acceptance; do not loop by reconstructing one-shot drivers.

## Evidence and validation

The existing logger module adds M1Evidence: a fresh, non-overwritable directory,
lossless before/after PNGs and fsynced events.jsonl. Events include resolved
configuration, exact model prompt and image hashes, output, model-version field,
operator confirmation, gate events, command ID, measured state and termination.
Credentials are excluded/redacted. Evidence stays local under ignored run-evidence;
it is not an authorized physical replay file.
Provider events preserve final text before validation, including malformed or
refused/truncated responses and a rejection reason. Output is capped at 64KiB
with explicit evidence truncation; thought parts, tool arguments and SDK
headers are excluded. Timeout/transport errors record only reviewed outcome
and error type. A late response after the request deadline is discarded.

```bash
.venv/bin/pytest --disable-socket -q tests/test_nova_m1.py
.venv/bin/mypy --strict --follow-imports=silent core/nova interpreters/nova_atomic_controller.py
.venv/bin/ruff check .
.venv/bin/ruff format --check core/nova core/vlm/roles.py core/record/episode_logger.py interpreters/nova_atomic_controller.py tests/test_nova_m1.py

# Upstream regressions need additional base dependencies and loopback HTTP fixtures.
.venv/bin/python -m pip install -r requirements/requirements.txt
.venv/bin/pytest -q tests/ --allow-hosts=127.0.0.1,localhost
```

Nova tests explicitly block all sockets. HTTP wire tests use the real SDK with
an injected httpx send method, checking a single request on 408/409/429/500 and
transport failures for both APIs. The SDK's Interactions retry-count conversion
requires an isolated configuration workaround; remove it only after equivalent
wire checks pass with a corrected SDK. CI installs the pinned sibling gate and
runs these checks. Its hosted execution is separate from local verification.


## Reviewed task/workbench research interface

The separate `live --grasp` and `python -m core.nova.workbench` entry points support
three named RGB views and EEF delta or token decisions. They are offline-tested
research interfaces, not completed physical grasp acceptance. Existing M1 retains
its one-decision, one-confirmation behavior. DONE remains NOT_SCORED.

Task execution requires explicit connection/motion flags, current-session readiness,
`--rig-config`, `--semantic-config`, `--monitoring`, `--cameras`, and
`--gripper-settings /path/to/trusted/teleop_config.py`. The gripper settings reuse
X-Trainer's existing Python configuration and therefore must be trusted; they are
not data-only JSON. The sibling pinned X-Trainer runtime and its serial dependencies
must be installed. No device path, polarity, calibration ID, direction convention
or minimum height is inferred by the task interface. A local +Z micro-move profile
cannot be promoted into a general grasp workspace. `--max-step-mm` may only narrow
the reviewed semantic/gate budget.

Firmware compatibility remains required unless the operator explicitly supplies
`--allow-unknown-firmware` for an independently reviewed research session. That
exception never establishes firmware compatibility. Each action retains a one-use
gate capability; errors, cancellation and exhausted limits prevent subsequent sends.
Optional `--max-actions`, `--task-lifetime` and `--observation-timeout` bound the
session; their absence means no corresponding total limit, not a safety guarantee.
Fresh state/queue/gripper checks still precede dispatch. No retry, enabling, clearing,
homing or automatic release is added.

Workbench startup opens no devices. Preview and task start are explicit local UI
actions; stop latches cancellation. Shutdown rejects new work, waits for the task
worker and attempts every camera close even if one close fails. The HTTP server
binds loopback and checks Host/Origin plus a per-process token for write requests.
An in-flight model HTTP call may outlive cancellation in its daemon thread; its
late response cannot resume robot execution. Physical stop timing is unverified.

Validate with socket-blocked `tests/test_nova_m1.py tests/test_nova_workbench.py`,
strict mypy on `core/nova interpreters/nova_atomic_controller.py`, Ruff, and the
upstream tests with loopback fixtures only. Local experimental traces and old
operator trials remain outside this public interface documentation.
