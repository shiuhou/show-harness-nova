# Nova integration

Apply the parent workspace's `.agents/docs/SPEC.md` engineering rules. This
repository owns Show-Harness roles/provider and Nova semantic integration;
`inspect-robots-dobot` owns the physical motion gate and protocol.

No current-session authorization means no robot, camera, Leader or gripper
connection. M1 is translation only, fixed orientation, gripper disabled, one
operator-confirmed action. Never import the upstream real runner for Nova: it
contains startup release and recovery behaviors. No movement fallback, retry,
reconnect, enabling, clearing faults, homing or automatic release in Nova.

Keep API keys in ignored `configs/nova.env`; rig settings and evidence are local.
Offline verification: use `requirements/requirements-nova.txt`, run socket-blocked
pytest, strict mypy on `core/nova`, and Ruff lint/format on changed Python files.
Also run upstream regression tests. Offline results cannot close M1 physical
acceptance or authorize M2. Preserve upstream history and Apache-2.0 license.

Publication destination: `shiuhou/show-harness-nova` origin/main. Publish focused,
verified offline changes before updating the parent submodule pin. Review staged
files, never stage credentials, settings, generated episodes or unrelated work.

## Shared engineering specification

Apply the Dobot workspace [engineering specification](https://github.com/shiuhou/dobot-astra-lab/blob/7106a99/.agents/docs/SPEC.md). In a Dobot workspace, read the root `.agents/docs/SPEC.md`; for a standalone clone use the pinned link. This is the single source for baseline/root-cause review, improvement acceptance, retirement, behavior validation and bounded artifacts. Preserve this repository's hardware, testing and publication rules. Keep tests/configuration in this repository; `.agents/` tracks only `docs/`, with plans/runs kept local. Complex tasks use one plan with a §0 progress/handoff section; simple tasks need none.
