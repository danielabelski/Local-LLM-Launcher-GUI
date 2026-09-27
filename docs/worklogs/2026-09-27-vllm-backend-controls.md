# vLLM backend controls — execution record

## Scope and delivery

Implemented the backend-controls plan on `feat/vllm-blackwell-controls`, based on the previous issue fixes at `27fb558`. User requested ce-plan, ce-work, and ponytail, followed by a local commit without pushing. A remote PR requires a pushed branch, so the PR body is prepared locally in `docs/pr/vllm-backend-controls.md`.

## Completed checklist

- [x] U1: selected-runtime native CLI/choice/package evidence with bounded subprocesses and a 60-second cache; Docker explicitly unverified.
- [x] U2: three catalog selectors and shared conservative checks for runtime, optional dependency, effective raw overrides, model precision/cache, parallelism, and GPU architecture.
- [x] U3: advice and direct launch integration; command-generation/API/browser regression coverage; documentation and local PR body.

The plan received independent repository/learning/flow research and a focused plan review with no blocking gaps. Tagged source verification corrected the stale `B12X_ATTN` release-prose spelling to `B12X` before shipping. The existing catalog-driven frontend handles the controls without new components, dependencies, or generated asset changes. Automatic selection emits no additional flags and performs no capability probes.

Ponytail scope choices: reuse generic selectors and advice messages; do not create container-probing lifecycle machinery, a package installer, an autotuner, or a generic engine schema. Native support follows actual help/choices rather than assuming a version-number cutoff, allowing backported features.

## Verification observed

- `PYTHONPATH=src ../../.venv/bin/python -m pytest -q`: **283 passed**, one existing Starlette/httpx deprecation warning.
- `npm run build` in `frontend/`: passed; generated bundle unchanged because controls come from the API catalog.
- `node tests/browser-vllm-backends.mjs`: passed; selector payloads, automatic reset, invalid-config blocking, Docker warning, and 390px mobile overflow check.
- `node tests/browser-smoke.mjs`: passed.
- `node tests/browser-placement.mjs`: passed.
- `node tests/browser-updater.mjs`: passed.
- `git diff --check`: passed.
- `npm run lint`: unavailable because the existing repository has no ESLint configuration file; no lint configuration changes included.

Tests use temporary native Python environments and actual subprocesses to establish target-runtime separation, package visibility, cache refresh, and script-directory behavior. Engine-launch API tests stub only process startup; they verify rejection before startup and unchanged successful configuration delivery. Browser tests use synthetic hardware/model/runtime evidence and do not launch models or modify engines.

The initial system-Python test command could not import FastAPI; rerunning with the project's virtual environment resolved the environment mismatch. No test assertion was relaxed to obtain a passing run.

## Review and fixes

An independent focused read-only review examined capability detection, compatibility rules, API/builders, and tests. It found two defects, both fixed:

1. Isolated Python package probing could falsely report b12x missing when the real runtime inherits PYTHONPATH or user-site packages. The probe now mirrors console-script import context and includes relevant environment variables in its cache identity.
2. Mixed GPU groups only warned even when tensor parallelism used every selected device. Known incompatible full selections now reject, including inherited native masks and trailing raw parallelism overrides; uncertain subsets still warn.

The independent reviewer rechecked both fixes, ran 83 focused tests, and reported no additional concrete regressions. Review coverage was focused, not a full cross-model review workflow. Final parent suite included 283 tests.

## Practical limits

- VM100 has no available GPU. No RTX 5060 Ti inference or performance benchmark was performed.
- CLI recognition and package presence do not prove compiled-kernel or full model compatibility. Warnings preserve this distinction.
- Docker image/package support is intentionally unverified; host Python evidence is never substituted.
- No vLLM upgrade, b12x installation, model download, host/VM change, push, or remote PR creation occurred.
