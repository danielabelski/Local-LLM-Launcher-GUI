---
title: "feat: Add vLLM Blackwell backend controls"
date: 2026-09-27
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
depth: standard
---

## Goal Capsule

Expose useful vLLM computation choices for RTX 5060 Ti and other supported GPUs, preserving automatic defaults and reporting known incompatibilities before launch. Build on the issue fixes at 27fb558. Commit locally; prepare a PR description without pushing or opening a remote PR, because the user's latest delivery constraint forbids pushing.

## Product Contract

### Requirements

- R1. Advanced controls expose linear, MoE (mixture-of-experts), and attention backends, including flashinfer_cutlass, b12x, and B12X where relevant. Automatic omits each override; no preset forces a backend.
- R2. Native support checks use the selected vLLM executable and its Python environment. Missing flags/package produce actionable feedback; failed probes remain unknown. No engine/package installation occurs.
- R3. Advice and direct launch reject definite B12X conflicts in model computation precision, KV cache, context/expert parallelism, and selected GPU architecture. Unknown model/runtime compatibility is explained rather than asserted.
- R4. Native and Docker command builders emit the same selected vLLM flags. Docker capability state remains independent of native evidence.
- R5. Tests prove command generation, error paths, target-runtime isolation, raw overrides, and visible control behavior. No GPU performance claim is made on VM100.

## Planning Contract

- KTD1. Reuse the catalog-driven controls and existing advice messages. Three null-default choices need no new UI component or dependency (R1).
- KTD2. A focused vLLM support module owns bounded, cached native help/package probes. Resolve interpreter from the target console script conservatively; unknown wrappers remain unknown. Cache by runtime identity plus a short lifetime so in-place dependency changes are eventually seen (R2).
- KTD3. Docker checks remain explicitly unverified rather than starting inspection containers during recurring advice. This avoids automatic pulls, container lifecycle machinery, and applying host package facts to images. Static compatibility rules still run; users receive container-specific installation/check guidance (R4).
- KTD4. Shared validation examines effective targeted options, honoring trailing extra-argument overrides. Unsupported/unrecognized raw configuration must not receive a false all-clear. Reuse existing HTTP 400 and per-flag warning behavior (R3).
- KTD5. NVFP4 cache controls, automatic tuning, source updater changes, new engines, host/VM configuration, and performance presets remain outside this focused change. Automatic W4A4 kernel prioritization comes from vLLM itself.

## Implementation Units

### U1. Target-runtime support evidence

**Requirements:** R2, R4. **Dependencies:** none.

**Files:** `src/local_llm_launcher/engines/vllm_capabilities.py`, `tests/test_vllm_capabilities.py`.

**Approach:** Probe native serve help and optional package metadata using the selected runtime, timeouts, and short-lived caching. Return supported/unsupported/unknown evidence without importing vLLM into the launcher. Docker returns explicit unknown evidence. Follow the bounded subprocess pattern in `hardware.py`.

**Test scenarios:**

- Configured executable in a separate environment supplies its own version/package state.
- Successful old help lacks a requested flag; failed help does not imply unsupported.
- Missing/present b12x, unknown interpreter, timeout, malformed output, and missing executable remain distinguishable.
- Cached calls avoid repeated processes; identity changes and expiry refresh evidence.
- Docker does not execute or inherit a native probe.

**Verification:** Focused tests pass without requiring GPU, Docker, or installed vLLM.

### U2. Backend choices and shared compatibility rules

**Requirements:** R1, R3, R4. **Dependencies:** U1 for runtime evidence contract.

**Files:** `src/local_llm_launcher/data/flags_vllm.json`, `src/local_llm_launcher/engines/vllm_backends.py`, `tests/test_vllm_backends.py`.

**Approach:** Curate verified choices, explain backend/model restrictions, preserve unset defaults. Evaluate effective structured/raw options; reject known errors and warn when engine/model compatibility cannot be established. Read local model metadata only; never infer quantization from model names. Architecture checks use selected GPUs only when selection is resolvable.

**Test scenarios:**

- Null/absent values emit no flags; explicit valid choices round-trip through both builders.
- B12X attention rejects explicit FP16, unsupported cache, and context parallelism >1; BF16 with auto/FP8 cache is allowed subject to evidence.
- B12X MoE rejects expert parallelism and definitely incompatible model metadata; unknown format produces guidance.
- Supported, unsupported, mixed, unknown, and explicitly selected GPUs get accurate feedback.
- Trailing raw values, equals syntax, malformed raw input, and switching back to automatic preserve established precedence without bypassing known conflict checks.

**Verification:** Catalog and focused compatibility tests pass, including failures before engine startup.

### U3. Integrate advice, launch enforcement, and UI verification

**Requirements:** R1–R5. **Dependencies:** U1, U2.

**Files:** `src/local_llm_launcher/api.py`, `src/local_llm_launcher/engines/vllm_native.py`, `src/local_llm_launcher/engines/vllm_docker.py`, `tests/test_vllm_backends.py`, `frontend/tests/browser-vllm-backends.mjs`, `README.md`, generated `src/local_llm_launcher/static/` assets if changed.

**Approach:** Apply the same checks during advice and launch; expose per-control evidence via existing warnings. Validate before Docker secret files are created. Verify the existing generic frontend actually renders and removes these overrides, shows incompatibility errors, and isolates native/Docker state. Prepare local PR text with dependencies on PR #17 and verification limits.

**Test scenarios:**

- Advice and direct launch fail consistently for definite incompatibilities.
- Unknown evidence is visible and does not masquerade as proven GPU compatibility.
- Built browser UI shows all three selectors; selection reaches advice/launch requests; automatic clears overrides; invalid configuration disables launch.
- Desktop/mobile controls remain usable; existing launch/update/placement checks still pass.

**Verification:** Full Python suite, frontend build, relevant browser checks, diff review, local commit; no push.

## Risks and Sources

Runtime help can fail due to CUDA imports; classify that as unknown. Package presence is not proof that compiled GPU kernels or a model work. Real GPU benchmarking remains unavailable here. Backend overrides may change memory/performance, so avoid promising fit from these checks. Native probing is bounded and cached because advice polls every eight seconds.

- Existing patterns: `catalog.py`, `engines/_args.py`, `hardware.py`, `frontend/src/views/Launch.jsx`, `tests/test_placement.py`.
- Research: `docs/research/2026-09-27-vllm-rtx-5060ti-followup.md`.
- [v0.30 backend choices](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/kernel.py).
- [B12X attention constraints](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/b12x.py).
- [B12X optional dependency and model support](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/b12x.md).

## Verification Contract

- Run focused tests as each unit lands, then the full isolated `pytest` suite.
- Build frontend with `npm run build`; run the built-UI browser checks against an isolated local app.
- Review substantive code changes and resolve actionable findings before committing.
- Record observed outcomes and limits in a worklog outside this plan.

## Definition of Done

All three controls work with native and Docker argument generation. Automatic defaults remain unchanged. Definite errors are caught at advice and launch boundaries; uncertain evidence is labeled. Tests and browser checks pass. Documentation and a local PR description are ready, and changes are committed on `feat/vllm-blackwell-controls` without pushing.
