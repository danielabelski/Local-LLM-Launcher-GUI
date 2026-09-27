# feat: Add vLLM backend controls for consumer Blackwell GPUs

## Publication state

Prepared locally only. Branch: `feat/vllm-blackwell-controls`. Depends on the issue fixes in [PR #17](https://github.com/jimdawdy-hub/Local-LLM-Launcher-GUI/pull/17), based at `27fb558`. When publication is authorized, use `fix/open-issues` as the stacked PR base while #17 remains open, or the integrated base after it merges. No branch was pushed and no GitHub PR was opened for this change.

## Summary

- Add advanced linear, MoE (mixture-of-experts), and attention backend selectors, including FlashInfer options and optional B12X implementations for SM120/121 GPUs such as RTX 5060 Ti.
- Preserve automatic defaults; inspect the configured native executable's supported flags/choices and optional package availability. Keep Docker evidence independent and explicitly unverified.
- Reject known precision/cache/parallelism/GPU conflicts through both advice and launch. Honor targeted trailing raw overrides and distinguish uncertainty from incompatibility.
- Use the actual vLLM 0.30 attention spelling `B12X`, correcting stale `B12X_ATTN` release discussion.

No engine upgrade or dependency installation is included. Automatic NVFP4 improvements are supplied by vLLM itself.

## Verification

- 283 Python tests passed, including real temporary runtime probe tests and API/command integration checks.
- Frontend build and four browser checks passed, including the new backend controls and mobile overflow check.
- Independent focused code review findings fixed and rechecked.
- Whitespace/diff check passed.
- Existing ESLint command cannot run because the repository lacks its configuration file.

## Limitations

No GPU inference or performance benchmark was possible on VM100. Package/CLI evidence does not guarantee a particular model's compiled kernels work. Docker package support remains unverified by design. No performance improvement percentage is claimed.

## Post-Deploy Monitoring & Validation

Owner: launcher maintainer. During the first local launch after adopting this change, compare automatic selection with one compatible explicit backend using the same model and context size. Review server logs for the selected backend, healthy startup, and errors containing `b12x`, `unsupported`, `unrecognized arguments`, or `No module named`. Record time to first token, generation speed, and peak GPU memory on the actual card; no centralized monitoring service is required.

Healthy signals: advice correctly identifies uncertainty, the engine starts, and logs reflect the intended backend. If an override fails or regresses performance, return its selector to automatic. If launcher validation blocks a known working configuration incorrectly, retain logs and revert this commit pending a fix. Do not infer container support from native package checks.

## Implementation record

See `docs/plans/2026-09-27-001-feat-vllm-backend-controls-plan.md` and `docs/worklogs/2026-09-27-vllm-backend-controls.md`.
