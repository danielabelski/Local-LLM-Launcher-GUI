# feat: Add vLLM backend controls for consumer Blackwell GPUs

## Publication state

Branch: `feat/vllm-blackwell-controls`, based at `27fb558`. Its prerequisite, [PR #17](https://github.com/jimdawdy-hub/Local-LLM-Launcher-GUI/pull/17), is merged, and `master` has the same tree as `27fb558`, so this PR targets `master` and its diff contains only this change.

## Summary

- Add advanced linear, MoE (mixture-of-experts), and attention backend selectors, including FlashInfer options and optional B12X implementations for SM120/121 GPUs such as RTX 5060 Ti.
- Preserve automatic defaults; inspect the configured native executable's supported flags/choices and optional package availability. Keep Docker evidence independent and explicitly unverified.
- Reject known precision/cache/parallelism/GPU conflicts through both advice and launch. Honor targeted trailing raw overrides and distinguish uncertainty from incompatibility.
- Use the actual vLLM 0.30 attention spelling `B12X`, correcting stale `B12X_ATTN` release discussion.
- Validate once, in `ServerManager.launch`, before the registry lock is taken, so a slow runtime probe never freezes server status, logs or stop. Advice never waits for a probe: it shows the last evidence (or "still being checked") and refreshes in the background. Each runtime is probed by one process at a time, `vllm serve --help=all` gets 60 seconds, and a background check that cannot start degrades to unknown evidence and is retried on the next poll.
- Cache runtime evidence for 10 minutes, but drop it at once when the executable, its interpreter, or any directory on its import path changes (installing a package changes its directory), so an open Launch page no longer re-runs the heavy probe every minute.
- Normalize the GPU list once (`placement.normalize_device_ids`) and use that value for advice, checks, `CUDA_VISIBLE_DEVICES` and Docker `--gpus`. Only a comma list of distinct nvidia-smi numbers or CUDA GPU UUIDs is accepted (not `0 1`, `gpu1` or `0,00`), and `device_ids=0` is GPU 0. Native launches always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so every GPU number, including an inherited `CUDA_VISIBLE_DEVICES`, means the card nvidia-smi shows. Advice's memory budget and the backend check now use the same GPUs.
- Read raw options the way vLLM 0.30's parser does: underscore spellings (`--linear_backend`); `--moe-backend`/`--linear-backend` values regardless of case or dashes (`B12X`, `flashinfer-b12x`); and vLLM's own `--device-ids` (nvidia-smi GPUs, or positions within `CUDA_VISIBLE_DEVICES` when one is set; reported as unverified inside Docker, which renumbers GPUs).
- Fixes to existing behavior found by the whole-repository reviews: the updater allows a fresh vLLM build 300 seconds for its first `--version` run (was 30); snapshot downloads skip top-level `.bin`/`.pt` copies when a repo ships top-level safetensors; installed-model size counts only the top-level files vLLM's automatic loader reads (safetensors over `.bin`, Mistral `consolidated` files over the sharded copy, index-listed files only).

No engine upgrade or dependency installation is included. Automatic NVFP4 improvements are supplied by vLLM itself.

## Verification

- 360 Python tests passed (repeated consecutive runs), including real temporary runtime probe tests, lock-responsiveness, per-runtime concurrency and background-retry tests, and API/command integration checks. Every fix was written test-first and its test was seen failing before the fix.
- Installed-model sizes compared on a real Hugging Face cache (32 models): 31 unchanged; `sentence-transformers/all-MiniLM-L6-v2` went from 0.27 GiB to 0.08 GiB, because it ships the same weights as safetensors, PyTorch `.bin` and two OpenVINO files and vLLM loads only `model.safetensors`.
- End-to-end API smoke with a fake runtime whose help takes 4 seconds: advice returned in 0.01s, the server list in 0.00s during a launch, and the launch shared the in-flight probe (4.0s total, one help run).
- Frontend build and four browser checks passed on the original commit, including the new backend controls and mobile overflow check. They were not re-run after the review fixes, which changed no frontend code (only the `device_ids` help text in the flag catalog).
- Independent code review findings fixed over several rounds: this change's own commit (registry lock, concurrent probes, probe timeout, float32 automatic dtype, duplicated validation, CUDA device ordering, device-list parsing, the `B12X_ATTN` message); a whole-repository review (GPU-list normalization, `device_ids=0`, per-runtime probe locks, background-start retry, probe cache lifetime, underscore spellings, updater version timeout, duplicate weight files); and a second whole-repository review (vLLM's real `--device-ids` option, backend value spelling, stricter GPU lists, advice/check GPU agreement, probe ordering, top-level-only weight files, a cache-eviction race). vLLM behavior claims were checked against the v0.30.0 source (`arg_utils.py`, `argparse_utils.py`, `default_loader.py`, `weight_utils.py`, `config/model.py`, `platforms/cuda.py`).
- A complexity review removed the probe cache's size limit and eviction code, a test-only helper, a download helper with one caller, and an unused constant.
- Whitespace/diff check passed.
- Existing ESLint command cannot run because the repository lacks its configuration file.

## Limitations

No GPU inference or performance benchmark was possible on VM100. Package/CLI evidence does not guarantee a particular model's compiled kernels work. Docker package support remains unverified by design. No performance improvement percentage is claimed.

## Post-Deploy Monitoring & Validation

Owner: launcher maintainer. During the first local launch after adopting this change, compare automatic selection with one compatible explicit backend using the same model and context size. Review server logs for the selected backend, healthy startup, and errors containing `b12x`, `unsupported`, `unrecognized arguments`, or `No module named`. Record time to first token, generation speed, and peak GPU memory on the actual card; no centralized monitoring service is required.

Healthy signals: advice correctly identifies uncertainty, the engine starts, and logs reflect the intended backend. If an override fails or regresses performance, return its selector to automatic. If launcher validation blocks a known working configuration incorrectly, retain logs and revert this commit pending a fix. Do not infer container support from native package checks.

## Implementation record

See `docs/plans/2026-09-27-001-feat-vllm-backend-controls-plan.md` and `docs/worklogs/2026-09-27-vllm-backend-controls.md`.
