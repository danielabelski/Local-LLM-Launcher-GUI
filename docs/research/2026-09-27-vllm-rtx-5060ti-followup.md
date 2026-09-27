# vLLM RTX 5060 Ti follow-up

Date: 2026-09-27. Research only; implementation and engine upgrades remain pending after the open-issue pass.

Implementation follow-up corrected the attention selector to `B12X`: the [tagged v0.30 registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/registry.py) is authoritative over the stale `B12X_ATTN` spelling in release discussion. The research-stage status above records the original stopping point; subsequent implementation is described in the backend-controls plan and worklog.

## Current implementation

PR #17 fixes explicit negative vLLM boolean flags and adds optional native NUMA interleaving (distributing RAM allocation across visible CPU memory nodes). It does not add attention, linear, or MoE backend selectors. The existing KV-cache selector offers only auto/fp8. Extra arguments remain available for supported CLI options.

## Release and hardware

- Latest release checked: [v0.30.0, September 22](https://github.com/vllm-project/vllm/releases/tag/v0.30.0).
- NVIDIA lists the RTX 5060 Ti under [compute capability 12.0](https://developer.nvidia.com/cuda/gpus), also called SM120. Do not assume SM100 data-center Blackwell changes apply to this card.

## Relevant improvements and missing controls

1. **Automatic NVFP4 selection:** v0.30 prefers W4A4 (4-bit weights and intermediate activations) over weight-only W4A16 for compatible NVFP4 model files on SM120/121. This requires an engine upgrade, not a new launcher toggle. `--linear-backend flashinfer_cutlass` provides an explicit comparison option. Upstream measurements were on RTX PRO 6000 Blackwell Max-Q, not a 5060 Ti; workload-dependent benefits must be measured on the actual card. [Upstream fix](https://github.com/vllm-project/vllm/pull/55170).
2. **Linear and MoE backend selection:** expose `--linear-backend` and `--moe-backend`, preserving automatic selection by default. Optional `b12x` implementations target SM120/121 and require the `vllm[b12x]` extra package. MoE (mixture-of-experts) support depends on model format; expert parallelism is unsupported by this backend. The current source updater does not request this extra package. [Versioned documentation](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/b12x.md).
3. **Attention backend selection:** `--attention-backend B12X` is an optional SM120/121 implementation. It supports BF16 model computation and BF16/FP8 E4M3 KV cache (stored attention data). It does not support context parallelism or MLA (latent attention). Do not combine it indiscriminately with context splitting or NVFP4 KV-cache settings. [Upstream implementation](https://github.com/vllm-project/vllm/pull/52017).
4. **Cache follow-up:** the release also includes NVFP4 KV-cache work, but this research has not established its full SM120/backend compatibility. Do not recommend it as a verified 5060 Ti preset. Investigate independently from B12X attention. [Upstream change](https://github.com/vllm-project/vllm/pull/55031).

## Proposed next implementation scope

- Add version-aware backend selectors and optional-dependency detection.
- Explain supported GPU architecture and model/cache combinations in the UI.
- Retain auto defaults and validate incompatible combinations before launch.
- Verify installed-engine CLI support, command generation, and saved-profile round trips.
- Benchmark compatible models on the real 5060 Ti before claiming performance gains; VM100 has no available GPU for this test.
- Keep generic multi-CPU NUMA controls independent of Proxmox-specific guidance.

No application code or installed engines were changed during this follow-up.
