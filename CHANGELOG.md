# Changelog

## 2026-10-20 — v0.5.7

- Keep downloaded models visible when a newer cache revision contains metadata
  without model weights.
- Add **Edit** beside favorite launches to open their saved settings before starting.
- Add **Archive** with confirmation, the exact recovery-file location, and one
  private file containing archived favorites (#30, #32).
- Add **Reload previous settings** with the observed outcome of the retained
  model launch; preserve observed successful/failed outcomes across restarts (#31).
- Use bright green, yellow and red status badges and larger checkmark indicators
  in light and dark themes.
- Replace machine-specific examples and inference-run anecdotes with general
  hardware and engine guidance.

## 2026-10-18 — v0.5.6

- Add named launch favorites, server configuration display, and benchmark
  summaries from supported engine logs (#28).
- Keep host binding, credentials and internal fields out of favorites. Relaunches
  follow the current Settings LAN policy.

## 2026-10-16 — v0.5.5

- Add temperature, system prompt and explicit host controls.
- Expose CPU MoE placement, expert-cache, feed-forward offload and lazy-loading controls.
- Add context shifting, slots, terminal color and system-prompt-file settings.
- Add reasoning format/effort/budget and speculative draft-model controls.
- Validate incompatible loading and placement settings before launch.

## 2026-10-06 — v0.5.1

- Make the MTP draft count configurable, with a conservative default of one token.
- Account for micro-batch/context working buffers, quantized-cache layer copies,
  conversation slots and recurrent state in llama.cpp memory estimates.
- Explain CPU fallback when a model exceeds GPU capacity.
- Add a configurable freeze watchdog for llama.cpp.

## 2026-10-05 — v0.5.0

- Read model metadata for hybrid, sliding-window and shared-cache estimates.
- Include speculative decoding memory and text-only encoder exclusions.
- Allow an explicit warned launch when memory estimates alone are unfavorable;
  invalid configurations and missing engines remain blocked.

## 2026-10-04 — v0.4.8

- Read the sidebar version from the installed backend.

## 2026-10-04 — v0.4.7

- Add exact-name allowed-host configuration for private reverse proxies.
- Keep loopback binding and local-only engine-update authorization.

## 2026-10-04 — v0.4.6

- Allow supported tensor splitting with quantized K/V cache and Flash Attention.
- Distinguish model weight compression from conversation-cache precision.

## 2026-09-28 — v0.4.5

- Add MTP selection for supported vLLM and llama.cpp model families.
- Detect engine versions, supported methods and separate MTP head files.
- Explain unsupported engine/model combinations and preserve raw-argument overrides.
- Improve plain-English settings guidance.

## 2026-09-27 — v0.4.0

- Add opt-in, isolated engine source builds with requirement checks and rollback paths.
- Add llama.cpp device selection, GPU proportions, tensor splitting and CPU expert controls.
- Add optional NUMA placement based on visible CPU/memory topology.
- Add vLLM linear, attention and MoE backend controls, including selected-runtime
  capability checks and architecture-specific validation.
- Translate memory-mapping controls for newer llama.cpp releases.
- Improve settings isolation, launch concurrency, resource estimates and diagnostics.

## 2026-08-11 — v0.3.1

- Fix server lifecycle, shutdown, port reservation and loopback binding.
- Improve Hugging Face search and link model names to repository pages.

## 2026-06-14 — v0.3.0

- Redesign the interface and add light/dark themes.

## 2026-06-13 — v0.2.0

- Add cache offload and automatic selection of a free server port.
- Protect settings and Docker credentials, improve modal/toast accessibility,
  and bound concurrent downloads.
- Add PATCH support for settings and make hardware caching thread-safe.

## 2026-06-11

- Add Open WebUI integration and connection configuration.
- Add text-only model advice and per-GPU weight/cache/headroom estimates.
- Refresh memory readings and translate common parser, memory and argument errors.
- Prefer configured accelerator builds and preserve explicit launch arguments.

## 2026-06-10 — Initial build

- Add the Python API, hardware detection, flag catalogs, model discovery/downloads,
  launch advice, engine adapters, server management and browser interface.
- Adapt model discovery and configuration concepts from the MIT-licensed vllm-cli.
