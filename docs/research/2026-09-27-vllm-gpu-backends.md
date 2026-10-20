# vLLM GPU backend reference

These notes concern the tagged v0.30 backend interfaces. Check the installed
engine's own help and supported kernels before applying them to another release.

- Optional B12X backends target SM120/SM121. Other GPU architectures require
  a compatible backend; a product-family label alone is insufficient.
- The attention selector is `B12X`, as defined by the
  [tagged registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/registry.py).
- B12X requires the optional package in the selected engine environment.
  See the [versioned backend documentation](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/b12x.md).
- Weight quantization and K/V-cache precision are separate compatibility choices.
- Preserve automatic defaults, check the selected devices and model format,
  and compare performance on the intended workload before retaining an override.
- NUMA controls should follow OS-visible topology on any supported machine.

These are compatibility notes, not an inference-performance guarantee.
