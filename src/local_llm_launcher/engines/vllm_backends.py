"""Conservative checks for explicitly selected vLLM computation backends."""
from __future__ import annotations

import os
import re
import shlex

from . import vllm_capabilities
from .. import hardware as detection
from .placement import parse_device_ids

BACKENDS = ('linear_backend', 'moe_backend', 'attention_backend')
_VALUE_OPTIONS = {f'--{key.replace("_", "-")}': key for key in (
    *BACKENDS, 'dtype', 'kv_cache_dtype', 'decode_context_parallel_size',
    'prefill_context_parallel_size', 'quantization', 'tensor_parallel_size')}
# vLLM's own --device-ids is not the launcher's device_ids (which sets CUDA_VISIBLE_DEVICES).
_VALUE_OPTIONS.update({'--device-ids': 'vllm_device_ids', '-dcp': 'decode_context_parallel_size',
                       '-pcp': 'prefill_context_parallel_size', '-q': 'quantization',
                       '-tp': 'tensor_parallel_size'})
_BOOL_OPTIONS = {'--enable-expert-parallel': True, '-ep': True,
                 '--no-enable-expert-parallel': False}


def selected(config):
    """Cheap detection, including malformed raw overrides that need a useful error."""
    # vLLM accepts --linear_backend as --linear-backend; values do not matter here.
    raw = str(config.get('extra_args') or '').replace('_', '-')
    return any(config.get(key) is not None for key in BACKENDS) or any(
        option in raw for option in _VALUE_OPTIONS if _VALUE_OPTIONS[option] in BACKENDS)


def effective_config(config):
    """Read targeted trailing CLI overrides without altering command argument order."""
    result = dict(config)
    try:
        tokens = shlex.split(str(config.get('extra_args') or ''))
    except ValueError as exc:
        raise ValueError(f'Extra arguments have invalid quoting: {exc}') from None
    i = 0
    while i < len(tokens):
        option, sep, value = tokens[i].partition('=')
        if option == '--':
            break
        if option.startswith('--'):
            # vLLM's FlexibleArgumentParser reads underscores in option names as dashes.
            option = option.replace('_', '-')
        if option in _BOOL_OPTIONS:
            if sep:
                raise ValueError(f'{option} does not take a value; use --no-enable-expert-parallel to disable it.')
            result['enable_expert_parallel'] = _BOOL_OPTIONS[option]
        elif option in _VALUE_OPTIONS:
            if not sep:
                i += 1
                if i == len(tokens) or tokens[i].startswith('-'):
                    raise ValueError(f'{option} requires a value.')
                value = tokens[i]
            if not value:
                raise ValueError(f'{option} requires a value.')
            result[_VALUE_OPTIONS[option]] = value
        i += 1
    for key in ('linear_backend', 'moe_backend'):
        if isinstance(result.get(key), str):
            # vLLM parses these values with s.lower().replace('-', '_').
            result[key] = result[key].lower().replace('-', '_')
    return result


def _model_config(model):
    cfg = (model or {}).get('config') or {}
    return cfg if isinstance(cfg, dict) else {}


def _gpu_warning(config, hardware):
    hardware = hardware or {}
    gpus = hardware.get('gpus') or []
    # Every GPU number here is nvidia-smi's: native launches pin CUDA_DEVICE_ORDER.
    mask = config.get('device_ids')
    if mask is None or not str(mask).strip():
        mask = hardware.get('cuda_visible_devices')
    try:
        ids = parse_device_ids(mask)
        if mask is not None and ids is None:
            raise ValueError('an empty mask hides every GPU')
        picks = parse_device_ids(config.get('vllm_device_ids'))
        if picks is not None and hardware.get('container'):
            raise ValueError('a container numbers its GPUs itself')
        if picks is not None:
            # vLLM's --device-ids are positions within the mask when one is set.
            ids = picks if ids is None else [ids[i] for i in picks]
    except (ValueError, IndexError):
        return 'Selected GPU identities could not be verified.'
    if ids is not None:
        by_index = {gpu.get('index'): gpu for gpu in gpus}
        if any(i not in by_index for i in ids):
            return 'Selected GPU identities could not be verified.'
        gpus = [by_index[i] for i in ids]
    if not gpus:
        return 'GPU architecture could not be verified on this computer.'
    capabilities = [str(gpu.get('compute_capability') or '') for gpu in gpus]
    if any(not re.fullmatch(r'\d+\.\d+', value) for value in capabilities):
        return 'Some selected GPU architectures could not be verified.'
    supported = [value in ('12.0', '12.1') for value in capabilities]
    if not any(supported):
        raise ValueError('B12X requires selected NVIDIA GPUs with compute capability 12.0 or 12.1 (SM120/SM121).')
    if not all(supported):
        # A resolved device pool fully consumed by tensor parallel workers is definite.
        # Without that evidence vLLM may use only a supported subset of the pool.
        if ids is not None and str(config.get('tensor_parallel_size')) == str(len(gpus)):
            raise ValueError('B12X requires SM120/SM121 on every participating GPU; the selected tensor parallel group includes an unsupported GPU.')
        return ('Available GPUs have mixed architectures; the GPUs actually used by vLLM '
                'could not be verified. B12X requires SM120/SM121 on every participating GPU.')
    return ''


def check(config, model=None, hardware=None, capabilities=None):
    """Return per-control warnings; reject only configurations known incompatible."""
    if not selected(config):
        return {}
    cfg = effective_config(config)
    caps = capabilities or {}
    warnings = {}
    if cfg.get('attention_backend') == 'B12X_ATTN':
        raise ValueError('vLLM v0.30 names the attention backend B12X, not B12X_ATTN. Select B12X.')
    active = {key: cfg[key] for key in BACKENDS if cfg.get(key) is not None}
    for key, backend in active.items():
        if not isinstance(backend, str) or not backend:
            raise ValueError(f'{key} must be a backend name or automatic.')
        flag = '--' + key.replace('_', '-')
        if caps.get('flags') is not None and flag not in caps['flags']:
            raise ValueError(f'The selected vLLM runtime does not support {flag}. Choose automatic or update that runtime.')
        choices = (caps.get('choices') or {}).get(flag)
        if choices is not None and backend not in choices:
            raise ValueError(f'The selected vLLM runtime does not accept {backend} for {flag}. Choose automatic or a supported backend.')
        messages = ['Backend and model compatibility is ultimately checked by vLLM at startup.']
        if caps.get('flags') is None:
            messages.append(caps.get('message') or 'Runtime flag support could not be verified.')
        if config.get('extra_args'):
            messages.append('Extra arguments can change behavior; only recognized backend compatibility options are checked.')
        warnings[key] = {'level': 'yellow', 'message': ' '.join(messages)}
    b12x = [key for key, value in active.items() if value in ('b12x', 'B12X')]
    sm12x = [key for key, value in active.items() if value in ('b12x', 'B12X', 'flashinfer_b12x')]
    if b12x:
        if caps.get('b12x') is False:
            raise ValueError('The selected vLLM runtime is missing the optional b12x package. Install vllm[b12x] in that runtime or select automatic.')
        if caps.get('b12x') is None:
            for key in b12x:
                warnings[key]['message'] += ' Optional b12x package availability is unverified; install vllm[b12x] in the selected native environment or Docker image.'
    if sm12x:
        message = _gpu_warning(cfg, hardware)
        if message:
            for key in sm12x:
                warnings[key]['message'] += ' ' + message
    metadata = _model_config(model)
    text = metadata.get('text_config')
    text = text if isinstance(text, dict) else {}
    if cfg.get('attention_backend') == 'B12X':
        dtype = cfg.get('dtype') or 'auto'
        if dtype == 'auto':
            dtype = text.get('dtype') or text.get('torch_dtype') or metadata.get('dtype') or metadata.get('torch_dtype')
            if dtype in ('float32', 'float'):
                # vLLM's automatic dtype downcasts float32 checkpoints to the GPU's first
                # supported dtype: bfloat16 on SM80+ (so on SM120/SM121).
                dtype = 'bfloat16'
                warnings['attention_backend']['message'] += (
                    ' The float32 checkpoint runs as bfloat16 under automatic dtype; '
                    'embedding (pooling) models use float16 instead, which B12X does not support.')
        if dtype not in (None, 'bfloat16', 'bf16'):
            raise ValueError('B12X attention requires BF16 model computation. Select bfloat16 for dtype.')
        if dtype is None:
            warnings['attention_backend']['message'] += ' Automatic model dtype is unknown; B12X attention requires bfloat16 (BF16).'
        if (cfg.get('kv_cache_dtype') or 'auto') not in ('auto', 'fp8', 'fp8_e4m3', 'bfloat16'):
            raise ValueError('B12X attention requires automatic/BF16 or FP8 E4M3 KV cache, not the selected cache format.')
        for key in ('decode_context_parallel_size', 'prefill_context_parallel_size'):
            value = cfg.get(key)
            value = 1 if value is None else value
            if isinstance(value, bool) or not re.fullmatch(r'[0-9]+', str(value)) or int(value) < 1:
                raise ValueError(f'{key} must be a positive whole number.')
            if int(value) > 1:
                raise ValueError('B12X attention does not support context parallelism; set decode and prefill context parallel sizes to 1.')
        if text.get('kv_lora_rank') or metadata.get('kv_lora_rank'):
            raise ValueError('B12X attention does not support MLA (latent attention) models.')
    if cfg.get('moe_backend') == 'b12x':
        if cfg.get('enable_expert_parallel'):
            raise ValueError('B12X MoE does not support expert parallelism; disable --enable-expert-parallel or select automatic.')
        quant = metadata.get('quantization_config') or text.get('quantization_config') or {}
        method = cfg.get('quantization') or (quant.get('quant_method') if isinstance(quant, dict) else None)
        if method in ('awq', 'awq_marlin', 'gptq', 'gptq_marlin', 'bitsandbytes', 'fp8', 'fbgemm_fp8'):
            raise ValueError('B12X MoE requires compatible NVFP4 or MXFP4 expert weights; the selected quantization format is incompatible.')
        warnings['moe_backend']['message'] += ' Requires compatible NVFP4 or MXFP4 expert weights; model format and compiled kernels must still be checked by vLLM.'
    return warnings


def server_hardware(mode, hardware):
    """Hardware as the server sees it: native vLLM inherits the launcher's GPU mask."""
    if mode == 'vllm-docker':
        return {**hardware, 'container': True}  # Docker selects and renumbers GPUs itself
    if not mode.startswith('vllm'):
        return hardware
    return {**hardware, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def validate(mode, model, config, hardware=None, binary='vllm', *, wait=True):
    """The one validation path for advice and launch: probe the runtime, then check.

    Without `hardware`, detects it (detect_hardware never raises).
    """
    if not mode.startswith('vllm') or not selected(config):
        return {}
    evidence = vllm_capabilities.probe(mode, binary, wait=wait)
    hardware = detection.detect_hardware().to_dict() if hardware is None else hardware
    return check(config, model, server_hardware(mode, hardware), evidence)
