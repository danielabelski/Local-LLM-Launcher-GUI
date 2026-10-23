"""Validation shared by advice and command builders; native memory policy wrapper."""
import math
import re
from .. import hardware


_GPU_ITEM = re.compile(r'[0-9]+|(?:GPU|MIG)-[0-9A-Za-z/-]+')  # nvidia-smi number or CUDA UUID


def normalize_device_ids(raw):
    """The one spelling of a GPU list that advice, checks and the engine all use.

    Returns None when unset. Raises ValueError for anything but a comma list of
    distinct nvidia-smi numbers or CUDA GPU UUIDs (so not '0 1', 'gpu1' or '0,00').
    """
    text = '' if raw is None else str(raw).strip()
    if not text:
        return None
    items = [item.strip() for item in text.split(',') if item.strip()]
    items = [str(int(item)) if item.isascii() and item.isdigit() else item for item in items]
    if (isinstance(raw, bool) or not items or len(set(items)) != len(items)
            or not all(_GPU_ITEM.fullmatch(item) for item in items)):
        raise ValueError('Enter GPU numbers separated by commas, such as 0,1.')
    return ','.join(items)


def parse_device_ids(raw):
    """GPU numbers (nvidia-smi order) in the order given; None when unset. Raises ValueError."""
    ids = normalize_device_ids(raw)
    return None if ids is None else [int(item) for item in ids.split(',')]


def validate(engine, config, numa=None):
    host = config.get('host')
    if host is not None and (not isinstance(host, str) or any(ch.isspace() for ch in host.strip())):
        raise ValueError('Enter a valid listen address without spaces, such as 127.0.0.1 or 0.0.0.0.')
    if config.get('numactl_interleave'):
        if engine in ('vllm-docker', 'tgi'):
            raise ValueError('Memory interleaving supports native engines only; turn it off for Docker.')
        numa = numa if numa is not None else hardware.detect_numa()
        if not numa.get('linux') or not numa.get('numactl_path'):
            raise ValueError('Memory interleaving requires Linux and numactl installed on PATH.')
    if engine.startswith('vllm') or engine in ('sglang', 'tgi'):
        normalize_device_ids(config.get('device_ids'))
    if engine != 'llamacpp':
        return
    mode = config.get('split_mode') or 'layer'
    if mode not in ('layer', 'row', 'none', 'tensor'):
        raise ValueError('Choose layer, row, none, or tensor for the GPU split style.')
    devices = str(config.get('device') or '').split(',') if config.get('device') else []
    if devices and (any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', d) for d in devices) or len(set(devices)) != len(devices)):
        raise ValueError('Use distinct llama.cpp device names separated by commas, such as CUDA0,CUDA1.')
    if 'none' in devices and len(devices) != 1:
        raise ValueError('Device none cannot be combined with GPU devices.')
    ratios = config.get('tensor_split')
    if ratios is not None:
        try:
            values = [float(v) for v in str(ratios).split(',')]
            if not values or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError()
        except ValueError:
            raise ValueError('GPU split proportions must be positive finite numbers, such as 40,40,40.') from None
        if mode == 'none' or devices == ['none']:
            raise ValueError('GPU proportions require a multi-GPU split style and GPU devices.')
        if devices and len(values) != len(devices):
            raise ValueError('Provide one GPU split proportion per selected device.')
    for key in ('main_gpu', 'n_cpu_moe', 'moe_cache_mib', 'n_cpu_ffn', 'spec_draft_ngl', 'spec_draft_n_max', 'spec_draft_n_min', 'ubatch_size'):
        value = config.get(key)
        min_val = 1 if key in ('ubatch_size', 'spec_draft_n_max') else 0
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < min_val):
            raise ValueError(f'{key} must be a whole number at least {min_val}.')
    budget = config.get('reasoning_budget')
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < -1):
        raise ValueError('reasoning_budget must be a whole number at least -1.')
    if config.get('spec_draft_n_min') is not None and config.get('spec_draft_n_max') is not None:
        if config['spec_draft_n_min'] > config['spec_draft_n_max']:
            raise ValueError('Minimum speculative draft tokens cannot exceed maximum speculative draft tokens.')
    if config.get('system_prompt') and config.get('system_prompt_file'):
        raise ValueError('Choose either an inline system prompt or a system prompt file, not both.')
    if devices and config.get('main_gpu') is not None and config['main_gpu'] >= len(devices):
        raise ValueError('Main GPU index must refer to a selected device (numbered from zero).')
    if config.get('cpu_moe') and config.get('n_cpu_moe', 0):
        raise ValueError('Choose all CPU experts or a CPU expert layer count, not both.')
    if config.get('numa') not in (None, 'distribute', 'isolate', 'numactl'):
        raise ValueError('NUMA mode must be distribute, isolate, numactl, or unset.')
    if config.get('lazy_mode') not in (None, 'auto', 'on', 'off'):
        raise ValueError('Lazy mode must be auto, on, off, or unset.')
    if config.get('color') not in (None, 'auto', 'on', 'off'):
        raise ValueError('Console color mode must be auto, on, off, or unset.')
    if config.get('reasoning') not in (None, 'auto', 'on', 'off'):
        raise ValueError('Reasoning mode must be auto, on, off, or unset.')
    if config.get('reasoning_format') not in (None, 'auto', 'deepseek', 'deepseek-legacy', 'none'):
        raise ValueError('Reasoning format must be auto, deepseek, deepseek-legacy, none, or unset.')
    if config.get('reasoning_effort') not in (None, 'default', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        raise ValueError('Reasoning effort must be default, minimal, low, medium, high, xhigh, max, or unset.')
    if mode == 'tensor':
        if config.get('flash_attn') == 'off':
            raise ValueError('Tensor splitting requires flash attention on or auto.')
        # Upstream PR #23792 added tensor splitting with quantized KV cache.
        # Model/backend-specific support belongs to the installed engine, not
        # a blanket ban based on cache precision or model weight quantization.


def wrap(argv, config):
    if config.get('numactl_interleave'):
        return [hardware.detect_numa()['numactl_path'], '--interleave=all', '--'] + argv
    return argv
