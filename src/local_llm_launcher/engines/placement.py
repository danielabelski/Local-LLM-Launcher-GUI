"""Validation shared by advice and command builders; native memory policy wrapper."""
import math
import re
from .. import hardware


def validate(engine, config, numa=None):
    if config.get('numactl_interleave'):
        if engine == 'vllm-docker':
            raise ValueError('Memory interleaving supports native engines only; turn it off for Docker.')
        numa = numa if numa is not None else hardware.detect_numa()
        if not numa.get('linux') or not numa.get('numactl_path'):
            raise ValueError('Memory interleaving requires Linux and numactl installed on PATH.')
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
    for key in ('main_gpu', 'n_cpu_moe', 'ubatch_size'):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == 'ubatch_size' else 0)):
            raise ValueError(f'{key} must be a whole number at least {1 if key == "ubatch_size" else 0}.')
    if devices and config.get('main_gpu') is not None and config['main_gpu'] >= len(devices):
        raise ValueError('Main GPU index must refer to a selected device (numbered from zero).')
    if config.get('cpu_moe') and config.get('n_cpu_moe', 0):
        raise ValueError('Choose all CPU experts or a CPU expert layer count, not both.')
    if config.get('numa') not in (None, 'distribute', 'isolate', 'numactl'):
        raise ValueError('NUMA mode must be distribute, isolate, numactl, or unset.')
    if mode == 'tensor':
        if config.get('flash_attn') == 'off':
            raise ValueError('Tensor splitting requires flash attention on or auto.')
        if any(config.get(key, 'f16') not in ('f16', 'bf16', 'f32') for key in ('cache_type_k','cache_type_v')):
            raise ValueError('Tensor splitting requires uncompressed K and V caches (f16, bf16, or f32).')


def wrap(argv, config):
    if config.get('numactl_interleave'):
        return [hardware.detect_numa()['numactl_path'], '--interleave=all', '--'] + argv
    return argv
