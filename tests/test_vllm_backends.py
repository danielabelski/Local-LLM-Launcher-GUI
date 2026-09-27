"""Backend validation must follow the actual runtime and trailing CLI overrides."""
import pytest

from local_llm_launcher import catalog
from local_llm_launcher.engines import vllm_backends as backends
from local_llm_launcher.engines._args import build_args_and_env

FLAGS = ['--linear-backend', '--moe-backend', '--attention-backend']
CAPS = {'flags': FLAGS, 'b12x': True, 'source': 'native', 'version': '0.30.0'}
HARDWARE = {'gpus': [{'index': 0, 'compute_capability': '12.0'}]}
MODEL = {'config': {'torch_dtype': 'bfloat16'}}


def check(config, **kwargs):
    return backends.check(config, model=kwargs.pop('model', MODEL),
                          hardware=kwargs.pop('hardware', HARDWARE),
                          capabilities=kwargs.pop('capabilities', CAPS))


def test_catalog_automatic_omits_flags_and_choices_emit_exact_case():
    specs = {item['key']: item for item in catalog.load_catalog('vllm')['flags']}
    for key in backends.BACKENDS:
        assert specs[key]['default'] is None
        assert build_args_and_env('vllm', {key: None})[0] == []
        for value in specs[key]['choices'][1:]:
            assert build_args_and_env('vllm', {key: value})[0] == [specs[key]['flag'], value]
    assert 'B12X' in specs['attention_backend']['choices']
    assert 'B12X_ATTN' not in specs['attention_backend']['choices']


def test_no_backend_selection_preserves_old_behavior():
    assert backends.check({'extra_args': '"bad quote'}) == {}
    assert not backends.selected({})
    assert backends.selected({'extra_args': '--linear-backend=b12x'})


@pytest.mark.parametrize('raw', ['--dtype half', '--dtype=half'])
def test_raw_dtype_override_cannot_bypass_attention_validation(raw):
    with pytest.raises(ValueError, match='BF16'):
        check({'attention_backend': 'B12X', 'dtype': 'bfloat16', 'extra_args': raw})


def test_trailing_backend_override_can_remove_b12x_restrictions():
    check({'attention_backend': 'B12X', 'dtype': 'half',
           'extra_args': '--attention-backend FLASHINFER'})
    check({'moe_backend': 'b12x', 'enable_expert_parallel': True,
           'extra_args': '--moe-backend auto'})


@pytest.mark.parametrize('raw', ['--attention-backend "', '--attention-backend',
                                '--attention-backend=', '--attention-backend --dtype bf16'])
def test_malformed_raw_backend_arguments_rejected(raw):
    with pytest.raises(ValueError):
        check({'extra_args': raw})


@pytest.mark.parametrize('raw', ['-dcp 2', '--decode-context-parallel-size=2',
                                '-pcp 2', '--prefill-context-parallel-size 2'])
def test_context_parallel_aliases_rejected(raw):
    with pytest.raises(ValueError, match='context parallelism'):
        check({'attention_backend': 'B12X', 'extra_args': raw})


@pytest.mark.parametrize('raw', ['-ep', '--enable-expert-parallel'])
def test_expert_parallel_aliases_and_later_disable(raw):
    with pytest.raises(ValueError, match='expert parallelism'):
        check({'moe_backend': 'b12x', 'extra_args': raw})
    check({'moe_backend': 'b12x', 'extra_args': raw + ' --no-enable-expert-parallel'})


@pytest.mark.parametrize('dtype', ['half', 'float16', 'float', 'float32'])
def test_b12x_attention_rejects_explicit_incompatible_dtype(dtype):
    with pytest.raises(ValueError, match='BF16'):
        check({'attention_backend': 'B12X', 'dtype': dtype})


def test_auto_dtype_uses_nested_metadata_or_reports_unknown():
    with pytest.raises(ValueError, match='BF16'):
        check({'attention_backend': 'B12X'}, model={'config': {'torch_dtype': 'bfloat16', 'text_config': {'dtype': 'float16'}}})
    warning = check({'attention_backend': 'B12X'}, model={'config': {}})
    assert 'dtype is unknown' in warning['attention_backend']['message']
    check({'attention_backend': 'B12X'}, model={'config': {'text_config': {'dtype': 'bfloat16'}}})


@pytest.mark.parametrize('cache', ['nvfp4', 'fp8_e5m2', 'float16'])
def test_b12x_attention_rejects_unsupported_cache(cache):
    with pytest.raises(ValueError, match='cache'):
        check({'attention_backend': 'B12X', 'kv_cache_dtype': cache})


@pytest.mark.parametrize('cache', ['auto', 'fp8', 'fp8_e4m3', 'bfloat16'])
def test_b12x_attention_accepts_supported_cache(cache):
    check({'attention_backend': 'B12X', 'kv_cache_dtype': cache})


def test_mla_metadata_rejected():
    with pytest.raises(ValueError, match='latent attention'):
        check({'attention_backend': 'B12X'}, model={'config': {'dtype': 'bfloat16', 'kv_lora_rank': 512}})


@pytest.mark.parametrize('method', ['awq', 'gptq', 'bitsandbytes', 'fp8'])
def test_moe_known_incompatible_quantization_rejected(method):
    with pytest.raises(ValueError, match='expert weights'):
        check({'moe_backend': 'b12x'}, model={'config': {'quantization_config': {'quant_method': method}}})


def test_moe_compressed_tensors_not_assumed_incompatible():
    result = check({'moe_backend': 'b12x'}, model={'config': {'quantization_config': {'quant_method': 'compressed-tensors'}}})
    assert 'NVFP4 or MXFP4' in result['moe_backend']['message']


def test_missing_package_rejects_b12x_but_not_flashinfer_b12x():
    caps = {**CAPS, 'b12x': False}
    for key, value in [('linear_backend', 'b12x'), ('moe_backend', 'b12x'), ('attention_backend', 'B12X')]:
        with pytest.raises(ValueError, match='optional b12x package'):
            check({key: value}, capabilities=caps)
    check({'linear_backend': 'flashinfer_b12x'}, capabilities=caps)
    check({'moe_backend': 'flashinfer_b12x'}, capabilities=caps)


def test_known_old_runtime_rejects_flag_but_unknown_allows_with_warning():
    with pytest.raises(ValueError, match='does not support --linear-backend'):
        check({'linear_backend': 'b12x'}, capabilities={**CAPS, 'flags': []})
    result = check({'linear_backend': 'b12x'}, capabilities={'flags': None, 'b12x': None, 'source': 'docker', 'message': 'Docker runtime is unverified.'})
    assert 'Docker runtime is unverified' in result['linear_backend']['message']
    assert 'package availability is unverified' in result['linear_backend']['message']


def test_mixed_gpus_check_only_selected_devices():
    hw = {'gpus': [{'index': 0, 'compute_capability': '8.9'}, {'index': 1, 'compute_capability': '12.1'}]}
    result = check({'linear_backend': 'b12x'}, hardware=hw)
    assert 'mixed architectures' in result['linear_backend']['message']
    check({'linear_backend': 'b12x', 'device_ids': '1'}, hardware=hw)
    with pytest.raises(ValueError, match='SM120/SM121'):
        check({'linear_backend': 'b12x', 'device_ids': '0'}, hardware=hw)
    warning = check({'linear_backend': 'b12x', 'device_ids': 'GPU-uuid'}, hardware=hw)
    assert 'identities could not be verified' in warning['linear_backend']['message']


@pytest.mark.parametrize('hw', [{}, {'gpus': []}, {'gpus': [{'index': 0}]}])
def test_missing_gpu_evidence_is_unknown(hw):
    result = check({'linear_backend': 'b12x'}, hardware=hw)
    assert 'could not be verified' in result['linear_backend']['message']


def test_release_note_attention_spelling_has_actionable_error():
    with pytest.raises(ValueError, match='Select B12X'):
        check({'attention_backend': 'B12X_ATTN'})


def test_runtime_choices_reject_unsupported_backend_value():
    with pytest.raises(ValueError, match='does not accept b12x'):
        check({'linear_backend': 'b12x'}, capabilities={**CAPS, 'choices': {'--linear-backend': ['auto', 'marlin']}})
    check({'linear_backend': 'b12x'}, capabilities={**CAPS, 'choices': {}})


@pytest.mark.parametrize('value', [0, -1, 1.5, True, 'invalid'])
def test_invalid_context_parallel_value_does_not_truncate_or_default(value):
    with pytest.raises(ValueError, match='positive whole number'):
        check({'attention_backend': 'B12X', 'decode_context_parallel_size': value})


@pytest.mark.parametrize('capability', ['', 'N/A', None])
def test_unknown_gpu_capability_does_not_become_definite_rejection(capability):
    result = check({'linear_backend': 'b12x'}, hardware={'gpus': [{'index': 0, 'compute_capability': capability}]})
    assert 'could not be verified' in result['linear_backend']['message']


def test_native_inherited_cuda_mask_is_honored_and_explicit_selection_wins():
    hw = {'gpus': [{'index': 0, 'compute_capability': '8.9'}, {'index': 1, 'compute_capability': '12.0'}], 'cuda_visible_devices': '1'}
    result = check({'linear_backend': 'b12x'}, hardware=hw)
    assert 'mixed architectures' not in result['linear_backend']['message']
    with pytest.raises(ValueError, match='SM120/SM121'):
        check({'linear_backend': 'b12x', 'device_ids': '0'}, hardware=hw)
    result = check({'linear_backend': 'b12x'}, hardware={**hw, 'cuda_visible_devices': ''})
    assert 'identities could not be verified' in result['linear_backend']['message']


def test_explicit_mixed_pool_warns_when_participating_subset_unknown():
    hw = {'gpus': [{'index': 0, 'compute_capability': '8.9'}, {'index': 1, 'compute_capability': '12.0'}]}
    result = check({'linear_backend': 'b12x', 'device_ids': '0,1', 'tensor_parallel_size': 1}, hardware=hw)
    assert 'mixed architectures' in result['linear_backend']['message']


@pytest.mark.parametrize('inherited', [False, True])
@pytest.mark.parametrize('raw', [None, '--tensor-parallel-size=2', '-tp 2'])
def test_mixed_pool_fully_used_by_tensor_parallelism_rejected(inherited, raw):
    hw = {'gpus': [{'index': 0, 'compute_capability': '8.9'}, {'index': 1, 'compute_capability': '12.0'}]}
    cfg = {'linear_backend': 'b12x'}
    if inherited:
        hw['cuda_visible_devices'] = '0,1'
    else:
        cfg['device_ids'] = '0,1'
    if raw:
        cfg['tensor_parallel_size'] = 1
        cfg['extra_args'] = raw
    else:
        cfg['tensor_parallel_size'] = 2
    with pytest.raises(ValueError, match='tensor parallel group includes an unsupported GPU'):
        check(cfg, hardware=hw)


def test_raw_tensor_parallel_override_can_leave_subset_uncertain():
    hw = {'gpus': [{'index': 0, 'compute_capability': '8.9'}, {'index': 1, 'compute_capability': '12.0'}]}
    result = check({'linear_backend': 'b12x', 'device_ids': '0,1',
                    'tensor_parallel_size': 2, 'extra_args': '-tp 1'}, hardware=hw)
    assert 'mixed architectures' in result['linear_backend']['message']
