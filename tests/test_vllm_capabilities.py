"""Probe real isolated Python scripts without vLLM, CUDA, or Docker."""
from pathlib import Path
import subprocess
from types import SimpleNamespace
import venv

import pytest

from local_llm_launcher.engines import vllm_capabilities as caps


@pytest.fixture
def runtime(tmp_path):
    environment = tmp_path / 'engine'
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / 'bin/python'
    site = Path(subprocess.check_output([str(python), '-I', '-c',
                                        'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    metadata = site / 'vllm-0.30.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: 0.30.0\n')
    package = site / 'vllm'
    package.mkdir()
    (package / '__init__.py').write_text('raise RuntimeError("vllm must not be imported")\n')
    binary = environment / 'bin/vllm'
    binary.write_text(f'#!{python}\nprint("usage: vllm serve --help --dtype --linear-backend --attention-backend")\n')
    binary.chmod(0o755)
    caps._native.cache_clear()
    return binary, site


def test_selected_environment_not_launcher_and_no_vllm_import(runtime, monkeypatch):
    binary, site = runtime
    monkeypatch.setenv('PYTHONPATH', '/does/not/exist')
    result = caps.probe('vllm-native', str(binary))
    assert result['version'] == '0.30.0'
    assert result['b12x'] is False
    assert result['flags'] == ['--attention-backend', '--dtype', '--help', '--linear-backend']
    assert result['choices'] == {}
    assert result['source'] == str(binary)


def test_package_change_seen_after_cache_expiry(runtime, monkeypatch):
    binary, site = runtime
    now = [120.0]
    monkeypatch.setattr(caps.time, 'monotonic', lambda: now[0])
    first = caps.probe('vllm-native', str(binary))
    (site / 'b12x.py').write_text('raise RuntimeError("must not import b12x")\n')
    assert caps.probe('vllm-native', str(binary))['b12x'] is False
    now[0] += 60
    assert caps.probe('vllm-native', str(binary))['b12x'] is True
    first['flags'].append('--fake')
    assert '--fake' not in caps.probe('vllm-native', str(binary))['flags']


def test_pythonpath_package_matches_real_console_and_refreshes_cache(runtime, tmp_path, monkeypatch):
    binary, _ = runtime
    monkeypatch.delenv('PYTHONPATH', raising=False)
    assert caps.probe('vllm-native', str(binary))['b12x'] is False
    extra = tmp_path / 'extra-packages'
    extra.mkdir()
    (extra / 'b12x.py').write_text('raise RuntimeError("package must not be imported")\n')
    monkeypatch.setenv('PYTHONPATH', str(extra))
    assert caps.probe('vllm-native', str(binary))['b12x'] is True
    monkeypatch.delenv('PYTHONPATH')
    assert caps.probe('vllm-native', str(binary))['b12x'] is False


def test_metadata_uses_script_directory_not_working_directory(runtime, tmp_path, monkeypatch):
    binary, _ = runtime
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'b12x.py').write_text('raise RuntimeError("cwd package is not visible to script")\n')
    assert caps.probe('vllm-native', str(binary))['b12x'] is False
    (binary.parent / 'b12x.py').write_text('raise RuntimeError("must not import script package")\n')
    caps._native.cache_clear()
    assert caps.probe('vllm-native', str(binary))['b12x'] is True


def test_cache_and_binary_identity_refresh(runtime, monkeypatch):
    binary, _ = runtime
    real_run = caps.subprocess.run
    calls = []
    def run(*args, **kwargs):
        calls.append(args[0])
        return real_run(*args, **kwargs)
    monkeypatch.setattr(caps.subprocess, 'run', run)
    caps.probe('vllm-native', str(binary))
    caps.probe('vllm-native', str(binary))
    assert len(calls) == 2
    binary.write_text(binary.read_text().replace('--linear-backend', '--old-option'))
    assert '--linear-backend' not in caps.probe('vllm-native', str(binary))['flags']
    assert len(calls) == 4


def test_env_python_uses_path_runtime(runtime, monkeypatch):
    binary, _ = runtime
    binary.write_text('#!/usr/bin/env python\nprint("--help --linear-backend")\n')
    monkeypatch.setenv('PATH', str(binary.parent) + ':/usr/bin')
    result = caps.probe('vllm-native', str(binary))
    assert result['version'] == '0.30.0' and result['b12x'] is False


def test_shell_wrapper_has_help_but_unknown_package(tmp_path):
    binary = tmp_path / 'vllm'
    binary.write_text('#!/bin/sh\necho "--help --model --old-option"\n')
    binary.chmod(0o755)
    result = caps.probe('vllm-native', str(binary))
    assert result['flags'] == ['--help', '--model', '--old-option']
    assert result['version'] is None and result['b12x'] is None


def test_wrapped_argparse_choices(runtime):
    binary, _ = runtime
    help_text = '''usage: vllm serve [options]
  --max-model-len MAX_MODEL_LEN
  --linear-backend  {auto, flashinfer_cutlass,
                    b12x, marlin}
  --moe-backend MOE_BACKEND
  --attention-backend
      {FLASH_ATTN, FLASHINFER, B12X}
'''
    shebang = binary.read_text().splitlines()[0]
    binary.write_text(shebang + '\nprint(' + repr(help_text) + ')\n')
    result = caps.probe('vllm-native', str(binary))
    assert result['choices'] == {
        '--linear-backend': ['auto', 'flashinfer_cutlass', 'b12x', 'marlin'],
        '--attention-backend': ['FLASH_ATTN', 'FLASHINFER', 'B12X'],
    }
    result['choices']['--attention-backend'].append('FAKE')
    assert 'FAKE' not in caps.probe('vllm-native', str(binary))['choices']['--attention-backend']


@pytest.mark.parametrize('help_text', [
    'usage: wrapper --help --version',
    '--attention-backend {FLASH_ATTN,FLASHINFER}',
    'Run vllm serve for model options.',
])
def test_partial_help_is_not_complete_flag_evidence(runtime, help_text):
    binary, _ = runtime
    shebang = binary.read_text().splitlines()[0]
    binary.write_text(shebang + '\nprint(' + repr(help_text) + ')\n')
    result = caps.probe('vllm-native', str(binary))
    assert result['flags'] is None and result['choices'] == {}
    assert result['version'] == '0.30.0'


@pytest.mark.parametrize('failure', ['timeout', 'nonzero', 'malformed'])
def test_failed_probes_remain_unknown(runtime, monkeypatch, failure):
    binary, _ = runtime
    def run(argv, **kwargs):
        assert kwargs['timeout'] <= 8
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return SimpleNamespace(returncode=1 if failure == 'nonzero' else 0,
                               stdout='not valid output', stderr='')
    monkeypatch.setattr(caps.subprocess, 'run', run)
    result = caps.probe('vllm-native', str(binary))
    assert result['flags'] is None and result['version'] is None and result['b12x'] is None
    assert 'could not be verified' in result['message']


def test_missing_binary_unknown(tmp_path):
    result = caps.probe('vllm-native', str(tmp_path / 'missing'))
    assert result['flags'] is None and result['version'] is None and result['b12x'] is None


def test_docker_never_probes_host(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Docker must not invoke host processes or inspect native Python')
    monkeypatch.setattr(caps.subprocess, 'run', unexpected)
    monkeypatch.setattr(caps.shutil, 'which', unexpected)
    result = caps.probe('vllm-docker')
    assert result['source'] == 'docker'
    assert result['flags'] is None and result['version'] is None and result['b12x'] is None
