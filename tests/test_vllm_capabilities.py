"""Probe real isolated Python scripts without vLLM, CUDA, or Docker."""
from pathlib import Path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    caps._cache.clear()
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


def test_package_install_seen_immediately_and_evidence_is_copied(runtime):
    binary, site = runtime
    first = caps.probe('vllm-native', str(binary))
    assert first['b12x'] is False
    (site / 'b12x.py').write_text('raise RuntimeError("must not import b12x")\n')
    # Installing into the runtime's import directories invalidates cached evidence.
    assert caps.probe('vllm-native', str(binary))['b12x'] is True
    first['flags'].append('--fake')
    assert '--fake' not in caps.probe('vllm-native', str(binary))['flags']


def test_cache_survives_advice_polling_and_expires_after_ttl(runtime, monkeypatch):
    binary, _ = runtime
    now = [1000.0]
    monkeypatch.setattr(caps.time, 'monotonic', lambda: now[0])
    real_run = caps.subprocess.run
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return real_run(argv, **kwargs)
    monkeypatch.setattr(caps.subprocess, 'run', run)
    caps.probe('vllm-native', str(binary))
    assert len(calls) == 2
    # The Launch page polls advice every 8 seconds; that must not re-run the heavy probe.
    now[0] += 5 * 60
    caps.probe('vllm-native', str(binary))
    assert len(calls) == 2
    now[0] += caps._TTL
    caps.probe('vllm-native', str(binary))
    assert len(calls) == 4


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
    caps._cache.clear()
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
        assert 0 < kwargs['timeout'] <= 120
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


def _gated_run(monkeypatch):
    """Hold every probe subprocess until released; record what ran."""
    real_run = caps.subprocess.run
    calls, gate = [], threading.Event()
    def run(argv, **kwargs):
        calls.append((list(argv), kwargs['timeout']))
        assert gate.wait(5)
        return real_run(argv, **kwargs)
    monkeypatch.setattr(caps.subprocess, 'run', run)
    return calls, gate


def _eventually(predicate):
    deadline = time.perf_counter() + 5
    while not predicate():
        assert time.perf_counter() < deadline, 'background probe did not finish'
        time.sleep(0.01)


def test_help_probe_allows_slow_vllm_startup(runtime, monkeypatch):
    binary, _ = runtime
    calls, gate = _gated_run(monkeypatch)
    gate.set()
    caps.probe('vllm-native', str(binary))
    help_timeout = next(timeout for argv, timeout in calls if 'serve' in argv)
    # `vllm serve --help=all` imports torch and vLLM; 10-30s is normal on a cold cache.
    assert help_timeout >= 30


def test_concurrent_probes_share_one_subprocess(runtime, monkeypatch):
    binary, _ = runtime
    calls, gate = _gated_run(monkeypatch)
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(caps.probe, 'vllm-native', str(binary)) for _ in range(4)]
        deadline = time.perf_counter() + 1
        while len(calls) < 2 and time.perf_counter() < deadline:
            time.sleep(0.01)
        gate.set()
        results = [future.result(10) for future in futures]
    assert sum('serve' in argv for argv, _ in calls) == 1
    assert all(result == results[0] for result in results)


def test_nonblocking_probe_without_evidence_reports_checking(runtime, monkeypatch):
    binary, _ = runtime
    calls, gate = _gated_run(monkeypatch)
    result = caps.probe('vllm-native', str(binary), wait=False)
    assert result['flags'] is None and result['b12x'] is None
    assert 'being checked' in result['message']
    gate.set()
    _eventually(lambda: caps.probe('vllm-native', str(binary), wait=False)['flags'] is not None)


def test_nonblocking_probe_serves_previous_evidence_while_refreshing(runtime, monkeypatch):
    binary, site = runtime
    now = [120.0]
    monkeypatch.setattr(caps.time, 'monotonic', lambda: now[0])
    assert caps.probe('vllm-native', str(binary))['b12x'] is False
    (site / 'b12x.py').write_text('raise RuntimeError("must not import b12x")\n')
    now[0] += 60
    calls, gate = _gated_run(monkeypatch)
    assert caps.probe('vllm-native', str(binary), wait=False)['b12x'] is False
    gate.set()
    _eventually(lambda: caps.probe('vllm-native', str(binary), wait=False)['b12x'] is True)
    assert sum('serve' in argv for argv, _ in calls) == 1


def test_slow_probe_of_one_runtime_does_not_delay_another(runtime, monkeypatch):
    binary, _ = runtime
    other = binary.with_name('vllm-other')
    other.write_text(binary.read_text())
    other.chmod(0o755)
    real_run = caps.subprocess.run
    gate, started = threading.Event(), threading.Event()
    def run(argv, **kwargs):
        if str(binary) in argv:
            started.set()
            assert gate.wait(5)
        return real_run(argv, **kwargs)
    monkeypatch.setattr(caps.subprocess, 'run', run)
    try:
        caps.probe('vllm-native', str(binary), wait=False)
        assert started.wait(5)
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(caps.probe, 'vllm-native', str(other)).result(3)
        assert result['flags'] is not None
    finally:
        gate.set()


def test_failed_background_start_can_be_retried(runtime, monkeypatch):
    binary, _ = runtime
    class Unstartable:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            raise RuntimeError("can't start new thread")
    with monkeypatch.context() as patch:
        patch.setattr(caps.threading, 'Thread', Unstartable)
        # Polled advice degrades to unknown evidence instead of failing the request.
        assert caps.probe('vllm-native', str(binary), wait=False)['flags'] is None
    _eventually(lambda: caps.probe('vllm-native', str(binary), wait=False)['flags'] is not None)


def test_install_during_help_probe_is_noticed(runtime, monkeypatch):
    binary, site = runtime
    real_run = caps.subprocess.run
    helps = []
    def run(argv, **kwargs):
        result = real_run(argv, **kwargs)
        if 'serve' in argv:
            helps.append(argv)
            if len(helps) == 1:
                # An upgrade lands while the (slow) help probe is still running.
                (site / 'upgraded.py').write_text('')
        return result
    monkeypatch.setattr(caps.subprocess, 'run', run)
    caps.probe('vllm-native', str(binary))
    caps.probe('vllm-native', str(binary))
    assert len(helps) == 2
