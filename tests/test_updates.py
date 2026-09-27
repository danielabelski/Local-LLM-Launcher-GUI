from pathlib import Path
import subprocess

import pytest
from local_llm_launcher.config import Settings
from local_llm_launcher.updates import UpdateManager

SHA = 'a' * 40

def ready(manager, monkeypatch):
    monkeypatch.setattr(manager, '_requirements', lambda engine: ([], 'cpu', 2))
    monkeypatch.setattr(manager, '_capture', lambda argv, **kw: SHA + '\trefs/heads/master')
    return manager.check('llamacpp')

def test_failed_build_keeps_previous_path(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    settings.update({'llamacpp_path': '/old/llama-server'})
    manager = UpdateManager(settings)
    check = ready(manager, monkeypatch)
    monkeypatch.setattr(manager, '_run', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('compile failed')))
    manager.start(check['check_id'])
    manager.thread.join(2)
    assert manager.status()['state'] == 'failed'
    assert settings.data['llamacpp_path'] == '/old/llama-server'

def test_success_builds_checked_sha_and_activates_only_after_validation(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    manager = UpdateManager(settings)
    check = ready(manager, monkeypatch)
    calls = []
    def run(argv, cwd, env):
        calls.append(argv)
        if '--build' in argv:
            binary = cwd / 'build/bin/llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
    monkeypatch.setattr(manager, '_run', run)
    manager.start(check['check_id'])
    manager.thread.join(2)
    assert manager.status()['state'] == 'complete'
    assert any(SHA in argv and 'fetch' in argv for argv in calls)
    assert calls[-1][-1] == '--version'
    assert settings.data['llamacpp_path'].endswith('/build/bin/llama-server')
    assert Path(settings.data['llamacpp_path']).is_file()

def test_rejects_unknown_check_and_concurrent_job(tmp_path, monkeypatch):
    manager = UpdateManager(Settings(tmp_path))
    with pytest.raises(ValueError):
        manager.start('untrusted')
    check = ready(manager, monkeypatch)
    manager.job = {'state': 'running'}
    with pytest.raises(RuntimeError):
        manager.start(check['check_id'])

def test_unsupported_platform(tmp_path, monkeypatch):
    monkeypatch.setattr('local_llm_launcher.updates.platform.system', lambda: 'Darwin')
    result = UpdateManager(Settings(tmp_path)).check('llamacpp')
    assert not result['supported']
    assert 'Linux' in ' '.join(result['reasons'])

@pytest.mark.parametrize('method,path,body', [
    ('GET', '/api/updates', None),
    ('POST', '/api/updates/check/llamacpp', None),
    ('POST', '/api/updates', {'check_id': 'bad'}),
])
@pytest.mark.parametrize('address,headers', [
    ('192.168.1.20', {'host': 'localhost'}),
    ('127.0.0.1', {'host': 'localhost', 'x-forwarded-for': '127.0.0.1'}),
])
def test_update_routes_reject_remote_or_forwarded_client(method, path, body, address, headers):
    from fastapi.testclient import TestClient
    from local_llm_launcher.app import create_app
    with TestClient(create_app(), client=(address, 1234)) as client:
        assert client.request(method, path, json=body, headers=headers).status_code == 403


def test_local_status_and_check_errors(monkeypatch):
    from fastapi.testclient import TestClient
    from local_llm_launcher.app import create_app
    from local_llm_launcher import api
    with TestClient(create_app(), base_url='http://localhost', client=('127.0.0.1', 1234)) as client:
        assert client.get('/api/updates').json()['state'] == 'idle'
        assert client.post('/api/updates/check/arbitrary-command').status_code == 400
        assert client.post('/api/updates', json={'check_id': 'bad'}).status_code == 400


def test_managed_vllm_build_and_future_launch(tmp_path, monkeypatch):
    from local_llm_launcher.registry import ServerManager
    settings = Settings(tmp_path)
    settings.update({'vllm_path': '/previous/vllm'})
    manager = UpdateManager(settings)
    monkeypatch.setattr(manager, '_requirements', lambda engine: ([], 'cuda', 2))
    monkeypatch.setattr(manager, '_capture', lambda *a, **kw: SHA + '\trefs/heads/main')
    check = manager.check('vllm')
    calls = []
    def run(argv, cwd, env):
        calls.append(argv)
        assert env['MAX_JOBS'] == '2'
        if 'install' in argv:
            binary = cwd.parent / 'venv/bin/vllm'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
            assert settings.data['vllm_path'] == '/previous/vllm'
    monkeypatch.setattr(manager, '_run', run)
    manager.start(check['check_id'])
    manager.thread.join(2)
    assert manager.status()['state'] == 'complete'
    binary = settings.data['vllm_path']
    spec = ServerManager(tmp_path).build_spec('vllm-native', {'repo_id': 'test/model'}, {}, vllm_binary=binary)
    assert spec['argv'][:3] == [binary, 'serve', 'test/model']
    assert any('--torch-backend=auto' in argv and '-e' in argv for argv in calls)


def test_managed_vllm_hardware_and_api_wiring(tmp_path, monkeypatch):
    from local_llm_launcher import hardware, api
    binary = tmp_path / 'vllm'
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    monkeypatch.setattr(hardware, '_detect_nvidia', lambda: [])
    monkeypatch.setattr(hardware, '_vllm_docker_available', lambda: False)
    assert hardware.detect_hardware(vllm_hint=str(binary)).engines.vllm_native
    api.settings.update({'vllm_path': str(binary)})
    monkeypatch.setattr(api, 'find_model', lambda repo: {'repo_id': repo})
    received = {}
    class Server:
        def status(self): return {'running': True}
    def launch(*args, **kwargs):
        received.update(kwargs)
        return Server()
    monkeypatch.setattr(api.servers, 'launch', launch)
    api.api_launch(api.LaunchRequest(engine_mode='vllm-native', repo_id='test/model'))
    assert received['vllm_binary'] == str(binary)


def test_preflight_reports_toolchain_gpu_and_disk_problems(tmp_path, monkeypatch):
    from types import SimpleNamespace
    manager = UpdateManager(Settings(tmp_path))
    monkeypatch.setattr('local_llm_launcher.updates.platform.system', lambda: 'Linux')
    monkeypatch.setattr('local_llm_launcher.updates.shutil.which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr('local_llm_launcher.updates.shutil.disk_usage', lambda path: SimpleNamespace(free=1))
    monkeypatch.setattr('local_llm_launcher.updates.os.cpu_count', lambda: 192)
    monkeypatch.setattr('local_llm_launcher.updates.psutil.virtual_memory', lambda: SimpleNamespace(available=8 * 1024**3))
    def capture(argv):
        if argv[0] == 'cmake': return 'cmake version 3.28.1'
        if argv[0] == 'g++': return '10.2.0'
        if '--query-gpu=compute_cap' in argv: return '6.1'
        return 'GPU'
    monkeypatch.setattr(manager, '_capture', capture)
    result = manager.check('vllm')
    assert not result['supported']
    assert result['jobs'] == 2
    assert '11.3' in ' '.join(result['reasons'])
    assert '7.5' in ' '.join(result['reasons'])
    assert '30 GiB' in ' '.join(result['reasons'])


def test_checked_revision_already_active(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    settings.update({'llamacpp_path': str(tmp_path / 'engines' / f'{SHA}-12345678' / 'llama-server')})
    manager = UpdateManager(settings)
    check = ready(manager, monkeypatch)
    assert not check['supported']
    assert check['current_revision'] == SHA


def test_changed_managed_path_invalidates_hardware_cache(monkeypatch):
    from types import SimpleNamespace
    from local_llm_launcher import api
    calls = []
    def detect(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {'engines': {'llamacpp_path': kwargs['llamacpp_hint']}})
    monkeypatch.setattr(api.hardware, 'detect_hardware', detect)
    api.settings.update({'llamacpp_path': '/old/llama-server'})
    assert api.get_hardware()['engines']['llamacpp_path'] == '/old/llama-server'
    api.settings.update({'llamacpp_path': '/new/llama-server'})
    assert api.get_hardware()['engines']['llamacpp_path'] == '/new/llama-server'
    assert len(calls) == 2


def test_start_rechecks_resources_and_hardware(tmp_path, monkeypatch):
    manager = UpdateManager(Settings(tmp_path))
    check = ready(manager, monkeypatch)
    monkeypatch.setattr(manager, '_requirements', lambda engine: ([], 'cuda', 1))
    with pytest.raises(ValueError, match='hardware changed'):
        manager.start(check['check_id'])
    monkeypatch.setattr(manager, '_requirements', lambda engine: ([], 'cpu', 1))
    seen = []
    monkeypatch.setattr(manager, '_build', lambda selected, path: seen.append(selected['jobs']))
    manager.start(check['check_id'])
    manager.thread.join(2)
    assert seen == [1]


@pytest.mark.parametrize('engine,key', [('llamacpp', 'llamacpp_path'), ('vllm', 'vllm_path')])
def test_failed_build_removes_only_inactive_artifacts_and_retains_log(tmp_path, monkeypatch, engine, key):
    previous = tmp_path / 'engines' / engine / 'previous' / 'bin' / 'engine'
    previous.parent.mkdir(parents=True)
    previous.write_text('previous working engine')
    settings = Settings(tmp_path)
    settings.update({key: str(previous)})
    manager = UpdateManager(settings)
    monkeypatch.setattr(manager, '_requirements', lambda engine: ([], 'cpu', 2))
    monkeypatch.setattr(manager, '_capture', lambda *a, **kw: SHA + '\trefs/heads/main')
    check = manager.check(engine)
    def fail(argv, cwd, env):
        (cwd / 'build').mkdir()
        (cwd / 'build' / 'large-artifact').write_text('compiler output')
        (cwd.parent / 'venv').mkdir()
        (cwd.parent / 'venv' / 'large-package').write_text('package output')
        (cwd.parent / 'build.log').write_text('diagnostic: compiler failed')
        raise RuntimeError('compile failed')
    monkeypatch.setattr(manager, '_run', fail)
    manager.start(check['check_id'])
    manager.thread.join(2)
    status = manager.status()
    assert status['state'] == 'failed'
    assert 'compile failed' in status['error']
    assert status['log'] == 'diagnostic: compiler failed'
    build = Path(status['path'])
    assert not (build / 'source').exists()
    assert not (build / 'venv').exists()
    assert previous.read_text() == 'previous working engine'
    assert settings.data[key] == str(previous)


def test_timed_out_build_retains_artifacts_that_children_may_be_using(tmp_path, monkeypatch):
    manager = UpdateManager(Settings(tmp_path))
    check = ready(manager, monkeypatch)
    def timeout(argv, cwd, env):
        (cwd / 'compiler-output').write_text('possibly still in use')
        raise subprocess.TimeoutExpired(argv, 7200)
    monkeypatch.setattr(manager, '_run', timeout)
    manager.start(check['check_id'])
    manager.thread.join(2)
    status = manager.status()
    assert status['state'] == 'failed'
    assert 'child build processes may still be running' in status['error']
    assert (Path(status['path']) / 'source' / 'compiler-output').is_file()


def test_failed_cleanup_preserves_selected_build(tmp_path):
    settings = Settings(tmp_path)
    manager = UpdateManager(settings)
    build = tmp_path / 'engines' / 'selected'
    binary = build / 'source' / 'build' / 'bin' / 'llama-server'
    binary.parent.mkdir(parents=True)
    binary.write_text('selected installation')
    settings.update({'llamacpp_path': str(binary)})
    manager._clean_failed_build(build)
    assert binary.read_text() == 'selected installation'
