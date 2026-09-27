"""Tests for the server manager and its persistence."""
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from local_llm_launcher.registry import ServerManager, port_in_use

GGUF = {
    "repo_id": "org/m-GGUF", "path": "/x/m-Q4.gguf", "format": "gguf",
    "size_bytes": 1, "source": "folder", "quant": "Q4_K_M", "config": {},
    "gguf_files": [{"filename": "m-Q4.gguf", "path": "/x/m-Q4.gguf", "size_bytes": 1, "quant": "Q4_K_M"}],
    "param_count_b": 8.0,
}


def test_launch_records_and_persists(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    # Swap the built command for a harmless long-running process.
    monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
        "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
        "env": {}, "port": 45123,
    })
    srv = mgr.launch("llamacpp", GGUF, {})
    try:
        assert srv.is_running()
        assert len(mgr.list()) == 1

        # A new manager instance (fresh GUI start) sees the same server.
        mgr2 = ServerManager(app_dir=tmp_path)
        assert len(mgr2.list()) == 1
        srv2 = mgr2.get(srv.server_id)
        assert srv2.is_running()
        assert srv2.pid == srv.pid
    finally:
        assert mgr.stop(srv.server_id)
    assert not srv.is_running()


def test_launch_auto_increments_port(tmp_path):
    import socket
    from local_llm_launcher.registry import find_free_port
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    next_port = find_free_port(port + 1)
    try:
        assert port_in_use(port)
        mgr = ServerManager(app_dir=tmp_path)
        # Verify find_free_port skips the blocked port
        assert next_port > port
        assert not port_in_use(next_port)
    finally:
        blocker.close()


def test_launch_resolves_port_before_build_spec(tmp_path, monkeypatch):
    import socket

    from local_llm_launcher.registry import find_free_port

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    occupied = blocker.getsockname()[1]
    next_port = find_free_port(occupied + 1)
    try:
        mgr = ServerManager(app_dir=tmp_path)
        seen_port = {}

        def fake_build(engine_mode, model, config, llamacpp_binary=None, vllm_binary=None):
            # Mirrors a real builder: the config port becomes the argv port.
            seen_port["config"] = config["port"]
            return {
                "argv": [sys.executable, "-c", "import time; time.sleep(60)",
                         "--port", str(config["port"])],
                "env": {},
                "port": config["port"],
            }

        monkeypatch.setattr(mgr, "build_spec", fake_build)
        srv = mgr.launch("llamacpp", GGUF, {"port": occupied})
        try:
            # The resolved port must reach the builder AND the argv.
            assert seen_port["config"] == next_port
            assert srv.port == next_port
            assert srv.argv[srv.argv.index("--port") + 1] == str(next_port)
        finally:
            assert mgr.stop(srv.server_id)
    finally:
        blocker.close()


def test_launch_reserves_port_before_server_listens(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "build_spec", lambda _mode, _model, config, _binary, _vllm_binary: {
        "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
        "env": {}, "port": config["port"],
    })
    first = mgr.launch("llamacpp", GGUF, {"port": 45126})
    try:
        second = mgr.launch("llamacpp", GGUF, {"port": 45126})
        try:
            assert second.port == first.port + 1
        finally:
            mgr.stop(second.server_id)
    finally:
        mgr.stop(first.server_id)


def test_remove_dead_server(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
        "argv": [sys.executable, "-c", "pass"], "env": {}, "port": 45124,
    })
    srv = mgr.launch("llamacpp", GGUF, {})
    srv.process.wait(timeout=10)
    assert not srv.is_running()
    assert mgr.remove(srv.server_id)
    assert mgr.list() == []


def test_failed_launch_keeps_record_and_log(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    monkeypatch.setattr(mgr, "build_spec", lambda *a, **k: {
        "argv": [str(tmp_path / "missing-llama-server")], "env": {}, "port": 45125,
    })
    with pytest.raises(RuntimeError, match="Check the logs"):
        mgr.launch("llamacpp", GGUF, {})

    [status] = mgr.list()
    assert status["running"] is False
    restored = ServerManager(app_dir=tmp_path).get(status["id"])
    assert restored is not None
    assert any("failed to start" in line for line in restored.tail_logs())


def test_concurrent_saves_do_not_share_temporary_file(tmp_path, monkeypatch):
    from pathlib import Path

    mgr = ServerManager(app_dir=tmp_path)
    first_writing = Event()
    release_first = Event()
    second_writing = Event()
    original = Path.write_text
    calls = 0

    def paused_write(path, *args, **kwargs):
        nonlocal calls
        if path == mgr.state_file.with_suffix(".tmp"):
            calls += 1
            if calls == 1:
                first_writing.set()
                assert release_first.wait(timeout=5)
            else:
                second_writing.set()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", paused_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(mgr._save)
        try:
            assert first_writing.wait(timeout=5)
            second = pool.submit(mgr._save)
            assert not second_writing.wait(timeout=0.2)
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)


def _registered_server(mgr, monkeypatch):
    from local_llm_launcher.engines.base import LocalServer

    srv = LocalServer(server_id="blocked", engine="llamacpp", model_label="test",
                      port=45129, argv=[], env={}, log_dir=mgr.log_dir)
    monkeypatch.setattr(srv, "is_running", lambda: True)
    mgr.servers[srv.server_id] = srv
    return srv


@pytest.mark.parametrize("operation", ["stop", "remove", "stop_all"])
def test_shutdown_leaves_queries_and_launch_responsive(tmp_path, monkeypatch, operation):
    mgr = ServerManager(app_dir=tmp_path)
    srv = _registered_server(mgr, monkeypatch)
    stopping, release = Event(), Event()

    def blocked_stop():
        # The process can exit before Docker/environment cleanup finishes.
        monkeypatch.setattr(srv, "is_running", lambda: False)
        stopping.set()
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(srv, "stop", blocked_stop)
    monkeypatch.setattr("local_llm_launcher.registry.port_in_use", lambda _port: False)
    monkeypatch.setattr("local_llm_launcher.registry.LocalServer.start", lambda self: True)
    monkeypatch.setattr(mgr, "build_spec", lambda _mode, _model, config, *args: {
        "argv": [], "env": {}, "port": config["port"],
    })
    with ThreadPoolExecutor(max_workers=2) as pool:
        shutdown = pool.submit(getattr(mgr, operation), *(() if operation == "stop_all" else (srv.server_id,)))
        try:
            assert stopping.wait(timeout=2)
            def query_and_launch():
                assert mgr.get(srv.server_id) is srv
                assert mgr.list()[0]["id"] == srv.server_id
                return mgr.launch("llamacpp", GGUF, {"port": srv.port})
            launched = pool.submit(query_and_launch).result(timeout=1)
            assert launched.port == srv.port + 1
        finally:
            release.set()
        shutdown.result(timeout=2)


def test_stop_and_remove_are_serialized_per_server(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    srv = _registered_server(mgr, monkeypatch)
    entered, release, second_entered = Event(), Event(), Event()
    calls = 0

    def blocked_stop():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=5)
        else:
            second_entered.set()
        return False

    monkeypatch.setattr(srv, "stop", blocked_stop)
    with ThreadPoolExecutor(max_workers=2) as pool:
        stopped = pool.submit(mgr.stop, srv.server_id)
        try:
            assert entered.wait(timeout=2)
            removed = pool.submit(mgr.remove, srv.server_id)
            assert not second_entered.wait(timeout=0.2)
        finally:
            release.set()
        assert stopped.result(timeout=2) is False
        assert removed.result(timeout=2) is False
    assert calls == 2
    assert mgr.get(srv.server_id) is srv
    assert ServerManager(app_dir=tmp_path).get(srv.server_id) is not None


def test_successful_remove_does_not_repeat_shutdown(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    srv = _registered_server(mgr, monkeypatch)
    entered, release = Event(), Event()
    calls = 0

    def blocked_stop():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        monkeypatch.setattr(srv, "is_running", lambda: False)
        return True

    monkeypatch.setattr(srv, "stop", blocked_stop)
    with ThreadPoolExecutor(max_workers=2) as pool:
        removed = pool.submit(mgr.remove, srv.server_id)
        try:
            assert entered.wait(timeout=2)
            stopped = pool.submit(mgr.stop, srv.server_id)
        finally:
            release.set()
        assert removed.result(timeout=2) is True
        assert stopped.result(timeout=2) is False
    assert calls == 1
    assert mgr.get(srv.server_id) is None


def test_remove_keeps_record_when_stop_raises(tmp_path, monkeypatch):
    mgr = ServerManager(app_dir=tmp_path)
    srv = _registered_server(mgr, monkeypatch)

    def failed_stop():
        raise OSError("shutdown failed")

    monkeypatch.setattr(srv, "stop", failed_stop)
    with pytest.raises(OSError, match="shutdown failed"):
        mgr.remove(srv.server_id)
    assert mgr.get(srv.server_id) is srv
    assert ServerManager(app_dir=tmp_path).get(srv.server_id) is not None
