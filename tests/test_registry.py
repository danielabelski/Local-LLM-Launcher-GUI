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
