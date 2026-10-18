"""Safe reusable launch settings and favorites persistence."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_llm_launcher import api, catalog
from local_llm_launcher.config import Settings
from local_llm_launcher.engines.base import LocalServer
from local_llm_launcher.registry import ServerManager


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def launched(tmp_path, monkeypatch, config=None, engine="llamacpp"):
    mgr = ServerManager(tmp_path)
    monkeypatch.setattr("local_llm_launcher.registry.vllm_backends.validate", lambda *a: None)
    monkeypatch.setattr("local_llm_launcher.registry.find_free_port", lambda *a, **k: 45124)
    monkeypatch.setattr(mgr, "build_spec", lambda mode, model, cfg, *a: {
        "argv": [], "env": {}, "port": cfg["port"],
    })
    monkeypatch.setattr(LocalServer, "start", lambda self: True)
    srv = mgr.launch(engine, {"repo_id": "org/model"}, config or {
        "port": 45123, "host": "127.0.0.1", "no_mmap": False,
        "gguf_file": "chosen.gguf", "hf_token": "private-token",
        "_mtp_args": ["private"], "unknown": "private", "extra_args": "--verbose",
    })
    monkeypatch.setattr(api, "servers", mgr)
    return srv


def test_capture_effective_settings_and_reload(tmp_path, monkeypatch):
    srv = launched(tmp_path, monkeypatch)
    expected = {"port": 45124, "no_mmap": False,
                "gguf_file": "chosen.gguf", "extra_args": "--verbose"}  # host follows Settings
    assert srv.status()["config"] == expected
    assert ServerManager(tmp_path).get(srv.server_id).config == expected
    assert "private-token" not in json.dumps(srv.to_record())


def test_save_reload_and_relaunch_with_current_environment(client, tmp_path, monkeypatch):
    srv = launched(tmp_path, monkeypatch, {"port": 45123, "host": "0.0.0.0",
                   "max_model_len": 8192, "hf_token": "old-token"}, "vllm-native")
    response = client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": "  My model  "})
    assert response.status_code == 200
    favorite = response.json()
    assert favorite["name"] == "My model"
    assert favorite["repo_id"] == "org/model"
    assert favorite["engine_mode"] == "vllm-native"
    assert favorite["created_at"]
    loaded = Settings(tmp_path)
    assert loaded.data["favorites"] == [favorite]
    monkeypatch.setattr(api, "settings", loaded)
    loaded.update({"hf_token": "current-token", "lan_access": False, "vllm_path": "/new/vllm"})
    monkeypatch.setattr(api, "find_model", lambda repo: {"repo_id": repo})
    monkeypatch.setattr(api, "get_hardware", lambda: {"engines": {}})
    seen = {}
    def launch(mode, model, config, **kwargs):
        seen.update(mode=mode, model=model, config=config, kwargs=kwargs)
        return SimpleNamespace(status=lambda: {"id": "relaunch"})
    monkeypatch.setattr(api.servers, "launch", launch)
    assert client.post(f"/api/favorites/{favorite['id']}/launch").json() == {"id": "relaunch"}
    assert seen["config"] == {**favorite["config"], "hf_token": "current-token", "host": "127.0.0.1"}
    assert seen["kwargs"]["vllm_binary"] == "/new/vllm"
    assert client.get("/api/favorites").json() == {"favorites": [favorite]}
    assert "current-token" not in client.get("/api/settings").text


@pytest.mark.parametrize("secret", ["--api-key token-secret --verbose", "--hf_token=token-secret", "-hft hf_ABC123", "--authorization Bearer-token", "--password=token-secret", "'unterminated"])
def test_secret_extra_args_not_retained_or_silently_saved(client, tmp_path, monkeypatch, secret):
    srv = launched(tmp_path, monkeypatch, {"port": 8080, "extra_args": secret, "ctx_size": 8192})
    assert "extra_args" not in srv.config
    assert "token-secret" not in json.dumps(srv.to_record())
    response = client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": "Secret"})
    assert response.status_code == 400
    assert "extra arguments" in response.json()["detail"]


def test_api_key_not_retained_and_cannot_silently_relaunch(client, tmp_path, monkeypatch):
    srv = launched(tmp_path, monkeypatch, {"port": 8080, "api_key": "private-key"})
    assert "api_key" not in srv.config
    assert client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": "Protected"}).status_code == 400


def test_old_records_and_unsafe_optional_config(client, tmp_path, monkeypatch):
    srv = LocalServer("old", "llamacpp", "org/model", 8080, [], {}, tmp_path)
    record = srv.to_record()
    record.pop("config", None)
    old = LocalServer.from_record(record, tmp_path)
    assert old.status()["config"] is None
    api.servers.servers["old"] = old
    assert client.post("/api/servers/old/favorites", json={"name": "Old"}).status_code == 400
    record["config"] = {"api_key": "secret", "hf_token": "secret", "_internal": "secret", "no_mmap": False}
    restored = LocalServer.from_record(record, tmp_path)
    assert restored.config == {"no_mmap": False}
    assert restored.config_save_error


def test_errors_and_atomic_rollback(client, tmp_path, monkeypatch):
    srv = launched(tmp_path, monkeypatch)
    for name in ["", "   ", "x" * 121]:
        assert client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": name}).status_code in (400, 422)
    assert client.post("/api/servers/missing/favorites", json={"name": "Missing"}).status_code == 404
    assert client.post("/api/favorites/missing/launch").status_code == 404
    monkeypatch.setattr(api.settings, "save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    response = client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": "No space"})
    assert response.status_code == 500
    assert api.settings.data["favorites"] == []


def test_missing_model_uses_existing_validation(client, tmp_path, monkeypatch):
    srv = launched(tmp_path, monkeypatch)
    favorite = client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": "Model"}).json()
    monkeypatch.setattr(api, "installed_models", lambda: [])
    assert client.post(f"/api/favorites/{favorite['id']}/launch").status_code == 404


def test_settings_list_defaults_are_independent(tmp_path):
    one, two = Settings(tmp_path / "one"), Settings(tmp_path / "two")
    one.append_record("favorites", {"id": "one"})
    assert two.data["favorites"] == []
    assert Settings(tmp_path / "one").data["favorites"] == [{"id": "one"}]


def test_selected_gguf_cannot_fall_back_on_favorite_launch(client, tmp_path, monkeypatch):
    srv = launched(tmp_path, monkeypatch)
    favorite = client.post(f"/api/servers/{srv.server_id}/favorites", json={"name": "Chosen file"}).json()
    monkeypatch.setattr(api, "find_model", lambda repo: {
        "repo_id": repo, "gguf_files": [{"filename": "other.gguf", "path": str(tmp_path / "other.gguf")}],
    })
    response = client.post(f"/api/favorites/{favorite['id']}/launch")
    assert response.status_code == 400
    assert "selected GGUF file" in response.json()["detail"]
    monkeypatch.setattr(api, "find_model", lambda repo: {
        "repo_id": repo, "gguf_files": [{"filename": "chosen.gguf", "path": str(tmp_path / "missing.gguf")}],
    })
    assert client.post(f"/api/favorites/{favorite['id']}/launch").status_code == 400


def test_simultaneous_favorite_appends_are_not_lost(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    settings = Settings(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: settings.append_record("favorites", {"id": n}), range(20)))
    assert sorted(f["id"] for f in Settings(tmp_path).data["favorites"]) == list(range(20))


def test_token_count_extra_arguments_can_be_saved():
    extra = "--max-num-batched-tokens 8192 --max-num-seqs 8"
    config, error = catalog.safe_launch_config("vllm-native", {"extra_args": extra})
    assert config["extra_args"] == extra
    assert error is None


def test_implicit_gguf_selection_is_captured(tmp_path, monkeypatch):
    launched(tmp_path, monkeypatch)
    model = {"repo_id": "org/model", "gguf_files": [
        {"filename": "mmproj.gguf", "path": str(tmp_path / "mmproj.gguf")},
        {"filename": "selected.gguf", "path": str(tmp_path / "selected.gguf")},
        {"filename": "other.gguf", "path": str(tmp_path / "other.gguf")},
    ]}
    srv = api.servers.launch("llamacpp", model, {"ctx_size": 8192})
    assert srv.config["gguf_file"] == "selected.gguf"
    assert ServerManager(tmp_path).get(srv.server_id).config["gguf_file"] == "selected.gguf"


@pytest.mark.parametrize('extra', ['--endpoint https://user:pass@example.test', '--verbose hf_ABC123'])
def test_indirect_extra_argument_credentials_are_not_saved(client, tmp_path, monkeypatch, extra):
    srv = launched(tmp_path, monkeypatch, {'extra_args': extra, 'hf_token': 'hf_ABC123'})
    assert 'extra_args' not in srv.to_record()['config']
    assert client.post(f'/api/servers/{srv.server_id}/favorites', json={'name': 'Unsafe'}).status_code == 400
