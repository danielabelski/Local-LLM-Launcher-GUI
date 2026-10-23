"""Version labels must describe the image that a builder actually starts."""
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from local_llm_launcher import engine_versions
from local_llm_launcher.app import create_app


def test_docker_tag_uses_exact_image(monkeypatch):
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        if argv[3] == "ghcr.io/huggingface/text-generation-inference:3.3.7":
            return '{"org.opencontainers.image.version":"3.3.7"}'
        return ""

    monkeypatch.setattr(engine_versions, "_run", run)
    assert engine_versions._docker_tag("ghcr.io/huggingface/text-generation-inference:3.3.7") == "3.3.7"
    assert engine_versions._docker_tag("vllm/vllm-openai:latest") is None
    assert calls == [
        ["docker", "image", "inspect", "ghcr.io/huggingface/text-generation-inference:3.3.7",
         "--format", "{{json .Config.Labels}}"],
        ["docker", "image", "inspect", "vllm/vllm-openai:latest",
         "--format", "{{json .Config.Labels}}"],
    ]


def test_latest_release_lookup_caches_success_and_keeps_last_value_on_failure(monkeypatch):
    calls = []
    now = [1000.0]
    offline = [False]

    def get(url, **_kwargs):
        calls.append(url)
        if offline[0]:
            raise httpx.ConnectError("offline")
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"tag_name": "v1.2.3"})

    monkeypatch.setattr(engine_versions, "_cache", {"at": 0.0, "latest": {}})
    monkeypatch.setattr(engine_versions.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(engine_versions.httpx, "get", get)

    first = engine_versions._latest_versions()
    assert set(first) == set(engine_versions.REPOS)
    assert all(version == "1.2.3" for version in first.values())
    assert len(calls) == len(engine_versions.REPOS)
    assert engine_versions._latest_versions() == first
    assert len(calls) == len(engine_versions.REPOS)

    now[0] += 3601
    offline[0] = True
    assert engine_versions._latest_versions() == first
    assert len(calls) == 2 * len(engine_versions.REPOS)


def test_engine_versions_endpoint_passes_detected_paths(monkeypatch):
    import local_llm_launcher.api as api

    captured = {}

    def get_versions(llama_binary, vllm_binary):
        captured.update(llama=llama_binary, vllm=vllm_binary)
        return {"llamacpp": {"installed": "0.7.0", "latest": "0.7.0"}}

    monkeypatch.setattr(api, "get_hardware", lambda: {"engines": {"llamacpp_path": "/bin/llama-server"}})
    monkeypatch.setattr(api.engine_versions, "get_versions", get_versions)
    response = TestClient(create_app(), base_url="http://127.0.0.1").get("/api/engine-versions")
    assert response.status_code == 200
    assert response.json()["llamacpp"] == {"installed": "0.7.0", "latest": "0.7.0"}
    assert captured["llama"] == "/bin/llama-server"
