"""Backend choices cross advice, command generation, and the launch boundary."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from local_llm_launcher import api
from local_llm_launcher.app import create_app
from local_llm_launcher.engines import vllm_capabilities, vllm_docker, vllm_native

MODEL = {"repo_id": "test/model", "format": "safetensors", "size_bytes": 1024**3,
         "param_count_b": 1, "config": {"torch_dtype": "bfloat16"}}
HW = {"gpus": [{"index": 0, "name": "RTX 5060 Ti", "compute_capability": "12.0",
                "vram_total_mb": 16384, "vram_free_mb": 15000}],
      "engines": {"vllm_native": True, "vllm_docker": True}, "ram_gb": 64,
      "cpu_cores": 32, "apple_silicon": None}
EVIDENCE = {"version": "0.30.0", "flags": ["--linear-backend", "--moe-backend", "--attention-backend"],
            "b12x": True, "source": "/target/bin/vllm", "message": ""}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "find_model", lambda _: MODEL)
    monkeypatch.setattr(api, "get_hardware", lambda: HW)
    monkeypatch.setattr(vllm_capabilities, "probe", lambda *a, **kw: EVIDENCE)
    return TestClient(create_app(), base_url="http://127.0.0.1")


@pytest.mark.parametrize("builder", [vllm_native.build, vllm_docker.build])
def test_explicit_backend_arguments_and_automatic_omission(builder, monkeypatch):
    probe = Mock(return_value=EVIDENCE)
    monkeypatch.setattr(vllm_capabilities, "probe", probe)
    automatic = builder(MODEL, {"linear_backend": None, "moe_backend": None, "attention_backend": None})
    assert "--linear-backend" not in automatic["argv"]
    assert "--moe-backend" not in automatic["argv"]
    assert "--attention-backend" not in automatic["argv"]
    probe.assert_not_called()
    argv = builder(MODEL, {"linear_backend": "flashinfer_cutlass", "attention_backend": "B12X"})["argv"]
    assert argv[argv.index("--linear-backend") + 1] == "flashinfer_cutlass"
    assert argv[argv.index("--attention-backend") + 1] == "B12X"
    assert argv.count("--attention-backend") == 1


@pytest.mark.parametrize("mode", ["vllm-native", "vllm-docker"])
def test_advice_and_launch_reject_same_precision_conflict(client, monkeypatch, mode):
    launch = Mock()
    monkeypatch.setattr(api.servers, "launch", launch)
    config = {"attention_backend": "B12X", "dtype": "float16"}
    advice = client.post("/api/advise", json={"engine": "vllm", "engine_mode": mode,
                                            "repo_id": "test/model", "config": config})
    response = client.post("/api/servers", json={"engine_mode": mode, "repo_id": "test/model", "config": config})
    assert advice.status_code == response.status_code == 400
    assert advice.json()["detail"] == response.json()["detail"]
    launch.assert_not_called()


def test_native_probe_uses_saved_executable(client, monkeypatch):
    api.settings.update({"vllm_path": "/target/bin/vllm"})
    probe = Mock(return_value=EVIDENCE)
    monkeypatch.setattr(vllm_capabilities, "probe", probe)
    response = client.post("/api/advise", json={"engine": "vllm", "engine_mode": "vllm-native",
                          "repo_id": "test/model", "config": {"linear_backend": "flashinfer_cutlass"}})
    assert response.status_code == 200
    probe.assert_called_once_with("vllm-native", "/target/bin/vllm")


def test_unknown_support_is_visible_warning(client, monkeypatch):
    monkeypatch.setattr(vllm_capabilities, "probe", lambda *a: {
        "version": None, "flags": None, "b12x": None, "source": "Docker",
        "message": "Docker runtime support is unverified."})
    response = client.post("/api/advise", json={"engine": "vllm", "engine_mode": "vllm-docker",
                          "repo_id": "test/model", "config": {"attention_backend": "B12X"}})
    assert response.status_code == 200
    report = response.json()
    assert report["flags"]["attention_backend"]["level"] == "yellow"
    assert report["overall"]["level"] != "green"
    assert report["overall"]["details"]


def test_unsupported_runtime_rejected_before_launch(client, monkeypatch):
    monkeypatch.setattr(vllm_capabilities, "probe", lambda *a: {**EVIDENCE, "flags": []})
    launch = Mock()
    monkeypatch.setattr(api.servers, "launch", launch)
    response = client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "test/model",
                          "config": {"linear_backend": "flashinfer_cutlass"}})
    assert response.status_code == 400
    assert "--linear-backend" in response.json()["detail"]
    launch.assert_not_called()


def test_native_gpu_mask_does_not_leak_into_docker_advice(client, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(api, "get_hardware", lambda: {**HW, "gpus": [
        HW["gpus"][0], {**HW["gpus"][0], "index": 1, "compute_capability": "8.9"}]})
    payload = {"engine": "vllm", "repo_id": "test/model",
               "config": {"attention_backend": "B12X"}}
    native = client.post("/api/advise", json={**payload, "engine_mode": "vllm-native"})
    docker = client.post("/api/advise", json={**payload, "engine_mode": "vllm-docker"})
    assert native.status_code == 400
    assert docker.status_code == 200


def test_valid_backend_reaches_launch_unchanged(client, monkeypatch):
    config = {"linear_backend": "flashinfer_cutlass", "attention_backend": "B12X", "dtype": "bfloat16"}
    launch = Mock(return_value=Mock(status=lambda: {"id": "test"}))
    monkeypatch.setattr(api.servers, "launch", launch)
    response = client.post("/api/servers", json={"engine_mode": "vllm-native", "repo_id": "test/model", "config": config})
    assert response.status_code == 200
    assert all(launch.call_args.args[2][key] == value for key, value in config.items())
