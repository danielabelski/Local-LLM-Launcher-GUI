"""Run Hugging Face Text Generation Inference in its official Docker image."""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict

from ._args import build_args_and_env
from .placement import normalize_device_ids, validate

IMAGE = "ghcr.io/huggingface/text-generation-inference:3.3.7"
HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def _secret_file(values: Dict[str, str]) -> str | None:
    if not values:
        return None
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ValueError("Tokens and API keys must be single-line values.")
    fd, path = tempfile.mkstemp(prefix="llml-tgi-", suffix=".env")
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")
    return path


def build(model: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    validate("tgi", config)
    if model.get("format") == "gguf":
        raise ValueError("TGI needs a Hugging Face model directory, not a GGUF file.")
    if (config.get("max_input_tokens") is not None and config.get("max_total_tokens") is not None
            and int(config["max_input_tokens"]) >= int(config["max_total_tokens"])):
        raise ValueError("Maximum input tokens must be smaller than maximum total tokens.")
    cfg = {k: v for k, v in config.items() if k not in ("port", "host")}
    flags, env, extra = build_args_and_env("tgi", cfg)
    secret_env = {key: value for key, value in env.items()
                  if any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET"))}
    host_port = int(config.get("port", 8082))
    host = str(config.get("host") or "127.0.0.1").strip()
    devices = normalize_device_ids(config.get("device_ids"))
    gpu_request = f'"device={devices}"' if devices is not None else "all"
    env_file = _secret_file(secret_env)
    name = f"llml-tgi-{uuid.uuid4().hex[:8]}"
    argv = ["docker", "run", "--rm", "--name", name,
            "--gpus", gpu_request, "--shm-size", "1g",
            "-p", f"{host}:{host_port}:80",
            "-v", f"{HF_CACHE}:/data"]
    if os.path.exists("/dev/nvidia-uvm"):
        argv.append("--device=/dev/nvidia-uvm")
    if env_file:
        argv += ["--env-file", env_file]
    argv += [IMAGE, "--model-id", model["repo_id"]] + flags + extra
    return {"argv": argv, "env": {}, "port": host_port,
            "health_url": f"http://127.0.0.1:{host_port}/health",
            "container_name": name, "env_file": env_file}
