"""Command builder for native (pip-installed) vLLM.

Adapted from vllm-cli's server command construction
(https://github.com/Chen-zexi/vllm-cli by Chen-zexi, MIT license).
"""
from __future__ import annotations

from typing import Any, Dict

from ._args import build_args_and_env
from .placement import normalize_device_ids, validate, wrap


def build(model: Dict[str, Any], config: Dict[str, Any], binary: str = "vllm") -> Dict[str, Any]:
    validate("vllm-native", config)
    config = {**config, "device_ids": normalize_device_ids(config.get("device_ids"))}
    flags, env, extra = build_args_and_env("vllm", config)
    # GPU numbers everywhere in the launcher (device_ids, an inherited
    # CUDA_VISIBLE_DEVICES, advice) are nvidia-smi's; CUDA defaults to fastest-first.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    port = int(config.get("port", 8000))
    host = config.get("host", "127.0.0.1")
    argv = ([binary, "serve", model["repo_id"]]
            + (["--host", host] if host else [])
            + flags + extra)
    return {
        "argv": wrap(argv, config),
        "env": env,
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
    }
