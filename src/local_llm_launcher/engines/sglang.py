"""Command builder for SGLang (pip-installed).

SGLang is a high-performance serving framework with RadixAttention prefix
caching and structured generation.  Launch command:
    python -m sglang.launch_server --model-path <repo_id> --host 127.0.0.1 --port 30000
"""
from __future__ import annotations

import sys
from typing import Any, Dict

from ._args import build_args_and_env
from .placement import normalize_device_ids, validate, wrap


def build(model: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    validate("sglang", config)
    cfg = {k: v for k, v in config.items() if k != "host"}
    cfg["device_ids"] = normalize_device_ids(config.get("device_ids"))
    flags, env, extra = build_args_and_env("sglang", cfg)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    port = int(config.get("port", 30000))
    host = str(config.get("host") or "127.0.0.1").strip()
    argv = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model["repo_id"],
        "--host", host,
    ] + flags + extra
    return {
        "argv": wrap(argv, config),
        "env": env,
        "port": port,
        "health_url": f"http://127.0.0.1:{port}/health",
    }
