"""Installed engine versions and cached GitHub stable release tags."""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .engines import tgi, vllm_docker

REPOS = {"llamacpp": "ggml-org/llama.cpp", "vllm": "vllm-project/vllm",
         "sglang": "sgl-project/sglang", "tgi": "huggingface/text-generation-inference"}
_cache = {"at": 0.0, "latest": {}}
_lock = threading.Lock()


def _run(argv, timeout=5, env=None):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return (out.stdout + "\n" + out.stderr).strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _python_package_version(name: str, executable: str | None = None) -> str | None:
    if executable:
        try:
            first = Path(executable).read_bytes().splitlines()[0].decode(errors="replace")
            parts = shlex.split(first[2:].strip()) if first.startswith("#!") else []
            interpreter = parts[0] if parts else None
            if interpreter and Path(interpreter).name == "env" and len(parts) > 1:
                interpreter = shutil.which(parts[1])
        except (OSError, ValueError):
            interpreter = None
        if interpreter and Path(interpreter).is_file():
            code = "import importlib.metadata as m; print(m.version('" + name + "'))"
            result = _run([interpreter, "-c", code], timeout=5)
            if result and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", result):
                return result
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _llama_version(binary: str | None) -> str | None:
    if not binary:
        return None
    env = dict(os.environ)
    directory = str(Path(binary).resolve().parent)
    env["LD_LIBRARY_PATH"] = directory + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    output = _run([binary, "--version"], timeout=5, env=env)
    match = re.search(r"version:\s*([^\s]+)(?:\s*\(build\s*(\d+))?", output)
    if not match:
        return None
    return f"{match[1]} (build {match[2]})" if match[2] else match[1]


def _docker_tag(image: str) -> str | None:
    """Report only the exact local image the engine builder will launch."""
    labels = _run(["docker", "image", "inspect", image,
                   "--format", "{{json .Config.Labels}}"], timeout=5)
    if not labels:
        return None
    tag = image.rsplit(":", 1)[-1]
    if re.fullmatch(r"v?\d+\.\d+\.\d+", tag):
        return tag.removeprefix("v")
    try:
        metadata = json.loads(labels)
        version = (metadata.get("org.opencontainers.image.version") or
                   metadata.get("ai.vllm.image.tag") or "").rsplit(":", 1)[-1]
        if re.fullmatch(r"v?\d+\.\d+\.\d+", version):
            return version.removeprefix("v")
    except (ValueError, TypeError, AttributeError):
        pass
    return f"{tag} tag (version unknown)"


def _latest_versions() -> dict[str, str | None]:
    with _lock:
        if time.monotonic() - _cache["at"] < 3600 and _cache["latest"]:
            return dict(_cache["latest"])
        latest = dict(_cache["latest"])
        for engine, repo in REPOS.items():
            try:
                response = httpx.get(f"https://api.github.com/repos/{repo}/releases/latest",
                                     headers={"Accept": "application/vnd.github+json"}, timeout=3)
                response.raise_for_status()
                tag = response.json().get("tag_name")
                if isinstance(tag, str) and re.fullmatch(r"v?[0-9][A-Za-z0-9._-]*", tag):
                    latest[engine] = tag.removeprefix("v")
            except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                latest.setdefault(engine, None)
        _cache.update(at=time.monotonic(), latest=latest)
        return dict(latest)


def get_versions(llama_binary: str | None = None, vllm_binary: str | None = None) -> dict:
    native_vllm = vllm_binary or shutil.which("vllm")
    latest = _latest_versions()
    return {
        "llamacpp": {"installed": _llama_version(llama_binary), "latest": latest.get("llamacpp")},
        "vllm": {"installed": _python_package_version("vllm", native_vllm) if native_vllm else None,
                 "latest": latest.get("vllm")},
        "vllm_docker": {"installed": _docker_tag(vllm_docker.IMAGE), "latest": latest.get("vllm")},
        "sglang": {"installed": _python_package_version("sglang"), "latest": latest.get("sglang")},
        "tgi": {"installed": _docker_tag(tgi.IMAGE),
                "latest": latest.get("tgi")},
    }
