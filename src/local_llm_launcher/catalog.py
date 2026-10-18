"""Flag catalog loading. Curated flag lists live in data/flags_*.json.

Catalog concept adapted from vllm-cli's argument schema
(https://github.com/Chen-zexi/vllm-cli by Chen-zexi, MIT license).
"""
from __future__ import annotations

import json
import re
import shlex
from functools import lru_cache
from importlib import resources
from typing import Any, Dict

ENGINES = ("vllm", "llamacpp")


@lru_cache(maxsize=None)
def load_catalog(engine: str) -> Dict[str, Any]:
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Expected one of {ENGINES}.")
    path = resources.files("local_llm_launcher.data") / f"flags_{engine}.json"
    return json.loads(path.read_text())


def defaults(engine: str) -> Dict[str, Any]:
    """Flat {key: default} for all flags that have a non-None default."""
    return {
        f["key"]: f["default"]
        for f in load_catalog(engine)["flags"]
        if f.get("default") is not None
    }


def safe_launch_config(engine_mode: str, config: Dict[str, Any]):
    """Keep launch inputs, excluding credentials and internal derived fields."""
    engine = "vllm" if engine_mode.startswith("vllm") else engine_mode
    keys = {f["key"] for f in load_catalog(engine)["flags"]}
    keys.add("gguf_file")
    # host is environment-owned: a favorite saved with LAN access on must not reopen the LAN
    # after it is switched off, so a relaunch always follows the current Settings.
    keys.difference_update(("hf_token", "api_key", "host"))
    safe = {k: v for k, v in config.items() if k in keys and not k.startswith("_")}
    error = None
    if config.get("api_key"):
        error = "This launch uses an API key that cannot be saved in Favorites."
    extra = config.get("extra_args")
    if extra:
        try:
            parts = shlex.split(str(extra))
            # Free-form credential flags and credential URLs must never reach saved metadata.
            unsafe = any(re.fullmatch(r"-hft|--(?:hf[-_])?token|--api[-_]?key(?:[-_]file)?|--(?:password|secret|authorization|credentials?)",
                                   part.split("=", 1)[0], re.I)
                         or re.search(r"://[^/\s]*@", part) for part in parts)
            unsafe = unsafe or any(str(config[k]) in str(extra) for k in ("hf_token", "api_key")
                                   if config.get(k))
        except ValueError:
            unsafe = True
        if unsafe:
            safe.pop("extra_args", None)
            error = "Favorites cannot safely retain these extra arguments. Remove credentials or fix quoting before saving."
    return safe, error
