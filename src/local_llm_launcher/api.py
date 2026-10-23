"""REST API for the GUI."""
from __future__ import annotations

import ipaddress
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__, advisor, benchmarks, catalog, discovery, engine_versions, failures, hardware
from .config import Settings
from .downloads import DownloadManager, repo_files, search_hub
from .engines import mtp, vllm_backends
from .openwebui import OpenWebUIManager
from .registry import ServerManager
from .updates import UpdateManager

router = APIRouter(prefix="/api")

settings = Settings()
updates = UpdateManager(settings)
servers = ServerManager()
downloads = DownloadManager()
openwebui = OpenWebUIManager()

_hw_cache: Dict[str, Any] = {"at": 0.0, "data": None}
_hw_lock = threading.Lock()


def get_hardware(max_age: float = 5.0) -> Dict[str, Any]:
    now = time.time()
    paths = (settings.data.get("llamacpp_path"), settings.data.get("vllm_path"))
    with _hw_lock:
        if _hw_cache.get("paths") != paths or _hw_cache["data"] is None or now - _hw_cache["at"] > max_age:
            hw = hardware.detect_hardware(llamacpp_hint=settings.data.get("llamacpp_path"),
                                          vllm_hint=settings.data.get("vllm_path"))
            _hw_cache["data"] = hw.to_dict()
            _hw_cache["at"] = now
            _hw_cache["paths"] = paths
        return _hw_cache["data"]


def installed_models() -> List[Dict[str, Any]]:
    models = discovery.list_installed(settings.data.get("gguf_folders") or [])
    return [m.to_dict() for m in models]


def find_model(repo_id: str) -> Dict[str, Any]:
    for m in installed_models():
        if m["repo_id"] == repo_id:
            return m
    raise HTTPException(404, f"Model '{repo_id}' is not installed.")


def recommended_engine(model: Dict[str, Any], hw: Dict[str, Any]) -> str:
    if model["format"] == "gguf":
        return "llamacpp"
    if hw.get("apple_silicon") or not hw.get("gpus"):
        return "llamacpp"
    if hw["engines"].get("vllm_native"):
        return "vllm-native"
    if hw["engines"].get("vllm_docker"):
        return "vllm-docker"
    if hw["engines"].get("sglang"):
        return "sglang"
    if hw["engines"].get("tgi"):
        return "tgi"
    return "vllm-native"


# ----------------------------------------------------------------------- system

@router.get("/hardware")
def api_hardware():
    return get_hardware()


@router.get("/engine-versions")
def api_engine_versions():
    hw = get_hardware()
    return engine_versions.get_versions(hw["engines"].get("llamacpp_path"),
                                        settings.data.get("vllm_path"))


@router.get("/about")
def api_about():
    return {
        "name": "Local-LLM-Launcher-GUI",
        "version": __version__,
        "credits": [
            {
                "name": "vllm-cli by Chen-zexi",
                "url": "https://github.com/Chen-zexi/vllm-cli",
                "note": "This app adapts vllm-cli's flag catalog, configuration profiles, "
                        "server lifecycle management, and model discovery concepts (MIT license).",
            },
            {"name": "vLLM", "url": "https://github.com/vllm-project/vllm"},
            {"name": "llama.cpp", "url": "https://github.com/ggml-org/llama.cpp"},
            {"name": "SGLang", "url": "https://github.com/sgl-project/sglang"},
            {"name": "Hugging Face TGI", "url": "https://github.com/huggingface/text-generation-inference"},
        ],
    }


@router.get("/settings")
def api_get_settings():
    return settings.public()


class SettingsUpdate(BaseModel):
    hf_token: Optional[str] = None
    gguf_folders: List[str] = Field(default_factory=list)
    llamacpp_path: Optional[str] = None
    vllm_path: Optional[str] = None
    lan_access: bool = False


@router.put("/settings")
@router.patch("/settings")
def api_put_settings(body: SettingsUpdate):
    changes = body.model_dump(exclude_unset=True)
    if changes.get("hf_token") == "********":
        changes.pop("hf_token")  # masked value bounced back — keep stored token
    settings.update(changes)
    with _hw_lock:
        _hw_cache["data"] = None  # llamacpp_path may have changed
    return settings.public()


# ----------------------------------------------------------------------- models

@router.get("/models")
def api_models():
    hw = get_hardware()
    out = []
    rank = {"green": 0, "yellow": 1, "red": 2}
    for m in installed_models():
        engine = recommended_engine(m, hw)
        if engine == "llamacpp":
            adv_engine = "llamacpp"
        elif engine == "sglang":
            adv_engine = "sglang"
        elif engine == "tgi":
            adv_engine = "tgi"
        else:
            adv_engine = "vllm"
        # The badge answers "can this run on my hardware at all?" — so report the
        # best outcome across presets, not just the conservative default.
        best = None
        for preset in advisor.presets(adv_engine, m, hw):
            advice = advisor.advise(adv_engine, m, preset["config"], hw)
            if best is None or rank[advice["overall"]["level"]] < rank[best["overall"]["level"]]:
                best = advice
            if best["overall"]["level"] == "green":
                break
        out.append({
            **m,
            "recommended_engine": engine,
            "fit": best["overall"]["level"],
            "fit_headline": best["overall"]["headline"],
        })
    return {"models": out}


@router.get("/models/search")
def api_models_search(q: str):
    try:
        return {"results": search_hub(q, token=settings.data.get("hf_token"))}
    except Exception:
        raise HTTPException(502, "Search failed — check your internet connection and try again.")


@router.get("/models/repo/{repo_id:path}")
def api_repo_detail(repo_id: str):
    try:
        return repo_files(repo_id, token=settings.data.get("hf_token"))
    except Exception:
        raise HTTPException(502, "Could not load repository details — check the name and try again.")


class DownloadRequest(BaseModel):
    repo_id: str
    filename: Optional[str] = None  # set for single-GGUF-file downloads


@router.post("/downloads")
def api_start_download(body: DownloadRequest):
    try:
        job = downloads.start(body.repo_id, body.filename, token=settings.data.get("hf_token"))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return job.to_dict()


@router.get("/downloads")
def api_downloads():
    return {"downloads": downloads.list()}


# ---------------------------------------------------------------- advise/launch

@router.get("/catalog/{engine}")
def api_catalog(engine: str):
    try:
        return catalog.load_catalog(engine)
    except ValueError as e:
        raise HTTPException(404, str(e))


class AdviseRequest(BaseModel):
    engine: str  # "vllm" | "llamacpp" | "sglang" | "tgi"
    engine_mode: Optional[str] = None
    repo_id: str
    config: Dict[str, Any] = {}


@router.post("/advise")
def api_advise(body: AdviseRequest):
    model = find_model(body.repo_id)
    mode = body.engine_mode or body.engine
    try:
        from .engines.placement import validate
        hw = vllm_backends.server_hardware(mode, get_hardware())
        validate(mode, body.config, hw.get("numa"))
        # Advice is polled; never wait on a runtime probe here (launch does).
        warnings = vllm_backends.validate(mode, model, body.config, hw,
                                          settings.data.get("vllm_path") or "vllm", wait=False)
        report = advisor.advise(body.engine, model, body.config, hw)
        report["flags"].update(warnings)
        if body.config.get("use_mtp"):
            decision = mtp.resolve(mode, model, body.config,
                                   vllm_binary=settings.data.get("vllm_path") or "vllm",
                                   llama_binary=hw["engines"].get("llamacpp_path"), wait=False)
            report["flags"]["use_mtp"] = {"level": decision["level"], "message": decision["message"]}
            if decision["level"] == "red":
                # Launch refuses this configuration, so the verdict must too.
                report["overall"] = {"level": "red", "headline": decision["message"],
                                     "details": [report["overall"]["headline"], *report["overall"]["details"]]}
            elif decision["level"] == "yellow":
                warnings["use_mtp"] = decision
        if warnings:
            if report["overall"]["level"] == "green":
                report["overall"]["level"] = "yellow"
            report["overall"]["details"].extend(w["message"] for w in warnings.values())
        return report
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/presets")
def api_presets(engine: str, repo_id: str):
    model = find_model(repo_id)
    return {"presets": advisor.presets(engine, model, get_hardware())}


class LaunchRequest(BaseModel):
    engine_mode: str  # "vllm-native" | "vllm-docker" | "llamacpp" | "sglang" | "tgi"
    repo_id: str
    config: Dict[str, Any] = {}


@router.post("/servers")
def api_launch(body: LaunchRequest):
    model = find_model(body.repo_id)
    config = dict(body.config)
    if body.engine_mode == "llamacpp" and config.get("gguf_file"):
        selected = next((f for f in model.get("gguf_files", [])
                         if f["filename"] == config["gguf_file"]), None)
        if selected is None or not Path(selected["path"]).is_file():
            raise HTTPException(400, "The selected GGUF file is no longer installed. Choose an available file in Launch.")
    if body.engine_mode.startswith("vllm") and settings.data.get("hf_token") and not config.get("hf_token"):
        config["hf_token"] = settings.data["hf_token"]
    if body.engine_mode in ("sglang", "tgi") and settings.data.get("hf_token") and not config.get("hf_token"):
        config["hf_token"] = settings.data["hf_token"]
    explicit_host = str(config.get("host") or "").strip()
    config["host"] = explicit_host or ("0.0.0.0" if settings.data.get("lan_access") else "127.0.0.1")
    hw = get_hardware()
    if body.engine_mode == "llamacpp" and not hw["engines"].get("llamacpp_path"):
        raise HTTPException(400, "llama.cpp (llama-server) was not found on this computer. "
                                 "See Settings for install instructions.")
    if body.engine_mode == "sglang" and not hw["engines"].get("sglang"):
        raise HTTPException(400, "SGLang is not installed on this computer. "
                                 "Install it with: pip install sglang")
    if body.engine_mode == "tgi" and not hw["engines"].get("tgi"):
        raise HTTPException(400, "TGI Docker image was not found. Pull ghcr.io/huggingface/text-generation-inference:3.3.7 first.")
    try:
        srv = servers.launch(body.engine_mode, model, config,
                             llamacpp_binary=hw["engines"].get("llamacpp_path"),
                             vllm_binary=settings.data.get("vllm_path"), hardware=hw)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))
    return _remember_launch(srv.status())


class FavoriteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@router.get("/favorites")
def api_favorites():
    return {"favorites": settings.data["favorites"], "archive_path": str(settings.favorites_archive)}


@router.delete("/favorites/{favorite_id}")
def api_archive_favorite(favorite_id: str):
    try:
        settings.archive_favorite(favorite_id)
    except KeyError:
        raise HTTPException(404, "No such favorite.")
    except (OSError, ValueError, TypeError):
        raise HTTPException(500, "Could not archive this favorite. It has not been removed from active settings.")
    return {"ok": True, "archive_path": str(settings.favorites_archive)}


@router.get("/previous-launch")
def api_previous_launch(repo_id: str):
    previous = [s for s in servers.list() if s["model"] == repo_id]
    previous += [s for s in settings.data["last_launches"] if s["model"] == repo_id]
    if not previous:
        return {"previous": None}
    status = max(previous, key=lambda s: s.get("started_at") or "")
    srv = servers.get(status["id"])
    if srv is not None:
        status = _enrich(srv.status())
    else:
        status = {**status, "running": False}
        if status.get("load_status") == "loading":
            status["load_status"] = "unknown"
    return {"previous": {key: status.get(key) for key in (
        "id", "engine", "model", "config", "config_save_error", "load_status",
        "started_at", "running", "healthy", "exit_code", "failure_explanation")}}


@router.post("/servers/{server_id}/favorites")
def api_save_favorite(server_id: str, body: FavoriteRequest):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Enter a name for the favorite.")
    srv = servers.get(server_id)
    if srv is None:
        raise HTTPException(404, "No such server.")
    if srv.config_save_error:
        raise HTTPException(400, srv.config_save_error)
    if srv.config is None:
        raise HTTPException(400, "Launch this model again before saving its settings; older servers have no saved configuration.")
    favorite = {"id": uuid.uuid4().hex[:12], "name": name, "engine_mode": srv.engine,
                "repo_id": srv.model_label, "config": dict(srv.config),
                "created_at": datetime.now(timezone.utc).isoformat()}
    try:
        settings.append_record("favorites", favorite)
    except OSError:
        raise HTTPException(500, "Could not save the favorite. Check available disk space and settings file permissions.")
    return favorite


@router.post("/favorites/{favorite_id}/launch")
def api_launch_favorite(favorite_id: str):
    favorite = next((f for f in settings.data["favorites"] if f["id"] == favorite_id), None)
    if favorite is None:
        raise HTTPException(404, "No such favorite.")
    config, error = catalog.safe_launch_config(favorite["engine_mode"], favorite["config"])
    if error:
        raise HTTPException(400, error)
    return api_launch(LaunchRequest(engine_mode=favorite["engine_mode"],
                                    repo_id=favorite["repo_id"], config=config))


@router.get("/servers")
def api_servers():
    return {"servers": [_enrich(s) for s in servers.list()]}


@router.get("/benchmarks")
def api_benchmarks():
    return {"benchmarks": settings.data["benchmarks"]}


@router.post("/servers/{server_id}/benchmark")
def api_benchmark(server_id: str):
    srv = servers.get(server_id)
    if srv is None:
        raise HTTPException(404, "No such server.")
    try:
        with srv.log_path.open(encoding="utf-8", errors="replace") as log:
            metrics = benchmarks.summarize(log, srv.engine)
    except OSError:
        raise HTTPException(400, "The server log is not available yet.")
    if not (metrics["prefill_samples"] or metrics["generation_samples"]):
        raise HTTPException(400, "No supported timing measurements in this log yet. Run a test chat, then try Benchmark again.")
    record = {"id": uuid.uuid4().hex[:12], "server_id": server_id,
              "repo_id": srv.model_label, "engine_mode": srv.engine,
              "config": srv.config, "config_note": srv.config_save_error,
              "created_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
    try:
        settings.append_record("benchmarks", record)
    except OSError:
        raise HTTPException(500, "Could not save the benchmark. Check available disk space and settings file permissions.")
    return record


def _remember_launch(status: Dict[str, Any]) -> Dict[str, Any]:
    try:
        settings.remember_launch(status)
    except OSError:
        # The process may already be running: a metadata write failure must not
        # masquerade as a failed launch and invite a duplicate start.
        logging.getLogger(__name__).warning("Could not retain the previous launch settings.")
        status["history_save_error"] = "Launch settings could not be retained. Check free disk space and settings permissions."
    return status


def _enrich(status: Dict[str, Any]) -> Dict[str, Any]:
    srv = servers.get(status["id"])
    if srv is None:
        return status
    if status["running"]:
        status["healthy"] = srv.health()
        if status["healthy"]:
            servers.record_load_status(status["id"], "successful")
    else:
        friendly = failures.translate(srv.tail_logs(300))
        if friendly:
            status["failure_explanation"] = friendly
        if srv.load_status == "loading" or (srv.load_status == "unknown" and friendly):
            servers.record_load_status(status["id"], "failed")
    status["load_status"] = srv.load_status
    return _remember_launch(status)


@router.get("/servers/{server_id}")
def api_server(server_id: str):
    srv = servers.get(server_id)
    if not srv:
        raise HTTPException(404, "No such server.")
    return _enrich(srv.status())


@router.get("/servers/{server_id}/logs")
def api_server_logs(server_id: str, n: int = 200):
    srv = servers.get(server_id)
    if not srv:
        raise HTTPException(404, "No such server.")
    return {"lines": srv.tail_logs(n)}


@router.post("/servers/{server_id}/stop")
def api_server_stop(server_id: str):
    if servers.get(server_id) is None:
        raise HTTPException(404, "No such server.")
    if not servers.stop(server_id):
        raise HTTPException(409, "The server didn't stop — check its log for errors.")
    return {"ok": True}


@router.delete("/servers/{server_id}")
def api_server_remove(server_id: str):
    srv = servers.get(server_id)
    if srv is None:
        raise HTTPException(404, "No such server.")
    if _enrich(srv.status()).get("history_save_error"):
        raise HTTPException(500, "Could not retain these launch settings. The server record has not been removed.")
    if not servers.remove(server_id):
        if servers.get(server_id) is not None:
            raise HTTPException(409, "The server didn't stop; its record and logs were retained. Check its log for errors.")
        raise HTTPException(404, "No such server.")
    return {"ok": True}


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.7


@router.post("/servers/{server_id}/chat")
def api_chat(server_id: str, body: ChatRequest):
    srv = servers.get(server_id)
    if not srv:
        raise HTTPException(404, "No such server.")
    if not srv.is_running():
        raise HTTPException(400, "That server is not running.")
    try:
        r = httpx.post(
            f"http://127.0.0.1:{srv.port}/v1/chat/completions",
            json={"model": srv.model_label, "messages": body.messages,
                  "max_tokens": body.max_tokens, "temperature": body.temperature,
                  "stream": False},
            timeout=600.0,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return {"content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or ""}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"The model server didn't answer: {e}")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise HTTPException(502, "The model server returned an invalid chat response.")


# --------------------------------------------------------------- Open WebUI

@router.get("/openwebui")
def api_openwebui_status():
    return openwebui.status()


@router.post("/openwebui/launch")
def api_openwebui_launch():
    # Pre-connect Open WebUI to every model that's currently running, so they
    # show up ready to chat with the moment it opens.
    connect_urls = [
        f"http://localhost:{s['port']}/v1"
        for s in servers.list()
        if s.get("running")
    ]
    try:
        return openwebui.launch(connect_urls=connect_urls,
                                host="0.0.0.0" if settings.data.get("lan_access") else "127.0.0.1")
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.post("/openwebui/stop")
def api_openwebui_stop():
    openwebui.stop()
    return openwebui.status()


# Source builds are local administrative actions, never LAN/proxy operations.
def _require_local_update(request: Request):
    try:
        local = request.client and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        local = False
    if not local or any(name in request.headers for name in
                        ("forwarded", "x-forwarded-for", "x-real-ip")):
        raise HTTPException(403, "Engine updates require a direct connection from this computer.")


@router.get("/updates")
def api_update_status(request: Request):
    _require_local_update(request)
    return updates.status()


@router.post("/updates/check/{engine}")
def api_update_check(engine: str, request: Request):
    _require_local_update(request)
    try:
        return updates.check(engine)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class UpdateRequest(BaseModel):
    check_id: str


@router.post("/updates")
def api_start_update(body: UpdateRequest, request: Request):
    _require_local_update(request)
    try:
        return updates.start(body.check_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
