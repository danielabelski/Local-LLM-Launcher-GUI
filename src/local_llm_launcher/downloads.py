"""Hugging Face Hub search and download management with polled progress."""
from __future__ import annotations

import glob
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import filter_repo_objects

from .discovery import DEFAULT_HF_HUB, guess_gguf_quant

_SNAPSHOT_IGNORE = ["*.gguf", "original/*", "*.pth"]
MAX_CONCURRENT_DOWNLOADS = 3


def search_hub(query: str, token: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    api = HfApi(token=token)
    results = []
    for m in api.list_models(search=query or None, sort="downloads", limit=limit):
        results.append({
            "repo_id": m.id,
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "gated": bool(getattr(m, "gated", False)),
            "is_gguf": "gguf" in m.id.lower() or "GGUF" in (getattr(m, "tags", None) or []),
            "updated": str(getattr(m, "last_modified", "") or ""),
        })
    return results


def repo_files(repo_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """File listing with sizes so the user can pick one GGUF quant file."""
    api = HfApi(token=token)
    info = api.model_info(repo_id, files_metadata=True)
    files = []
    total_weight_bytes = 0
    gguf = []
    for s in info.siblings or []:
        size = s.size or 0
        entry = {"filename": s.rfilename, "size_bytes": size,
                 "size_gb": round(size / 1024**3, 2),
                 "cache_key": getattr(getattr(s, "lfs", None), "sha256", None)
                              or getattr(s, "blob_id", None)}
        files.append(entry)
        if s.rfilename.endswith(".safetensors"):
            total_weight_bytes += size
        if s.rfilename.endswith(".gguf"):
            gguf.append({**entry, "quant": guess_gguf_quant(s.rfilename)})
    return {
        "repo_id": repo_id,
        "gated": bool(getattr(info, "gated", False)),
        "files": files,
        "gguf_files": sorted(gguf, key=lambda f: f["size_bytes"]),
        "safetensors_total_bytes": total_weight_bytes,
        "is_gguf": len(gguf) > 0,
    }


class DownloadJob:
    def __init__(self, repo_id: str, filename: Optional[str], total_bytes: int,
                 targets: List[tuple[Optional[str], int]]) -> None:
        self.id = uuid.uuid4().hex[:10]
        self.repo_id = repo_id
        self.filename = filename
        self.total_bytes = total_bytes
        self.targets = targets
        self.status = "running"  # running | done | error
        self.error: Optional[str] = None

    def progress_bytes(self) -> int:
        """Bytes of this job's target blobs already present in the HF cache."""
        blobs = DEFAULT_HF_HUB / f"models--{self.repo_id.replace('/', '--')}" / "blobs"
        total = 0
        for key, size in self.targets:
            if not key:
                continue
            try:
                blob = blobs / key
                if blob.is_file():
                    present = blob.stat().st_size
                else:
                    # Hub uses <etag>.<random>.incomplete (older versions used
                    # <etag>.incomplete); the largest is this target's progress.
                    present = max((p.stat().st_size for p in blobs.glob(f"{key}*.incomplete")
                                   if p.name == f"{key}.incomplete" or p.name.startswith(f"{key}.")),
                                  default=0)
                total += min(present, size)
            except OSError:
                continue
        return total

    def to_dict(self) -> Dict[str, Any]:
        done = self.progress_bytes()
        pct = (min(int(done * 100 / self.total_bytes), 99)
               if self.total_bytes and all(key for key, _ in self.targets) else None)
        if self.status == "done":
            pct = 100
        return {
            "id": self.id, "repo_id": self.repo_id, "filename": self.filename,
            "status": self.status, "error": self.error,
            "bytes_done": done, "bytes_total": self.total_bytes, "percent": pct,
        }


class DownloadManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, DownloadJob] = {}
        self._lock = threading.Lock()

    def start(self, repo_id: str, filename: Optional[str] = None,
              token: Optional[str] = None) -> DownloadJob:
        try:
            detail = repo_files(repo_id, token=token)
        except Exception as e:
            raise RuntimeError(_friendly_hub_error(e)) from e

        # vLLM reads top-level weights and prefers safetensors, so top-level .bin/.pt
        # files beside top-level safetensors are unused copies of the same weights.
        top = [f["filename"] for f in detail["files"] if "/" not in f["filename"]]
        ignore = _SNAPSHOT_IGNORE + ([glob.escape(n) for n in top if n.endswith((".bin", ".pt"))]
                                     if any(n.endswith(".safetensors") for n in top) else [])
        if filename:
            files = [f for f in detail["files"] if f["filename"] == filename]
        else:
            files = list(filter_repo_objects(
                detail["files"], ignore_patterns=ignore,
                key=lambda item: item["filename"]))

        targets = [(f.get("cache_key"), f["size_bytes"]) for f in files]
        job = DownloadJob(repo_id, filename, sum(size for _, size in targets), targets)
        with self._lock:
            active = sum(1 for j in self.jobs.values() if j.status == "running")
            if active >= MAX_CONCURRENT_DOWNLOADS:
                raise RuntimeError(
                    f"Too many downloads already running ({active}). "
                    "Wait for one to finish before starting another."
                )
            self.jobs[job.id] = job

        def run() -> None:
            try:
                if filename:
                    hf_hub_download(repo_id=repo_id, filename=filename, token=token)
                else:
                    snapshot_download(repo_id=repo_id, token=token,
                                      ignore_patterns=ignore)
                job.status = "done"
            except Exception as e:  # surfaced to the GUI, must not kill the app
                job.error = _friendly_hub_error(e)
                job.status = "error"

        threading.Thread(target=run, daemon=True).start()
        return job

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self.jobs.values())
        return [j.to_dict() for j in jobs]


def _friendly_hub_error(e: Exception) -> str:
    text = str(e)
    if "401" in text or "gated" in text.lower():
        return ("Access denied. If this model is gated, accept its license on huggingface.co "
                "and add your access token in Settings.")
    if "404" in text:
        return "Model not found — check the name."
    if "Connection" in text or "Timeout" in text:
        return "Network problem reaching huggingface.co — check your internet connection."
    return text[:300]
