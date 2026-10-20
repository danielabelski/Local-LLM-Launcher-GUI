"""App settings persisted in ~/.local-llm-launcher/settings.json (chmod 600 —
it may hold a Hugging Face token)."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "hf_token": None,
    "gguf_folders": [],
    "llamacpp_path": None,
    "vllm_path": None,
    "lan_access": False,
    "favorites": [],
    "benchmarks": [],
}


class Settings:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        from .registry import APP_DIR
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.path = self.app_dir / "settings.json"
        self._lock = threading.RLock()
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = copy.deepcopy(DEFAULTS)
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                stored = json.loads(self.path.read_text())
                if isinstance(stored, dict):
                    self.data.update(stored)
            except (OSError, json.JSONDecodeError):
                pass

    def save(self) -> None:
        with self._lock:
            fd, name = tempfile.mkstemp(dir=self.app_dir, prefix=".settings-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self.data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(name, self.path)
            finally:
                Path(name).unlink(missing_ok=True)

    def update(self, changes: Dict[str, Any]) -> None:
        with self._lock:
            previous = self.data
            self.data = {**previous, **{k: v for k, v in changes.items() if k in DEFAULTS}}
            try:
                self.save()
            except Exception:
                self.data = previous
                raise

    def append_record(self, key: str, record: Dict[str, Any]) -> None:
        """Append and persist without losing simultaneous saves or failed writes."""
        if key not in ("favorites", "benchmarks"):
            raise ValueError("Unknown record collection.")
        with self._lock:
            self.update({key: [*self.data[key], record]})

    @property
    def favorites_archive(self) -> Path:
        return self.app_dir / "favorites-archive.json"

    def archive_favorite(self, favorite_id: str) -> None:
        """Write the recovery copy before removing a favorite from active settings."""
        with self._lock:
            favorite = next((f for f in self.data["favorites"] if f["id"] == favorite_id), None)
            if favorite is None:
                raise KeyError(favorite_id)
            path = self.favorites_archive
            archived = json.loads(path.read_text()) if path.exists() else []
            if not isinstance(archived, list) or not all(isinstance(f, dict) for f in archived):
                raise ValueError("The favorites archive must contain a JSON list.")
            # A failed settings save may leave an existing recovery copy.
            if not any(f.get("id") == favorite_id for f in archived):
                archived.append({**favorite, "archived_at": datetime.now(timezone.utc).isoformat()})
            fd, name = tempfile.mkstemp(dir=self.app_dir, prefix=".favorites-archive-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(archived, stream, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, path)
                if os.name == "posix":
                    directory = os.open(self.app_dir, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                Path(name).unlink(missing_ok=True)
            self.update({"favorites": [f for f in self.data["favorites"] if f["id"] != favorite_id]})

    def public(self) -> Dict[str, Any]:
        """Settings safe to send to the browser — token masked."""
        out = dict(self.data)
        if out.get("hf_token"):
            out["hf_token"] = "********"
            out["hf_token_set"] = True
        else:
            out["hf_token_set"] = False
        return out
