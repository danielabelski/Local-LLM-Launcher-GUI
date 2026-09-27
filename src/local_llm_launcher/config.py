"""App settings persisted in ~/.local-llm-launcher/settings.json (chmod 600 —
it may hold a Hugging Face token)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "hf_token": None,
    "gguf_folders": [],
    "llamacpp_path": None,
    "lan_access": False,
}


class Settings:
    def __init__(self, app_dir: Optional[Path] = None) -> None:
        from .registry import APP_DIR
        self.app_dir = Path(app_dir) if app_dir else APP_DIR
        self.path = self.app_dir / "settings.json"
        self._lock = threading.RLock()
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = dict(DEFAULTS)
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

    def public(self) -> Dict[str, Any]:
        """Settings safe to send to the browser — token masked."""
        out = dict(self.data)
        if out.get("hf_token"):
            out["hf_token"] = "********"
            out["hf_token_set"] = True
        else:
            out["hf_token_set"] = False
        return out
