"""Settings persistence and app-directory isolation."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

from local_llm_launcher.config import Settings


def test_app_directory_override_precedes_import_side_effects(tmp_path):
    fallback = tmp_path / "fake-home"
    chosen = tmp_path / "chosen"
    script = """
import sys
from pathlib import Path
Path.home = classmethod(lambda cls: Path(sys.argv[1]))
from local_llm_launcher import api
assert api.settings.app_dir == Path(sys.argv[2])
assert api.servers.app_dir == Path(sys.argv[2])
api.settings.update({'gguf_folders': ['sentinel']})
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(fallback), str(chosen)],
        env={**os.environ, "LOCAL_LLM_LAUNCHER_HOME": str(chosen)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (chosen / "settings.json").is_file()
    assert not (fallback / ".local-llm-launcher").exists()


def test_failed_settings_save_preserves_file_and_memory(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    settings.update({"hf_token": "hf_original"})
    before = settings.path.read_bytes()
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk error"):
        settings.update({"hf_token": "hf_replacement"})
    assert settings.path.read_bytes() == before
    assert settings.data["hf_token"] == "hf_original"
    assert list(tmp_path.iterdir()) == [settings.path]


def test_settings_save_restores_private_permissions(tmp_path):
    settings = Settings(tmp_path)
    settings.save()
    settings.path.chmod(0o644)
    settings.update({"hf_token": "hf_private"})
    assert settings.path.stat().st_mode & 0o777 == 0o600
