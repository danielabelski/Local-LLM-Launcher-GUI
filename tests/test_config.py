"""Settings persistence and app-directory isolation."""
import os
from pathlib import Path
import subprocess
import sys


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
