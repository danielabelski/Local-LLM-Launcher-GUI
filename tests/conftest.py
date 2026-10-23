"""Isolate state before collection imports the API's global managers."""
import os
import sys
import tempfile

import pytest


def pytest_configure(config):
    config.launcher_temp = tempfile.TemporaryDirectory(prefix="launcher-tests-")
    config.launcher_previous_dir = os.environ.get("LOCAL_LLM_LAUNCHER_HOME")
    os.environ["LOCAL_LLM_LAUNCHER_HOME"] = config.launcher_temp.name


def pytest_unconfigure(config):
    previous = config.launcher_previous_dir
    if previous is None:
        os.environ.pop("LOCAL_LLM_LAUNCHER_HOME", None)
    else:
        os.environ["LOCAL_LLM_LAUNCHER_HOME"] = previous
    config.launcher_temp.cleanup()


@pytest.fixture(autouse=True)
def isolated_api_state(tmp_path, monkeypatch):
    api = sys.modules.get("local_llm_launcher.api")
    if api is not None:
        monkeypatch.setattr(api, "settings", api.Settings(app_dir=tmp_path))
        monkeypatch.setattr(api, "updates", api.UpdateManager(api.settings))
        monkeypatch.setattr(api, "servers", api.ServerManager(app_dir=tmp_path))
        monkeypatch.setattr(api, "downloads", api.DownloadManager())
        monkeypatch.setattr(api, "openwebui", api.OpenWebUIManager(app_dir=tmp_path))
        monkeypatch.setattr(api, "_hw_cache", {"at": 0.0, "data": None})
