"""Download job progress and admission control."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

from local_llm_launcher import downloads


def test_repo_files_exposes_hub_cache_keys(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename="large.gguf", size=10, blob_id="git-pointer",
                        lfs=SimpleNamespace(sha256="lfs-hash")),
        SimpleNamespace(rfilename="config.json", size=2, blob_id="git-blob", lfs=None),
    ]
    monkeypatch.setattr(downloads, "HfApi", lambda token=None: SimpleNamespace(
        model_info=lambda *a, **k: SimpleNamespace(siblings=siblings, gated=False)))
    files = downloads.repo_files("org/model")["files"]
    assert [f["cache_key"] for f in files] == ["lfs-hash", "git-blob"]


def test_progress_counts_only_requested_cache_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "DEFAULT_HF_HUB", tmp_path)
    blobs = tmp_path / "models--org--model" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "old-quant").write_bytes(b"x" * 100)
    (blobs / "wanted.abc123.incomplete").write_bytes(b"y" * 4)
    job = downloads.DownloadJob("org/model", "wanted.gguf", 10, [("wanted", 10)])
    assert job.to_dict()["bytes_done"] == 4
    assert job.to_dict()["percent"] == 40


def test_snapshot_total_excludes_ignored_files(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "DEFAULT_HF_HUB", tmp_path)
    monkeypatch.setattr(downloads, "repo_files", lambda *a, **k: {"files": [
        {"filename": "model.safetensors", "size_bytes": 10, "cache_key": "model"},
        {"filename": "q4.gguf", "size_bytes": 20, "cache_key": "gguf"},
        {"filename": "original/weights.bin", "size_bytes": 30, "cache_key": "original"},
        {"filename": "README.md", "size_bytes": 2, "cache_key": "readme"},
    ]})
    monkeypatch.setattr(downloads, "snapshot_download", lambda **kwargs: None)
    job = downloads.DownloadManager().start("org/model")
    assert job.total_bytes == 12


def test_progress_unknown_when_cache_key_missing():
    job = downloads.DownloadJob("org/model", "wanted.gguf", 10, [(None, 10)])
    assert job.to_dict()["percent"] is None


def test_concurrent_starts_obey_limit(monkeypatch):
    monkeypatch.setattr(downloads, "MAX_CONCURRENT_DOWNLOADS", 1)
    metadata_barrier = Barrier(2)
    worker_release = Event()

    def repo_files(*args, **kwargs):
        metadata_barrier.wait(timeout=5)
        return {"files": [{"filename": "model.gguf", "size_bytes": 1, "cache_key": "key"}]}

    monkeypatch.setattr(downloads, "repo_files", repo_files)
    monkeypatch.setattr(downloads, "hf_hub_download", lambda **kwargs: worker_release.wait(timeout=5))
    manager = downloads.DownloadManager()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(manager.start, "org/model", "model.gguf") for _ in range(2)]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=5))
                except RuntimeError as exc:
                    outcomes.append(exc)
        assert sum(isinstance(value, downloads.DownloadJob) for value in outcomes) == 1
        assert sum(isinstance(value, RuntimeError) for value in outcomes) == 1
    finally:
        worker_release.set()
