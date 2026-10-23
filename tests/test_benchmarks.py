"""Measured log summaries, distinct from synthetic benchmark workloads."""
import pytest

from local_llm_launcher.benchmarks import summarize


def test_llama_weights_by_measured_time():
    result = summarize([
        'slot print_timing: prompt eval time = 100.00 ms / 100 tokens (1 ms per token, 1000 tokens per second)',
        'slot print_timing: eval time = 1000.00 ms / 20 tokens (50 ms per token, 20 tokens per second)',
        'llama_print_timings: prompt eval time = 900.00 ms / 300 tokens (3 ms per token, 333.33 tokens per second)',
        'llama_print_timings: eval time = 3000.00 ms / 30 runs (100 ms per token, 10 tokens per second)',
        'total time = 5000 ms / 450 tokens',
    ], 'llamacpp')
    assert result['prefill_tps'] == 400
    assert result['generation_tps'] == 12.5
    assert result['prompt_tokens'] == 400
    assert result['generated_tokens'] == 50
    assert result['prefill_ms'] == 1000
    assert result['generation_ms'] == 4000
    assert result['prefill_samples'] == result['generation_samples'] == 2
    assert result['basis'] == 'tokens / measured time'


def test_vllm_interval_averages_include_reported_zero():
    result = summarize([
        '\x1b[32mINFO Engine 000: Avg prompt throughput: 200.0 tokens/s, Avg generation throughput: 20.0 tokens/s, GPU KV cache usage: 40.0%\x1b[0m',
        'INFO Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 10.0 tokens/s',
        'INFO Avg prompt throughput: 400.0 tokens/s, Avg generation throughput: 0.0 tokens/s',
    ], 'vllm-docker')
    assert result['prefill_tps'] == 200
    assert result['generation_tps'] == 10
    assert result['prefill_samples'] == result['generation_samples'] == 3
    assert result['prompt_tokens'] is None
    assert result['generated_tokens'] is None
    assert result['basis'] == 'mean of logged intervals'


@pytest.mark.parametrize('engine', ['llamacpp', 'vllm-native'])
def test_startup_or_invalid_metrics_are_unavailable(engine):
    result = summarize([
        'loaded 100 tokens; 42 tokens/s',
        'prompt eval time = 0 ms / 0 tokens (nan tokens per second)',
        'Avg prompt throughput: -10 tokens/s, Avg generation throughput: inf tokens/s',
        'prompt eval time = 1e999 ms / 10 tokens',
    ], engine)
    assert result['prefill_tps'] is None
    assert result['generation_tps'] is None
    assert result['prefill_samples'] == result['generation_samples'] == 0


def test_missing_phase_is_not_zero():
    result = summarize(['prompt eval time = 100 ms / 20 tokens'], 'llamacpp')
    assert result['prefill_tps'] == 200
    assert result['generation_tps'] is None


def test_stream_reads_samples_before_large_ui_tail():
    lines = iter(['prompt eval time = 100 ms / 20 tokens'] + ['startup padding' * 100] * 3000)
    assert summarize(lines, 'llamacpp')['prefill_tps'] == 200


def test_benchmark_api_persists_full_log_snapshot(tmp_path):
    from fastapi.testclient import TestClient
    from local_llm_launcher import api
    from local_llm_launcher.app import create_app
    from local_llm_launcher.config import Settings
    from local_llm_launcher.engines.base import LocalServer

    srv = LocalServer('bench', 'llamacpp', 'org/model', 8080, [], {}, tmp_path,
                      config={'ctx_size': 8192})
    srv.log_path.write_text('prompt eval time = 100 ms / 20 tokens\n' + 'padding\n' * 50000 +
                            'eval time = 1000 ms / 25 tokens\n')
    api.servers.servers[srv.server_id] = srv
    client = TestClient(create_app(), base_url='http://127.0.0.1')
    response = client.post('/api/servers/bench/benchmark')
    assert response.status_code == 200
    record = response.json()
    assert record['metrics']['prefill_tps'] == 200
    assert record['metrics']['generation_tps'] == 25
    assert record['config'] == {'ctx_size': 8192}
    assert record['repo_id'] == 'org/model'
    assert record['created_at']
    assert api.servers.remove('bench')
    assert Settings(api.settings.app_dir).data['benchmarks'] == [record]
    assert client.get('/api/benchmarks').json() == {'benchmarks': [record]}


def test_benchmark_api_missing_and_unmeasured_logs(tmp_path):
    from fastapi.testclient import TestClient
    from local_llm_launcher import api
    from local_llm_launcher.app import create_app
    from local_llm_launcher.engines.base import LocalServer

    client = TestClient(create_app(), base_url='http://127.0.0.1')
    assert client.post('/api/servers/missing/benchmark').status_code == 404
    srv = LocalServer('empty', 'llamacpp', 'org/model', 8080, [], {}, tmp_path)
    api.servers.servers['empty'] = srv
    assert client.post('/api/servers/empty/benchmark').status_code == 400
    srv.log_path.write_text('model loaded\n')
    response = client.post('/api/servers/empty/benchmark')
    assert response.status_code == 400
    assert 'No supported timing' in response.json()['detail']
    assert api.settings.data['benchmarks'] == []


def test_simultaneous_servers_keep_benchmark_logs_separate(tmp_path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from local_llm_launcher.engines import base

    monkeypatch.setattr(base, 'datetime', SimpleNamespace(now=lambda: datetime(2026, 10, 13)))
    one = base.LocalServer('one', 'llamacpp', 'org/model', 8080, [], {}, tmp_path)
    two = base.LocalServer('two', 'llamacpp', 'org/model', 8081, [], {}, tmp_path)
    assert one.log_path != two.log_path
    one.log_path.write_text('eval time = 1000 ms / 20 tokens\n')
    two.log_path.write_text('eval time = 1000 ms / 40 tokens\n')
    assert summarize(one.log_path.read_text().splitlines(), 'llamacpp')['generation_tps'] == 20
    assert summarize(two.log_path.read_text().splitlines(), 'llamacpp')['generation_tps'] == 40
    assert base.LocalServer.from_record(one.to_record(), tmp_path).log_path == one.log_path
