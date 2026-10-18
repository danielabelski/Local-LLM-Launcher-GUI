"""Isolated browser fixture: actual launcher APIs and real harmless HTTP child processes."""
import json
import os
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

if len(sys.argv) > 1 and sys.argv[1] == 'model':
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            print('slot print_timing: prompt eval time = 100.00 ms / 100 tokens (1.00 ms per token, 1000.00 tokens per second)', flush=True)
            print('slot print_timing: eval time = 1000.00 ms / 25 tokens (40.00 ms per token, 25.00 tokens per second)', flush=True)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'message':{'content':'Browser fixture response: the launch and chat path works.'}}]}).encode())
    HTTPServer(('127.0.0.1', int(sys.argv[2])), Handler).serve_forever()
else:
    import uvicorn
    from local_llm_launcher import api, hardware
    from local_llm_launcher.app import create_app
    from local_llm_launcher.engines.base import LocalServer
    root = Path(os.environ['LOCAL_LLM_LAUNCHER_HOME'])
    model_path = root / 'Qwen3.8-Q4_K_M.gguf'
    model_path.write_bytes(b'fixture only')
    model = {'repo_id': 'browser-fixture/Qwen3.8-Unsloth', 'format': 'gguf', 'path': str(model_path), 'size_bytes': 12,
             'source': 'folder', 'quant': 'Q4_K_M', 'config': {}, 'param_count_b': 8,
             'gguf_files': [{'filename': model_path.name, 'path': str(model_path), 'size_bytes': 12, 'quant': 'Q4_K_M'}]}
    hw = hardware.Hardware(gpus=[], apple_silicon=None, cpu_cores=8, ram_gb=32, disk_free_gb=100,
                          engines=hardware.EngineAvailability(vllm_native=False, vllm_docker=False, llamacpp_path='/fixture/llama-server')).to_dict()
    api.get_hardware = lambda *a, **kw: hw
    api.installed_models = lambda: [model]
    api.servers.build_spec = lambda mode, model, cfg, *args: {
        'argv': [sys.executable, __file__, 'model', str(cfg['port'])], 'env': {}, 'port': cfg['port']}
    if not api.servers.list():
        api.api_launch(api.LaunchRequest(engine_mode='llamacpp', repo_id=model['repo_id'], config={
            'port': 18881, 'ctx_size': 32768, 'n_gpu_layers': 99, 'flash_attn': True,
            'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0', 'gguf_file': model_path.name}))
        for ident, engine, text in [
            ('vllm-fixture', 'vllm-native', 'INFO Avg prompt throughput: 240.0 tokens/s, Avg generation throughput: 40.0 tokens/s\nINFO Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 20.0 tokens/s\n'),
            ('legacy-fixture', 'llamacpp', 'model loaded; no timing measurements yet\n')]:
            srv = LocalServer(ident, engine, 'browser-fixture/Comparison-Model', 18883, [], {}, api.servers.log_dir,
                              config={'max_model_len': 8192} if ident == 'vllm-fixture' else None)
            srv.log_path.write_text(text)
            api.servers.servers[ident] = srv
        api.servers._save()
    try:
        uvicorn.run(create_app(), host='127.0.0.1', port=18768, log_level='warning')
    finally:
        api.servers.stop_all()
