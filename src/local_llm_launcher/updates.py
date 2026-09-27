"""Opt-in official source builds; activate only a validated isolated executable.

Recipes: llama.cpp/docs/build.md and docs.vllm.ai/en/latest/getting_started/
installation/gpu/ (checked September 2026). Managed directories are permanent:
vLLM's editable install and virtualenv scripts contain absolute paths.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import psutil

# A freshly built vLLM imports torch with a cold bytecode cache on its first run.
VERSION_TIMEOUT = 300

SOURCES = {
    'llamacpp': ('https://github.com/ggml-org/llama.cpp.git', 'master', 'llamacpp_path'),
    'vllm': ('https://github.com/vllm-project/vllm.git', 'main', 'vllm_path'),
}


class UpdateManager:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.app_dir / 'engines'
        self.lock = threading.RLock()
        self.checks = {}
        self.job = None
        self.thread = None

    def _capture(self, argv, **kwargs):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30,
                                check=True, **kwargs)
        return result.stdout.strip()

    def _requirements(self, engine):
        reasons = []
        if platform.system() != 'Linux':
            return ['Managed source builds currently support Linux only; use manual installation on this platform.'], 'cpu', 1
        tools = ['git', 'cmake', 'gcc', 'g++', 'make']
        if engine == 'vllm':
            tools += ['uv', 'nvcc', 'nvidia-smi']
        for tool in tools:
            if not shutil.which(tool):
                reasons.append(f'Install {tool} and add it to PATH before building.')
        cmake_min = (3, 26) if engine == 'vllm' else (3, 14)
        try:
            match = re.search(r'(\d+)\.(\d+)', self._capture(['cmake', '--version']))
            if not match or tuple(map(int, match.groups())) < cmake_min:
                reasons.append(f'The {engine} build needs CMake {cmake_min[0]}.{cmake_min[1]} or newer.')
        except (OSError, subprocess.SubprocessError):
            reasons.append('Could not verify a working CMake installation.')
        backend = 'cuda' if shutil.which('nvcc') and shutil.which('nvidia-smi') else 'cpu'
        if backend == 'cuda':
            try:
                if not self._capture(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']):
                    backend = 'cpu'
            except (OSError, subprocess.SubprocessError):
                backend = 'cpu'
        if engine == 'vllm':
            if not (3, 10) <= sys.version_info[:2] <= (3, 13):
                reasons.append('vLLM source builds need Python 3.10–3.13; run the launcher with a supported Python.')
            try:
                version = tuple(int(x) for x in self._capture(['g++', '-dumpfullversion']).split('.')[:2])
                if version < (11, 3):
                    reasons.append('vLLM requires GCC/G++ 11.3 or newer.')
            except (OSError, ValueError, subprocess.SubprocessError):
                reasons.append('Could not verify GCC/G++ 11.3 or newer.')
            try:
                caps = self._capture(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).splitlines()
                if not caps or any(float(c) < 7.5 for c in caps):
                    reasons.append('vLLM CUDA builds require NVIDIA GPU compute capability 7.5 or higher.')
                self._capture(['nvcc', '--version'])
            except (OSError, ValueError, subprocess.SubprocessError):
                reasons.append('A working NVIDIA driver/GPU and CUDA toolkit (nvcc) are required.')
        free = shutil.disk_usage(self.settings.app_dir).free / 1024**3
        required = 30 if engine == 'vllm' else 5
        if free < required:
            reasons.append(f'Free at least {required} GiB on the app data disk (currently {free:.1f} GiB).')
        if not os.access(self.settings.app_dir, os.W_OK):
            reasons.append('The app data directory must be writable.')
        # vLLM documents large per-compiler memory requirements. Keep both CPU
        # use and available-memory use conservative, including on large servers.
        per_job = 4 if engine == 'vllm' else 2
        jobs = max(1, min(8, os.cpu_count() or 1,
                          int(psutil.virtual_memory().available / 1024**3 / per_job)))
        return reasons, backend, jobs

    def check(self, engine):
        if engine not in SOURCES:
            raise ValueError('Choose llama.cpp or vLLM.')
        reasons, backend, jobs = self._requirements(engine)
        repo, branch, key = SOURCES[engine]
        current = self.settings.data.get(key)
        revision = None
        if not reasons:
            try:
                revision = self._capture(['git', 'ls-remote', repo, f'refs/heads/{branch}']).split()[0]
                if not re.fullmatch(r'[0-9a-f]{40}', revision):
                    raise ValueError('Invalid upstream revision')
            except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
                reasons.append(f'Cannot check official upstream source: {exc}')
        current_match = re.search(r'/([0-9a-f]{40})-[0-9a-f]{8}/', current or '')
        result = dict(current_revision=current_match.group(1) if current_match else None, engine=engine, supported=not reasons, reasons=reasons,
                      backend=backend, jobs=jobs, current_path=current, revision=revision,
                      source=repo, branch=branch, check_id=uuid.uuid4().hex, checked_at=time.time())
        if current and revision and f'/{revision}-' in current:
            result['supported'] = False
            result['reasons'] = ['This source revision is already selected.']
        with self.lock:
            # Checks expire; bound memory even if clients repeatedly check.
            self.checks = {k: v for k, v in self.checks.items() if time.time() - v['checked_at'] < 600}
            if len(self.checks) >= 20:
                self.checks.pop(next(iter(self.checks)))
            self.checks[result['check_id']] = result
        return dict(result)

    def status(self):
        with self.lock:
            result = dict(self.job) if self.job else {'state': 'idle'}
        if result.get('log_path'):
            try:
                with open(result['log_path'], 'rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 24000))
                    result['log'] = stream.read().decode(errors='replace')
            except OSError:
                result['log'] = ''
        return result

    def start(self, check_id):
        with self.lock:
            if self.job and self.job['state'] == 'running':
                raise RuntimeError('An engine build is already running.')
            check = self.checks.get(check_id)
            if not check or time.time() - check['checked_at'] > 600:
                raise ValueError('Check requirements again before updating (checks expire after 10 minutes).')
            if not check['supported']:
                raise ValueError('; '.join(check['reasons']))
            reasons, backend, jobs = self._requirements(check['engine'])
            if backend != check['backend']:
                reasons.append('Detected build hardware changed; check requirements again.')
            if reasons:
                raise ValueError('; '.join(reasons))
            check = {**check, 'jobs': min(check['jobs'], jobs)}
            build = self.root / check['engine'] / (check['revision'] + '-' + uuid.uuid4().hex[:8])
            build.mkdir(parents=True)
            self.job = dict(state='running', engine=check['engine'], revision=check['revision'],
                            stage='Preparing isolated source build', error=None,
                            previous_path=self.settings.data.get(SOURCES[check['engine']][2]),
                            log_path=str(build / 'build.log'), path=str(build))
            self.thread = threading.Thread(target=self._build, args=(dict(check), build), daemon=True)
            self.thread.start()
            return dict(self.job)

    def _run(self, argv, cwd, env):
        with self.lock:
            self.job['stage'] = ' '.join(str(a) for a in argv[:4])
        with open(self.job['log_path'], 'ab') as log:
            log.write(('\n$ ' + ' '.join(map(str, argv)) + '\n').encode())
            log.flush()
            subprocess.run(list(map(str, argv)), cwd=cwd, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=VERSION_TIMEOUT if argv[-1] == '--version' else 7200)

    def _clean_failed_build(self, build):
        # Only this attempt's heavy artifacts are disposable. Logs and job
        # metadata remain, and a selected installation must never be removed.
        for _, _, key in SOURCES.values():
            selected = self.settings.data.get(key)
            if selected and Path(selected).resolve().is_relative_to(build.resolve()):
                return
        for name in ('source', 'venv'):
            artifact = build / name
            if artifact.is_symlink():
                artifact.unlink()
            elif artifact.exists():
                shutil.rmtree(artifact)

    def _build(self, check, build):
        try:
            engine = check['engine']
            repo, _, key = SOURCES[engine]
            source = build / 'source'
            source.mkdir()
            env = dict(os.environ)
            # A full source build must not silently turn into a prebuilt install.
            env.pop('VLLM_USE_PRECOMPILED', None)
            env.pop('VLLM_PRECOMPILED_WHEEL_LOCATION', None)
            env.update(CC='gcc', CXX='g++', MAX_JOBS=str(check['jobs']), CMAKE_BUILD_PARALLEL_LEVEL=str(check['jobs']),
                       GIT_TERMINAL_PROMPT='0')
            for argv in (['git', 'init'], ['git', 'remote', 'add', 'origin', repo],
                         ['git', 'fetch', '--depth', '1', 'origin', check['revision']],
                         ['git', 'checkout', '--detach', check['revision']]):
                self._run(argv, source, env)
            if engine == 'llamacpp':
                self._run(['cmake', '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release',
                           '-DLLAMA_BUILD_TESTS=OFF',
                           '-DGGML_CUDA=' + ('ON' if check['backend'] == 'cuda' else 'OFF')], source, env)
                self._run(['cmake', '--build', 'build', '--config', 'Release', '--target',
                           'llama-server', '--parallel', str(check['jobs'])], source, env)
                binary = source / 'build/bin/llama-server'
            else:
                venv = build / 'venv'
                self._run(['uv', 'venv', '--python', sys.executable, str(venv)], source, env)
                self._run(['uv', 'pip', 'install', '--python', str(venv / 'bin/python'),
                           '-e', '.', '--torch-backend=auto'], source, env)
                binary = venv / 'bin/vllm'
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError('Build produced no usable executable; inspect the build log.')
            self._run([str(binary), '--version'], source, env)
            self.settings.update({key: str(binary)})
            with self.lock:
                self.job.update(state='complete', stage='Ready for future launches', executable=str(binary))
        except Exception as exc:
            cleanup_error = ''
            if isinstance(exc, subprocess.TimeoutExpired):
                # subprocess.run stops its direct child on timeout, but compiler
                # descendants may remain. Do not delete files they may be using.
                cleanup_error = ' Build artifacts retained because child build processes may still be running after the timeout.'
            else:
                try:
                    self._clean_failed_build(build)
                except OSError as cleanup_exc:
                    cleanup_error = f' Could not remove failed build artifacts: {cleanup_exc}.'
            with self.lock:
                self.job.update(state='failed', stage='Build failed; previous engine retained',
                                error=f'{exc}.{cleanup_error} Review the build log and requirements, then check again to retry.')
