"""Bounded evidence from the selected vLLM runtime, never the launcher's Python."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time

# `vllm serve --help=all` imports torch and vLLM; 10-30s is normal on a cold cache.
_HELP_TIMEOUT = 60
_METADATA_TIMEOUT = 20
# Evidence also expires as soon as the executable, its interpreter, or a directory
# on its import path changes (installing a package changes its directory); the TTL
# only bounds changes those checks cannot see.
_TTL = 600
# ponytail: unbounded, one small entry per runtime revision; cap it if keys churn.
_cache: dict = {}  # runtime key -> (checked_at, evidence, import-path stamps)
_refreshing: set = set()
_cache_lock = threading.Lock()
# One probe at a time per runtime: each run starts a heavy Python process, and
# callers that arrive during a run should reuse its result, not start their own.
_probe_locks: dict = {}


_METADATA = """
import os, sys
if not getattr(sys.flags, 'safe_path', False):
    sys.path[0] = os.path.dirname(os.path.realpath(sys.argv[1]))
import importlib.metadata, importlib.util, json
# Stamp the import path before looking, so any later install is noticed.
paths = []
for path in sys.path:
    try:
        paths.append([path, os.stat(path).st_mtime_ns])
    except OSError:
        pass
try:
    version = importlib.metadata.version('vllm')
except importlib.metadata.PackageNotFoundError:
    version = None
# MTP method names and model families come from the installed source text, read
# without importing vLLM (that would import torch): the names changed across
# releases, so no version table can be trusted.
mtp = None
try:
    import re
    spec = importlib.util.find_spec('vllm')
    for root in (spec.submodule_search_locations or []) if spec else []:
        for relative in ('config/speculative.py', 'config/__init__.py', 'config.py'):
            try:
                with open(os.path.join(root, relative), encoding='utf-8') as stream:
                    text = stream.read()
            except OSError:
                continue
            if 'class SpeculativeConfig' not in text:
                continue
            block = (re.search(r'MTPModelTypes\\s*=\\s*Literal\\[(.*?)\\]', text, re.S)
                     or re.search(r'SpeculativeMethod\\s*=\\s*Literal\\[(.*?)\\]', text, re.S))
            body = re.search(r'def hf_config_override\\(.*?(?=\\n    (?:@|def )|\\Z)', text, re.S)
            mtp = {'methods': sorted({name for name in re.findall(r'["\\']([a-z0-9_]+)["\\']', block.group(1))
                                      if name.endswith('mtp')}) if block else [],
                   'models': sorted(set(re.findall(r'["\\']([A-Za-z0-9_.-]+)["\\']', body.group(0)))) if body else None}
            break
        if mtp:
            break
except Exception:
    mtp = None
print(json.dumps({'version': version, 'b12x': importlib.util.find_spec('b12x') is not None,
                  'mtp': mtp, 'paths': paths}))
"""


def _interpreter(binary: str) -> str | None:
    """Recognize Python console scripts; arbitrary shell launchers stay unknown."""
    try:
        with open(binary, 'rb') as stream:
            line = stream.readline(4096).decode('utf-8').strip()
        if not line.startswith('#!'):
            return None
        parts = shlex.split(line[2:])
        if len(parts) == 2 and parts[0] == '/usr/bin/env':
            candidate = shutil.which(parts[1])
        elif len(parts) == 1 and os.path.isabs(parts[0]):
            candidate = parts[0]
        else:
            return None
        if candidate and re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(candidate).name):
            # Do not resolve symlinks: a venv's python may link to system Python.
            return candidate
    except (OSError, UnicodeError, ValueError):
        pass
    return None


def _identity(path: str | None) -> tuple:
    try:
        stat = os.stat(path) if path else None
        return (path, stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size) if stat else (path,)
    except OSError:
        return (path,)


def _native(binary: str, interpreter: str | None) -> tuple[dict, tuple]:
    stamps: tuple = ()
    result = {'version': None, 'flags': None, 'choices': {}, 'b12x': None, 'mtp': None,
              'source': binary, 'message': ''}
    unknown = []
    # Metadata first: its import-path stamps must predate the slow help run, so an
    # install that lands during help invalidates this evidence.
    if interpreter:
        try:
            output = subprocess.run([interpreter, '-c', _METADATA, binary],
                                    capture_output=True, text=True, timeout=_METADATA_TIMEOUT)
            if output.returncode == 0:
                metadata = json.loads(output.stdout)
                if isinstance(metadata, dict):
                    if isinstance(metadata.get('version'), str):
                        result['version'] = metadata['version']
                    if isinstance(metadata.get('b12x'), bool):
                        result['b12x'] = metadata['b12x']
                    if isinstance(metadata.get('mtp'), dict):
                        result['mtp'] = metadata['mtp']
                    stamps = tuple((path, mtime) for path, mtime in metadata.get('paths') or ()
                                   if isinstance(path, str) and isinstance(mtime, int))
        except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
            pass
    try:
        output = subprocess.run([binary, 'serve', '--help=all'], capture_output=True,
                                text=True, timeout=_HELP_TIMEOUT)
        if output.returncode == 0:
            help_text = output.stdout + '\n' + output.stderr
            flags = sorted(set(re.findall(r'(?<![\w-])--[a-z][a-z0-9-]*(?=[\s=,\]{}]|$)', help_text)))
            # Top-level wrapper help is not evidence of serve's full flag set.
            if {'--model', '--dtype', '--max-model-len'}.intersection(flags):
                result['flags'] = flags
                for flag in ('--linear-backend', '--moe-backend', '--attention-backend'):
                    match = re.search(re.escape(flag) + r'\s+\{([^{}]+)\}', help_text)
                    if match:
                        choices = [choice.strip() for choice in match[1].split(',')]
                        if all(re.fullmatch(r'[A-Za-z0-9_-]+', choice) for choice in choices):
                            result['choices'][flag] = choices
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    if result['flags'] is None:
        unknown.append('Command help could not be verified.')
    if result['version'] is None or result['b12x'] is None:
        unknown.append('Version or optional b12x package could not be verified in the selected Python environment.')
    result['message'] = ' '.join(unknown) or 'Checked the selected runtime; package presence does not prove GPU kernel compatibility.'
    return result, stamps


def _mtime(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _fresh(entry) -> bool:
    return (entry is not None and time.monotonic() - entry[0] < _TTL
            and all(_mtime(path) == mtime for path, mtime in entry[2]))


def _refresh(key: tuple, binary: str, interpreter: str | None) -> dict:
    with _cache_lock:
        lock = _probe_locks.setdefault(key, threading.Lock())
    with lock:
        with _cache_lock:
            entry = _cache.get(key)
        if _fresh(entry):
            return entry[1]
        result, stamps = _native(binary, interpreter)
        with _cache_lock:
            _cache[key] = (time.monotonic(), result, stamps)
        return result


def _refresh_in_background(key: tuple, binary: str, interpreter: str | None) -> None:
    with _cache_lock:
        if key in _refreshing:
            return
        _refreshing.add(key)

    def run():
        try:
            _refresh(key, binary, interpreter)
        finally:
            with _cache_lock:
                _refreshing.discard(key)
    try:
        threading.Thread(target=run, name='vllm-probe', daemon=True).start()
    except RuntimeError:
        # Cannot start a thread: advice shows unknown evidence and the next poll retries.
        with _cache_lock:
            _refreshing.discard(key)


def probe(engine_mode: str, binary: str = 'vllm', *, wait: bool = True) -> dict:
    """Return unknown on failures; refresh native evidence once the runtime changes.

    wait=False is for polled advice: never block on a subprocess. Return the last
    evidence for this runtime (or unknown) and refresh it in the background.
    """
    if engine_mode == 'vllm-docker':
        return {'version': None, 'flags': None, 'choices': {}, 'b12x': None, 'mtp': None, 'source': 'docker',
                'message': 'Docker image compatibility is unverified. Check vLLM version and install optional b12x inside the image; host packages do not apply.'}
    resolved = shutil.which(binary) or os.path.abspath(binary)
    interpreter = _interpreter(resolved)
    environment = tuple(os.environ.get(key) for key in (
        'PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONUSERBASE', 'PYTHONSAFEPATH'))
    key = (_identity(resolved), _identity(interpreter), environment)
    with _cache_lock:
        entry = _cache.get(key)
    if _fresh(entry):
        result = entry[1]
    elif wait:
        result = _refresh(key, resolved, interpreter)
    else:
        _refresh_in_background(key, resolved, interpreter)
        result = entry[1] if entry else {
            'version': None, 'flags': None, 'choices': {}, 'b12x': None, 'mtp': None, 'source': resolved,
            'message': 'The selected vLLM runtime is still being checked; advice will update shortly.'}
    return copy.deepcopy(result)  # callers may add advice; the cache stays untouched
