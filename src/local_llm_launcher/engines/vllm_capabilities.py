"""Bounded evidence from the selected vLLM runtime, never the launcher's Python."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


_METADATA = """
import os, sys
if not getattr(sys.flags, 'safe_path', False):
    sys.path[0] = os.path.dirname(os.path.realpath(sys.argv[1]))
import importlib.metadata, importlib.util, json
try:
    version = importlib.metadata.version('vllm')
except importlib.metadata.PackageNotFoundError:
    version = None
print(json.dumps({'version': version, 'b12x': importlib.util.find_spec('b12x') is not None}))
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


@lru_cache(maxsize=32)
def _native(binary_identity: tuple, interpreter_identity: tuple, environment: tuple, lifetime: int) -> dict:
    binary, interpreter = binary_identity[0], interpreter_identity[0]
    result = {'version': None, 'flags': None, 'choices': {}, 'b12x': None,
              'source': binary, 'message': ''}
    unknown = []
    try:
        output = subprocess.run([binary, 'serve', '--help=all'], capture_output=True,
                                text=True, timeout=8)
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
    if interpreter:
        try:
            output = subprocess.run([interpreter, '-c', _METADATA, binary],
                                    capture_output=True, text=True, timeout=5)
            if output.returncode == 0:
                metadata = json.loads(output.stdout)
                if isinstance(metadata, dict):
                    if isinstance(metadata.get('version'), str):
                        result['version'] = metadata['version']
                    if isinstance(metadata.get('b12x'), bool):
                        result['b12x'] = metadata['b12x']
        except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
            pass
    if result['version'] is None or result['b12x'] is None:
        unknown.append('Version or optional b12x package could not be verified in the selected Python environment.')
    result['message'] = ' '.join(unknown) or 'Checked the selected runtime; package presence does not prove GPU kernel compatibility.'
    return result


def probe(engine_mode: str, binary: str = 'vllm') -> dict:
    """Return unknown on failures; refresh cached native evidence within 60 seconds."""
    if engine_mode == 'vllm-docker':
        return {'version': None, 'flags': None, 'choices': {}, 'b12x': None, 'source': 'docker',
                'message': 'Docker image compatibility is unverified. Check vLLM version and install optional b12x inside the image; host packages do not apply.'}
    resolved = shutil.which(binary) or os.path.abspath(binary)
    interpreter = _interpreter(resolved)
    environment = tuple(os.environ.get(key) for key in (
        'PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONUSERBASE', 'PYTHONSAFEPATH'))
    result = _native(_identity(resolved), _identity(interpreter), environment, int(time.monotonic() // 60))
    # Callers may add advice; keep the cached evidence immutable to them.
    return {**result, 'flags': list(result['flags']) if result['flags'] is not None else None,
            'choices': {flag: list(choices) for flag, choices in result['choices'].items()}}
