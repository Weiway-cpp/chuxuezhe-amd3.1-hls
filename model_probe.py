"""Bounded, direct model-list probe; failure always means not ready."""
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

MAX_RESPONSE_BYTES = 256 * 1024


def _request(config, timeout):
    try:
        headers = {}
        secret = os.environ.get(config.get('api_key_env', 'HLS_API_KEY'), '')
        if secret:
            headers['Authorization'] = 'Bearer ' + secret
        request = Request(config['base_url'].rstrip('/') + '/models', headers=headers)
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            return False
        data = json.loads(raw)
        models = data.get('data', []) if isinstance(data, dict) else []
        return (isinstance(models, list)
                and any(isinstance(item, dict) and item.get('id') == config['model']
                        for item in models))
    except (HTTPException, OSError, ValueError, TypeError, KeyError):
        return False


def probe_models(config, timeout=5.0):
    """Include startup, HTTP headers/body and decoding in the wall-time budget.

    A short budget can mistake a slow Python worker startup for a dead model.
    Five seconds is the default health-probe budget; explicit stricter limits
    still apply to callers that pass timeout. This is not model readiness: the
    separate Ollama check still enforces context and GPU placement.

    A socket timeout only bounds idle intervals. Terminating a dedicated worker
    also bounds a peer that keeps sending data; no timed-out reader is left in
    an application thread. Process cleanup has a separate one-second allowance.
    """
    started = time.monotonic()
    try:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            return False
        deadline = started + timeout
        payload = json.dumps({'config': {key: config[key] for key in
                             ('base_url', 'model', 'api_key_env') if key in config},
                              'timeout': timeout}, ensure_ascii=True)
        if time.monotonic() >= deadline:
            return False
        proc = subprocess.Popen(
            [sys.executable, '-m', 'serve.model_probe', '--worker'],
            cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8')
    except (OSError, ValueError, TypeError, OverflowError):
        return False
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        stdout, _ = proc.communicate(payload, timeout=remaining)
        if proc.returncode != 0 or time.monotonic() >= deadline:
            return False
        ready = json.loads(stdout) is True
        return ready and time.monotonic() < deadline
    except (subprocess.TimeoutExpired, OSError, ValueError, TypeError):
        return False
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                # Never accept readiness when the worker exceeded its budget.
                pass


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        ready = _request(request['config'], request['timeout'])
    except (ValueError, TypeError, KeyError):
        ready = False
    print(json.dumps(ready))
