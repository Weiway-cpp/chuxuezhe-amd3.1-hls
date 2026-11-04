"""Client for local chat-completions servers with a bounded worker process."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024


class ModelError(RuntimeError):
    def __init__(self, message, diagnostic=None, *, failure_kind=None):
        super().__init__(message)
        self.diagnostic = diagnostic
        # Only locally enforced budgets set this marker. A socket/HTTP failure
        # alone does not establish that the service was healthy.
        self.failure_kind = failure_kind


def http_error_diagnostic(error, *, secret='', messages=()):
    """Extract bounded, known JSON diagnostics, never arbitrary server text.

    Error pages can echo credentials, headers, or the submitted source. Rather
    than persist a best-effort redaction of free text, record fixed categories
    and narrowly formatted exit codes. Unknown details stay in server logs.
    """
    result = {'http_status': error.code, 'body_format': 'unavailable'}
    try:
        raw = error.read(MAX_ERROR_BYTES + 1)
        if len(raw) > MAX_ERROR_BYTES:
            result['body_format'] = 'too_large'
            return result
        try:
            body = json.loads(raw)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            result['body_format'] = 'non_json'
            return result
        result['body_format'] = 'json'
        detail = body.get('error') if isinstance(body, dict) else None
        if isinstance(detail, dict):
            message = detail.get('message', '')
            for key in ('type', 'code'):
                value = detail.get(key)
                if value != secret and value in ('server_error', 'internal_server_error', 'invalid_request_error',
                             'authentication_error', 'permission_error', 'rate_limit_error',
                             'model_not_found', 'context_length_exceeded', 'insufficient_quota'):
                    result['error_' + key] = value
        else:
            message = detail
        if not isinstance(message, str):
            return result
        if secret:
            message = message.replace(secret, '')
        # Exclude verbatim request echo before inferring any failure category.
        for row in messages:
            content = row.get('content', '')
            if content:
                message = message.replace(content, '')
        patterns = (
            ('out_of_memory', r'out.?of.?memory|outofdevicememory|std::bad_alloc|cannot allocate memory|failed to allocate|not enough (?:memory|vram)|insufficient (?:memory|vram)'),
            ('runner_terminated', r'(?:runner|llama(?:-server)?|server process)[^\r\n]{0,80}(?:terminated|crashed|exited|stopped|exit status)'),
            ('context_limit', r'context (?:length|window|size)[^\r\n]{0,60}(?:exceed|too (?:large|long)|limit)|exceed[^\r\n]{0,40}context'),
            ('backend_timeout', r'timed? out|timeout'),
            ('model_load_failed', r'failed to load (?:model|runner)|error loading model|unable to load model'),
            ('model_not_found', r'model[^\r\n]{0,80}not found'),
            ('device_error', r'device (?:lost|removed|unavailable)|no (?:compatible|available) (?:gpu|device)|invalid device'),
            ('connection_failed', r'connection (?:refused|reset|closed)|failed to connect'),
        )
        categories = [name for name, pattern in patterns if re.search(pattern, message, re.I)]
        if 'out_of_memory' in categories:
            if re.search(r'kv[ _-]+(?:cache|buffer)', message, re.I):
                categories.append('kv_cache_allocation_failed')
            if re.search(r'CPU(?:_REPACK)?[ _]+buffer|ggml_backend_cpu_buffer', message, re.I):
                categories.append('cpu_memory_allocation_failed')
        if categories:
            result['categories'] = categories
        exit_code = re.search(r'\bexit (?:status|code)\s*[:=]?\s*(0x[0-9a-f]{8}|-?\d{1,10})(?![\w.])', message, re.I)
        if exit_code:
            result['process_exit_code'] = exit_code.group(1).lower()
        return result
    except (OSError, ValueError, TypeError):
        return result
    finally:
        try:
            error.close()
        except OSError:
            pass


def http_error_message(diagnostic):
    text = 'Inference HTTP error ' + str(diagnostic['http_status'])
    details = list(diagnostic.get('categories', []))
    if 'process_exit_code' in diagnostic:
        details.append('exit_status=' + diagnostic['process_exit_code'])
    for key in ('error_type', 'error_code'):
        if key in diagnostic:
            details.append(key + '=' + diagnostic[key])
    if details:
        text += '; ' + ', '.join(details)
    else:
        text += '; no recognized safe JSON detail; inspect the inference server log'
    return text


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("Model configuration must be a JSON object")
    cfg = dict(cfg)
    allowed = {"base_url", "model", "max_tokens", "context_tokens", "max_prompt_chars",
               "temperature", "top_p", "seed", "api_key_env", "selftest_review",
               "design_patches", "compile_fallback", "design_first", "reference_selftest",
               "public_interface_check", "fast_csim", "portfolio_candidates",
               "portfolio_sample_seconds", "selftest_retry", "selftest_retry_inline", "majority_seed",
               "slow_endpoint_skip_selftest", "probe_throughput", "regeneration_max_tokens"}
    if set(cfg) - allowed:
        raise ValueError("Unknown model config keys: " + ", ".join(sorted(set(cfg)-allowed)))
    if not isinstance(cfg.get("base_url"), str):
        raise ValueError("base_url is required")
    url = urlsplit(cfg["base_url"])
    if (url.scheme not in ("http", "https") or not url.hostname
            or url.username or url.password or url.query or url.fragment):
        raise ValueError("base_url must be http(s) without credentials, query or fragment")
    _ = url.port  # Validate the port early.
    if not isinstance(cfg.get("model"), str) or not cfg["model"].strip():
        raise ValueError("Set model to the name served by the inference server")
    for key in ("max_tokens", "context_tokens", "max_prompt_chars"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if cfg["max_tokens"] >= cfg["context_tokens"]:
        raise ValueError("max_tokens must leave room for the input context")
    temperature = cfg.get("temperature", 0.1)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if 'top_p' in cfg and (type(cfg['top_p']) not in (int, float)
            or not math.isfinite(cfg['top_p']) or not 0 < cfg['top_p'] <= 1):
        raise ValueError('top_p must be greater than 0 and at most 1')
    if "seed" in cfg and type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if "selftest_review" in cfg and type(cfg['selftest_review']) is not bool:
        raise ValueError('selftest_review must be a boolean')
    for flag in ('design_patches', 'compile_fallback', 'design_first', 'reference_selftest',
                 'public_interface_check', 'fast_csim', 'selftest_retry', 'selftest_retry_inline', 'majority_seed',
                 'slow_endpoint_skip_selftest', 'probe_throughput'):
        if flag in cfg and type(cfg[flag]) is not bool:
            raise ValueError(flag+' must be a boolean')
    if 'portfolio_candidates' in cfg and (type(cfg['portfolio_candidates']) is not int
            or not 0 <= cfg['portfolio_candidates'] <= 4):
        raise ValueError('portfolio_candidates must be an integer between 0 and 4')
    if 'regeneration_max_tokens' in cfg and (type(cfg['regeneration_max_tokens']) is not int
            or not 256 <= cfg['regeneration_max_tokens'] <= 16384):
        raise ValueError('regeneration_max_tokens must be an integer between 256 and 16384')
    if "api_key_env" in cfg and (not isinstance(cfg["api_key_env"], str)
                                or not cfg["api_key_env"]):
        raise ValueError("api_key_env must name an environment variable")
    cfg["temperature"] = temperature
    return cfg


def load_config(path):
    with open(path, encoding="utf-8-sig") as stream:
        return validate_config(json.load(stream))


def _request(config, messages, timeout):
    payload = {
        "model": config["model"], "messages": messages,
        "max_tokens": config["max_tokens"], "temperature": config["temperature"],
        "stream": False,
    }
    if "seed" in config:
        payload["seed"] = config["seed"]
    if 'top_p' in config:
        payload['top_p'] = config['top_p']
    headers = {"Content-Type": "application/json"}
    secret = os.environ.get(config.get("api_key_env", "HLS_API_KEY"), "")
    if secret:
        headers["Authorization"] = "Bearer " + secret
    request = Request(
        config["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        # Explicit local or user-configured destination; do not inherit a proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ModelError("Response exceeds 8 MiB")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise ModelError("Invalid inference response: choices must be a nonempty list")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ModelError("Invalid inference response: missing message")
        content = choice["message"].get("content")
        if content is not None and not isinstance(content, str):
            raise ModelError("Invalid inference response: message.content must be a string or null")
        # An explicit normal stop with empty text is unusable model output,
        # not a transport failure. Preserve it for the controller's existing
        # bounded output retry. Missing content, tool calls and other endings
        # must not silently acquire this recovery path.
        empty_stop = ('content' in choice['message'] and
                      choice.get('finish_reason') == 'stop' and
                      not choice['message'].get('tool_calls') and
                      not choice['message'].get('function_call') and
                      not choice['message'].get('refusal'))
        if ((content is None or not content.strip())
                and choice.get('finish_reason') != 'length' and not empty_stop):
            raise ModelError("Server returned no text content")
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        # Keep finish reason, whitespace and usage as response evidence. The
        # source extractor rejects empty text; no empty candidate is delivered.
        return {"content": content or '', "finish_reason": choice.get("finish_reason"), "usage": usage}
    except HTTPError as exc:
        diagnostic = http_error_diagnostic(exc, secret=secret, messages=messages)
        raise ModelError(http_error_message(diagnostic), diagnostic) from None
    except (URLError, TimeoutError, OSError, ValueError, TypeError) as exc:
        raise ModelError("Inference request failed: " + type(exc).__name__) from None


class ChatClient:
    def __init__(self, config):
        self.config = validate_config(config)

    def generate(self, messages, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelError("Inference time budget exhausted", failure_kind='budget_exhausted')
        started = time.monotonic()
        deadline = started + timeout
        if sum(len(m["content"]) for m in messages) > self.config["max_prompt_chars"]:
            raise ModelError("Prompt exceeds max_prompt_chars; nothing was sent")
        payload = json.dumps({"config": self.config, "messages": messages, "timeout": timeout})
        if time.monotonic() >= deadline:
            raise ModelError("Inference budget exhausted while preparing request", failure_kind='budget_exhausted')
        # urllib's socket timeout alone is not a wall-clock deadline: a slow
        # trickling server can exceed it. A worker permits bounded termination.
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent.model", "--worker"],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        except OSError as exc:
            raise ModelError("Cannot start inference worker: " + type(exc).__name__) from None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired('inference worker startup', timeout)
            stdout, _ = proc.communicate(payload, timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                raise ModelError("Inference timeout; worker cleanup did not finish") from None
            raise ModelError("Inference wall-clock timeout", failure_kind='budget_exhausted') from None
        except BaseException:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise
        if proc.returncode != 0:
            raise ModelError("Inference worker exited with code " + str(proc.returncode))
        if time.monotonic() >= deadline:
            raise ModelError("Inference wall-clock timeout after worker exit", failure_kind='budget_exhausted')
        try:
            result = json.loads(stdout)
        except (ValueError, TypeError):
            raise ModelError("Invalid inference worker response") from None
        if "error" in result:
            raise ModelError(result["error"], result.get('diagnostic'), failure_kind=result.get('failure_kind'))
        finished = time.monotonic()
        if finished >= deadline:
            raise ModelError("Inference wall-clock timeout while decoding response", failure_kind='budget_exhausted')
        result["wall_seconds"] = round(finished - started, 3)
        return result


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        response = _request(request["config"], request["messages"], request["timeout"])
    except ModelError as exc:
        response = {"error": str(exc)}
        if exc.diagnostic is not None:
            response['diagnostic'] = exc.diagnostic
        if exc.failure_kind is not None:
            response['failure_kind'] = exc.failure_kind
    # The parent decodes UTF-8, while a Windows worker may default to GBK.
    # ASCII JSON is lossless after json.loads and works with either encoding.
    print(json.dumps(response, ensure_ascii=True))
