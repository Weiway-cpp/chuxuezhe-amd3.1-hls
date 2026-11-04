"""Loopback HTTP endpoint for the published competition contract.

Start with python -m serve.competition_api --port 7860 and FPGACHINA_TOKEN.
AGENT_CONFIG, LLM_BASE_URL and MODEL_NAME match agent.competition and baseline.
HEALTH_FILE (default runs/health.json) must contain a real measurement:
{"observed_at": <Unix seconds>, "model": "<served id>",
 "model_loaded": true, "vram_gb": <measured number>}.
Measurements expire after 120 seconds; /models is checked before reporting ready.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from serve.model_probe import probe_models

from scripts.verify_hls import _close_process_control, _launch_process, _stop_process_tree

ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
HEALTH_MAX_AGE_SECONDS = 120.0
LOG = logging.getLogger(__name__)


class RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def validate_request(data):
    required = {"task_id", "nonce", "mode", "prompt", "interface", "deadline_s"}
    if not isinstance(data, dict) or set(data) != required:
        raise RequestError(400, "Request must contain exactly the six contract fields")
    for name, limit in (("task_id", 1024), ("nonce", 1024),
                        ("prompt", 1024 * 1024), ("interface", 512 * 1024)):
        value = data[name]
        if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
            raise RequestError(400, "Invalid " + name)
    if data["mode"] not in ("agent", "baseline"):
        raise RequestError(400, "mode must be agent or baseline")
    first_line = data["interface"].splitlines()[0].strip()
    if not re.fullmatch(r"//\s*header:\s*[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:h|hh|hpp|hxx)", first_line):
        raise RequestError(400, "interface must begin with // header: <safe header filename>")
    seconds = data["deadline_s"]
    try:
        valid_seconds = type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
    except OverflowError:
        valid_seconds = False
    if not valid_seconds:
        raise RequestError(400, "deadline_s must be positive and finite")
    return dict(data)


def _trace_event(event, **fields):
    return json.dumps({"ts": time.time(), "event": event, **fields}, ensure_ascii=False) + "\n"


def _read_artifact(path, workspace):
    if not path.exists():
        return ""
    if (path.is_symlink() or not path.resolve().is_relative_to(workspace)
            or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES):
        raise ValueError("Unsafe or oversized output artifact")
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("Oversized output artifact")
    return raw.decode("utf-8")


def cancel_worker(proc, control=None, *, graceful=False):
    """Allow POSIX agent cleanup before killing its outer process group.

    Only an orderly exit or our controlled cancellation code acknowledges the
    unwind. Other exits or a missed grace deadline fence future work.
    """
    errors = []
    if graceful and control is None and os.name != 'nt':
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            code = proc.wait(timeout=2.0)
            if code not in (0, 128 + signal.SIGTERM):
                errors.append('Agent exited without cooperative cleanup acknowledgement')
        except ProcessLookupError:
            if proc.poll() != 0:
                errors.append('Agent disappeared without cleanup acknowledgement')
        except subprocess.TimeoutExpired:
            errors.append('Agent cancellation grace expired; nested EDA cleanup unconfirmed')
        except OSError as exc:
            errors.append('Cooperative cancellation failed: ' + str(exc))
    return errors + _stop_process_tree(proc, control)


def run_subprocess(request, input_dir, output_dir, seconds, *, root=ROOT, env=None,
                   stop_event=None):
    """Supervise children; cancellation is checked at most 100 ms apart."""
    stop_event = stop_event or threading.Event()
    # Match the agent/probe's direct model connection without editing baseline.
    # Remove mixed-case aliases before inserting both conventional spellings:
    # urllib's environment scan can otherwise let a later No_proxy override it.
    env = {key:value for key,value in (os.environ if env is None else env).items()
           if key.lower() != 'no_proxy'}
    env.update(NO_PROXY='*', no_proxy='*')
    env.update(AGENT_DEADLINE_S=str(seconds), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if request["mode"] == "baseline":
        command = [sys.executable, str(root / "baseline.py"),
                   str(input_dir), str(output_dir), "hls"]
    else:
        command = [sys.executable, "-m", "agent.competition",
                   "--input", str(input_dir), "--output", str(output_dir),
                   "--deadline-seconds", str(seconds)]
    proc = control = None
    result = {"returncode": None, "timeout": False, "cancelled": False, "cleanup_errors": []}
    deadline = time.monotonic() + seconds
    with (output_dir.parent / "process.log").open("w", encoding="utf-8") as log:
        try:
            if stop_event.is_set():
                result["cancelled"] = True
                return result
            proc, control = _launch_process(command, cwd=root, stdout=log, env=env)
            while True:
                remaining = deadline - time.monotonic()
                if stop_event.is_set() or remaining <= 0:
                    result["cancelled"] = stop_event.is_set()
                    result["timeout"] = not result["cancelled"]
                    result["cleanup_errors"].extend(cancel_worker(proc, control, graceful=request['mode']=='agent'))
                    result["returncode"] = proc.poll()
                    break
                try:
                    result["returncode"] = proc.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except OSError as exc:
            result["error"] = type(exc).__name__
        finally:
            # POSIX needs process-group cleanup even after natural parent exit.
            if proc is not None and control is None and not (result["timeout"] or result["cancelled"]):
                result["cleanup_errors"].extend(_stop_process_tree(proc, control))
            result["cleanup_errors"].extend(_close_process_control(control))
            if control is not None and not control.cleanup_confirmed:
                result["cleanup_errors"].append("Process cleanup not confirmed")
    result["cleanup_errors"] = list(dict.fromkeys(result["cleanup_errors"]))
    return result


class CompetitionService:
    def __init__(self, config, token, *, root=ROOT, work_root=None,
                 health_file=None, runner=None, model_probe=None):
        if (not isinstance(token, str) or not token or token.strip() != token
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("Set FPGACHINA_TOKEN to a nonempty printable ASCII token")
        self.token = token
        self.root = Path(root).resolve()
        self.config = json.loads(json.dumps(config))
        if not self.config.get("model") or not self.config.get("base_url"):
            raise ValueError("Model name and inference base URL are required")
        self.work_root = Path(
            work_root or os.environ.get("EDA_TMP") or os.environ.get("AGENT_TMP")
            or self.root / "runs" / "api").resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.health_file = Path(
            health_file or os.environ.get("HEALTH_FILE") or self.root / "runs" / "health.json")
        self._runner = runner
        self._model_probe = model_probe or probe_models
        self._compute_lock = threading.Lock()
        self._probe_condition = threading.Condition()
        self._active_probes = 0
        self._nonce_lock = threading.Lock()
        self._seen_nonces = set()
        self._supervision_ok = True
        self.stop_event = threading.Event()

    def authorized(self, value):
        return isinstance(value, str) and hmac.compare_digest(
            value.encode("utf-8"), ("Bearer " + self.token).encode("ascii"))

    def health(self):
        health = self._health_snapshot()
        if not health['ready']:
            return health
        # The bounded model-list probe can take seconds. A collector may publish
        # an unload/over-limit sample (or stop) during that wait. Read and validate
        # telemetry again afterwards; a successful list response is not proof of
        # continued GPU readiness. Only one probe is dispatched per health call.
        health['ready'] = False
        try:
            if self._probe_ready():
                return self._health_snapshot()
        except (OSError, ValueError, TypeError, OverflowError):
            pass
        return health

    def _health_snapshot(self):
        """Validate current local evidence; the caller must also probe the model."""
        model = self.config["model"]
        health = {"ready": False, "track": "hls", "model": model, "vram_gb": None}
        try:
            if self.health_file.stat().st_size > 16384:
                return health
            measurement = json.loads(self.health_file.read_text(encoding="utf-8"))
            if not isinstance(measurement, dict):
                return health
            observed, vram = measurement.get("observed_at"), measurement.get("vram_gb")
            if (type(observed) not in (int, float) or not math.isfinite(observed)
                    or not 0 <= time.time() - observed <= HEALTH_MAX_AGE_SECONDS
                    or type(vram) not in (int, float) or not math.isfinite(vram) or vram <= 0
                    or measurement.get("model") != model):
                return health
            health["vram_gb"] = vram
            peak = measurement.get('peak_observed_vram_bytes')
            peak_ok = (peak is None or (type(peak) is int and 0 < peak <= 32 * 1024**3
                                       and peak >= vram * 1024**3))
            available = (self._runner is not None
                         or ((self.root / "baseline.py").is_file()
                             and (self.root / "agent" / "competition.py").is_file()))
            health["ready"] = bool(
                self._supervision_ok and not self.stop_event.is_set() and available and vram <= 32 and peak_ok
                and measurement.get("model_loaded") is True
                and not model.startswith("REPLACE_"))
        except (OSError, ValueError, TypeError, OverflowError):
            pass
        return health

    def _probe_ready(self):
        # HTTP request threads are daemons. Register a probe before launching
        # its child so orderly shutdown waits for that child's cleanup too.
        with self._probe_condition:
            if self.stop_event.is_set():
                return False
            self._active_probes += 1
        try:
            return bool(self._model_probe(self.config)) and not self.stop_event.is_set()
        finally:
            with self._probe_condition:
                self._active_probes -= 1
                self._probe_condition.notify_all()

    def request_stop(self):
        """Signal-safe shutdown request; active workers own child cleanup."""
        self.stop_event.set()

    def wait_stopped(self, timeout=8.5):
        """Wait for solver and health children within one shutdown budget."""
        deadline = time.monotonic() + max(0, min(timeout, threading.TIMEOUT_MAX))
        acquired = self._compute_lock.acquire(timeout=max(0, deadline - time.monotonic()))
        if acquired:
            self._compute_lock.release()
        with self._probe_condition:
            probes_stopped = self._probe_condition.wait_for(
                lambda: self._active_probes == 0, timeout=max(0, deadline - time.monotonic()))
        return acquired and probes_stopped

    def _acquire_compute(self, deadline):
        # Short waits both admit arbitrary finite deadlines and wake on SIGTERM.
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._compute_lock.acquire(timeout=min(0.1, remaining)):
                return True
        return False

    def solve(self, data, *, received_at=None):
        started = time.monotonic() if received_at is None else received_at
        request = validate_request(data)
        digest = hashlib.sha256(request["nonce"].encode("utf-8")).digest()
        with self._nonce_lock:
            if digest in self._seen_nonces:
                raise RequestError(409, "nonce has already been used")
            self._seen_nonces.add(digest)
        response = {"task_id": request["task_id"], "solution": "", "trace": "", "elapsed_s": 0.0}
        deadline = started + request["deadline_s"]
        reserve = min(8.0, max(0.02, request["deadline_s"] * 0.05))
        acquired, workspace = False, None
        delivery_failure = None
        def check_delivery(phase):
            nonlocal delivery_failure
            reason = ('service_stopping' if self.stop_event.is_set() else
                      'deadline_exhausted' if time.monotonic() >= deadline else None)
            if reason and delivery_failure is None:
                delivery_failure = reason
                had_solution = bool(response['solution'])
                response['solution'] = ''
                if response['trace'] and not response['trace'].endswith('\n'):
                    response['trace'] += '\n'
                response['trace'] += _trace_event(reason, phase=phase, discarded_solution=had_solution)
                return True
            return False
        def persist_response():
            if workspace is None:
                return
            try:
                (workspace / 'response_trace.jsonl').write_bytes(response['trace'].encode('utf-8'))
                (workspace / 'response.json').write_text(json.dumps({
                    'task_id': response['task_id'], 'elapsed_s': response['elapsed_s'],
                    'elapsed_scope': 'Snapshot before this summary write; external response wall time is authoritative',
                    'delivery_failure': delivery_failure,
                    'has_solution': bool(response['solution']),
                    'solution_sha256': hashlib.sha256(response['solution'].encode('utf-8')).hexdigest(),
                    'trace_sha256': hashlib.sha256(response['trace'].encode('utf-8')).hexdigest(),
                }, ensure_ascii=False, indent=2), encoding='utf-8')
            except OSError:
                LOG.warning('Could not persist response summary')
        try:
            # Retain sources, trace targets and EDA logs for reproduction. This
            # local scratch is never read as context by subsequent requests.
            workspace = Path(tempfile.mkdtemp(
                prefix="hls-" + digest.hex()[:12] + "-", dir=self.work_root)).resolve()
            input_dir, output_dir = workspace / "input", workspace / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "prompt.txt").write_bytes(request["prompt"].encode('utf-8'))
            (input_dir / "interface.txt").write_bytes(request["interface"].encode('utf-8'))
            config_path = workspace / "model_config.json"
            config_path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
            (workspace / "request.json").write_text(json.dumps({
                "task_id": request["task_id"], "nonce_sha256": digest.hex(),
                "mode": request["mode"], "deadline_s": request["deadline_s"],
                "received_at": time.time() - max(0, time.monotonic() - started),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            acquired = self._acquire_compute(deadline - reserve)
            if self.stop_event.is_set():
                response["trace"] = _trace_event("service_stopping")
                return response
            if not acquired or time.monotonic() >= deadline - reserve:
                response["trace"] = _trace_event("deadline_exhausted", phase="queue")
                return response
            if not self._supervision_ok:
                response["trace"] = _trace_event("service_unavailable", reason="process_cleanup")
                return response
            seconds = deadline - time.monotonic() - reserve
            if seconds <= 0:
                response["trace"] = _trace_event("deadline_exhausted", phase="input")
                return response
            if self._runner is None:
                env = os.environ.copy()
                env.update(AGENT_CONFIG=str(config_path), LLM_BASE_URL=self.config["base_url"],
                           MODEL_NAME=self.config["model"])
                result = run_subprocess(request, input_dir, output_dir, seconds,
                                        root=self.root, env=env, stop_event=self.stop_event)
            else:
                result = self._runner(request, input_dir, output_dir, seconds)
            result = result or {}
            if result.get("cleanup_errors"):
                self._supervision_ok = False
                response["trace"] = _trace_event("service_unavailable", reason="process_cleanup")
                return response
            trace = _read_artifact(output_dir / "trace.jsonl", workspace)
            if any(not isinstance(json.loads(line), dict) for line in trace.split('\n') if line.strip()):
                raise ValueError("Trace records must be JSON objects")
            response["trace"] = trace or _trace_event(
                "solver_finished", returncode=result.get("returncode"),
                timeout=result.get("timeout", False), cancelled=result.get("cancelled", False))
            if (result.get("timeout") or result.get("cancelled") or result.get("returncode") != 0
                    or self.stop_event.is_set() or time.monotonic() >= deadline):
                return response
            code = _read_artifact(output_dir / "solution.cpp", workspace)
            if code.strip() and trace.strip():
                response["solution"] = code
        except Exception as exc:
            LOG.warning("Solver returned no solution (%s)", type(exc).__name__)
            response["solution"] = ""
            response["trace"] = _trace_event("solver_failure", error=type(exc).__name__)
        finally:
            if acquired:
                self._compute_lock.release()
            check_delivery('artifact_read')
            response["elapsed_s"] = round(max(0.0, time.monotonic() - started), 4)
            persist_response()
            # Slow/failed filesystem operations may use the response reserve.
            # Do not return stale code; rewrite changed delivery evidence once.
            if check_delivery('response_evidence'):
                response['elapsed_s'] = round(max(0.0, time.monotonic() - started), 4)
                persist_response()
            response['elapsed_s'] = round(max(0.0, time.monotonic() - started), 4)
        return response


class _DeadlineReader:
    """Bound the entire HTTP header/body read, including slow trickling peers."""
    def __init__(self, connection, deadline):
        self.connection, self.deadline = connection, deadline
        self.buffer = bytearray()
        self.eof = False

    def _receive(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP request read deadline")
        self.connection.settimeout(remaining)
        data = self.connection.recv(65536)
        self.buffer.extend(data)
        self.eof = not data

    def readline(self, size=-1):
        while True:
            index = self.buffer.find(b"\n")
            count = index + 1 if index >= 0 else len(self.buffer)
            if size >= 0:
                count = min(count, size)
            if index >= 0 or self.eof or (size >= 0 and len(self.buffer) >= size):
                result = bytes(self.buffer[:count])
                del self.buffer[:count]
                return result
            self._receive()

    def read(self, size=-1):
        if size < 0:
            raise ValueError("HTTP reads require an explicit bound")
        while len(self.buffer) < size and not self.eof:
            self._receive()
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def close(self):
        self.buffer.clear()


class CompetitionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32
    request_read_seconds = 10.0

    def __init__(self, address, service):
        if address[0] != "127.0.0.1":
            raise ValueError("Competition endpoint must bind 127.0.0.1")
        self.service = service
        self._request_slots = threading.BoundedSemaphore(32)
        super().__init__(address, CompetitionHandler)

    def process_request(self, request, client_address):
        if self.service.stop_event.is_set() or not self._request_slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                body = b'{"error":"Too many pending requests"}'
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    b"Connection: close\r\nContent-Length: " + str(len(body)).encode("ascii")
                    + b"\r\n\r\n" + body)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class CompetitionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HLSCompetition/1"

    def setup(self):
        self.received_at = time.monotonic()
        super().setup()
        self.rfile.close()
        self.rfile = _DeadlineReader(
            self.connection, self.received_at + self.server.request_read_seconds)

    def log_message(self, format, *args):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.close_connection = True
        try:
            self.connection.settimeout(5)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass

    def _authorize(self):
        values = self.headers.get_all("Authorization", [])
        if len(values) != 1 or not self.server.service.authorized(values[0]):
            self._reply(401, {"error": "Invalid bearer token"})
            return False
        return True

    def do_GET(self):
        if not self._authorize():
            return
        if self.path != "/v1/health":
            self._reply(404, {"error": "Unknown endpoint"})
            return
        self._reply(200, self.server.service.health())

    def do_POST(self):
        if not self._authorize():
            return
        if self.path != "/v1/solve":
            self._reply(404, {"error": "Unknown endpoint"})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise RequestError(415, "Content-Type must be application/json")
            lengths = self.headers.get_all("Content-Length", [])
            if (len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0])
                    or self.headers.get("Transfer-Encoding")):
                raise RequestError(400, "A single Content-Length is required")
            length = int(lengths[0])
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise RequestError(413, "Request body exceeds limit or is empty")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise RequestError(400, "Incomplete request body")
            data = json.loads(raw.decode("utf-8"), object_pairs_hook=_object)
            response = self.server.service.solve(data, received_at=self.received_at)
            self._reply(200, response)
        except RequestError as exc:
            self._reply(exc.status, {"error": str(exc)})
        except (ValueError, UnicodeError):
            self._reply(400, {"error": "Malformed JSON request"})
        except OSError:
            self._reply(408, {"error": "Request body timeout"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    from agent.competition import competition_config
    try:
        service = CompetitionService(competition_config(), os.environ.get("FPGACHINA_TOKEN", ""))
        server = CompetitionHTTPServer(("127.0.0.1", args.port), service)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("HLS endpoint listening on 127.0.0.1:" + str(args.port), flush=True)
    previous_handlers = {}
    def stop(signum, frame):
        service.request_stop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop)
    server.timeout = 0.2
    try:
        # handle_request, unlike shutdown() from serve_forever's own thread,
        # lets the signal handler stop acceptance without a shutdown deadlock.
        while not service.stop_event.is_set():
            server.handle_request()
    finally:
        service.request_stop()
        server.server_close()
        if not service.wait_stopped(timeout=8.5):
            LOG.error("Shutdown deadline: process cleanup could not be confirmed")
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
