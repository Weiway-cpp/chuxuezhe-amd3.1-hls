"""Fast local C++ compile+run of a design against a testbench (no Vitis).

This is the HLS analogue of a small local simulator: a candidate and its
(self-)testbench are compiled with a host C++ compiler and executed in a
scratch directory, so a functional failure can be reported in seconds
instead of waiting for a full Vitis run.

Scope and honesty rules:
  * It is only used to *reject* a candidate early; every accepted candidate is
    still judged by Vitis. A fast pass never becomes a success verdict.
  * If the sources need HLS-specific headers (ap_int.h, hls_stream.h, ...) or
    no host compiler exists, it reports ``unsupported`` and the caller falls
    back to Vitis unchanged.
  * No task answer, expected value or judge data is read here.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# Headers that only the Vitis front-end understands; a host compiler cannot
# reproduce them, so the fast lane must stay out of the way.
_HLS_ONLY = re.compile(
    r'ap_(?:int|fixed|float|uint|ufixed)\.h|hls_stream\.h|hls_video\.h|'
    r'hls_axi|ap_axi_sdata|hls_fft|hls_fpo\.h|ap_shift_reg\.h', re.I)

_INTERESTING = re.compile(
    r'Test\s+(?:case|Case)\s+\d+|FAIL\s+case|Expected|Actual|'
    r'AddressSanitizer|runtime error|Assertion|AssertionError|'
    r'segmentation|SIGSEGV|SIGABRT|error:|undefined reference', re.I)

MAX_DIAGNOSTIC_LINES = 12
MAX_DIAGNOSTIC_CHARS = 700


def find_compiler():
    """Return a host C++ compiler path, or None."""
    candidates = [os.environ.get('HLS_FAST_CXX'), 'g++', 'clang++', 'c++']
    for name in candidates:
        if not name:
            continue
        found = shutil.which(name)
        if found:
            return found
    return None


def _copy_tree(source, testbench, support_files, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, workdir / 'candidate.cpp')
    shutil.copyfile(testbench, workdir / 'testbench.cpp')
    for name, path in (support_files or {}).items():
        target = workdir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return workdir


def _compile(compiler, workdir, timeout, sanitize):
    flags = ['-O2', '-std=c++14', '-I.', '-w']
    if sanitize:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    command = [compiler, *flags, 'candidate.cpp', 'testbench.cpp', '-o', 'fast_csim', '-lm']
    started = time.monotonic()
    try:
        proc = subprocess.run(command, cwd=str(workdir), capture_output=True,
                              text=True, encoding='utf-8', errors='replace',
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, 'compile_timeout', '', time.monotonic() - started
    except OSError:
        return None, 'compile_unavailable', '', time.monotonic() - started
    output = (proc.stderr or '') + (proc.stdout or '')
    if proc.returncode != 0 and sanitize and re.search(
            r'unrecognized command[- ]line option|unknown argument|'
            r'unsupported option|ignoring unknown|cannot find -lasan|'
            r'cannot find -lubsan|libasan|libubsan|sanitizer', output, re.I):
        return None, 'retry_without_sanitizer', output, time.monotonic() - started
    return proc.returncode, 'compiled' if proc.returncode == 0 else 'compile_failed', output, time.monotonic() - started


def _excerpt(text, limit=MAX_DIAGNOSTIC_LINES):
    lines = []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        if _INTERESTING.search(line):
            lines.append(line[:MAX_DIAGNOSTIC_CHARS])
        if len(lines) >= limit:
            break
    if not lines:
        tail = [row.strip() for row in (text or '').splitlines() if row.strip()][-8:]
        lines = [row[:MAX_DIAGNOSTIC_CHARS] for row in tail]
    return lines


def run_fast(source, testbench, top, support_files, workdir, timeout=60.0,
             compiler=None, sanitize=True):
    """Compile and run ``source`` with ``testbench``; never raises on tool issues.

    Returns a dict with ``status`` in
    ``passed | compile_failed | test_failed | unsupported``.
    A caller may early-reject on ``compile_failed``/``test_failed``; every other
    status means "no opinion" and the normal verifier must still run.
    """
    started = time.monotonic()
    base = {'fast_lane': True, 'scope': 'host_compiler_csim_not_official',
            'top': top, 'sanitizers': bool(sanitize)}
    compiler = compiler or find_compiler()
    if compiler is None:
        return dict(base, status='unsupported', reason='no host C++ compiler found')
    try:
        work = _copy_tree(source, testbench, support_files, workdir)
    except OSError as exc:
        return dict(base, status='unsupported', reason='copy failed: ' + type(exc).__name__)
    code, phase, output, seconds = _compile(compiler, work, timeout, sanitize)
    if phase == 'retry_without_sanitizer':
        code, phase, output, seconds = _compile(compiler, work, timeout, False)
        base['sanitizers'] = False
    base['compiler'] = compiler
    base['compile_seconds'] = round(seconds, 3)
    if phase in ('compile_timeout', 'compile_unavailable'):
        return dict(base, status='unsupported', reason=phase)
    if code != 0:
        if _HLS_ONLY.search(output):
            return dict(base, status='unsupported', reason='HLS-only headers')
        return dict(base, status='compile_failed', compile_pass=False,
                    csim_pass=False, synthesis_pass=False,
                    failure_kind='compile', diagnostics=_excerpt(output),
                    wall_seconds=round(time.monotonic() - started, 3))
    exe = work / ('fast_csim.exe' if os.name == 'nt' else 'fast_csim')
    try:
        run = subprocess.run([str(exe)], cwd=str(work), capture_output=True,
                             text=True, encoding='utf-8', errors='replace',
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        # A slow host run does not prove the design is wrong; let Vitis judge it.
        return dict(base, status='unsupported', reason='host execution timeout',
                    wall_seconds=round(time.monotonic() - started, 3))
    except OSError as exc:
        return dict(base, status='unsupported', reason='run failed: ' + type(exc).__name__)
    text = (run.stderr or '') + (run.stdout or '')
    wall = round(time.monotonic() - started, 3)
    if run.returncode == 0:
        return dict(base, status='passed', compile_pass=True, csim_pass=True,
                    synthesis_pass=False, failure_kind=None,
                    diagnostics=[], wall_seconds=wall, returncode=0)
    return dict(base, status='test_failed', compile_pass=True, csim_pass=False,
                synthesis_pass=False, failure_kind='functional',
                diagnostics=_excerpt(text), wall_seconds=wall,
                returncode=run.returncode)
