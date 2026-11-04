"""Preserve case/value associations from this invocation's local CSIM output."""
import hashlib
from pathlib import Path
import re

from agent.repair_context import repair_diagnostics, _CASE, _ERROR

MAX_STREAM_BYTES = 128 * 1024
_CRASH = re.compile(r'segmentation (?:fault|violation)|access violation|bus error|floating point exception', re.I)


def preserve_counterexamples(evaluation):
    """Read only the verifier's local stdout/stderr, never task/judge files.

    The generic diagnostic selector prioritizes individual lines; it may have
    separated case headers from values. Recover contiguous records BEFORE any
    additional compaction, retaining original diagnostics and provenance.
    Failure to read optional excerpts must not change the verification verdict.
    """
    if evaluation.get('status') not in ('test_failed', 'csim_failed'):
        return evaluation
    previous = evaluation.get('diagnostics', [])
    if any(_CRASH.search(str(row)) for row in previous) or not evaluation.get('run_dir'):
        return evaluation
    records, evidence = [], {}
    try:
        root = Path(evaluation['run_dir']).resolve()
        for name in ('csim.stderr.txt', 'csim.stdout.txt'):
            path = root/'evidence'/name
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            with path.open('rb') as stream:
                raw = stream.read(MAX_STREAM_BYTES+1)
            truncated = len(raw) > MAX_STREAM_BYTES
            raw = raw[:MAX_STREAM_BYTES]
            # Never manufacture a record from a partial final line.
            if truncated:
                raw = raw.rsplit(b'\n', 1)[0] if b'\n' in raw else b''
            rows = raw.decode('utf-8', errors='replace').splitlines()
            if any(_ERROR.search(row) for row in rows):
                return evaluation  # Crash/compiler diagnostics retain precedence.
            if not any(_CASE.search(row) for row in rows):
                continue
            records.extend(repair_diagnostics(rows, evaluation['status']))
            evidence['evidence/'+name] = {'excerpt_sha256':hashlib.sha256(raw).hexdigest(),
                                         'excerpt_bytes':len(raw),'truncated':truncated}
    except (OSError, ValueError):
        return evaluation
    if not records:
        return evaluation
    return dict(evaluation, diagnostics=repair_diagnostics(records, evaluation['status']),
                original_diagnostics=previous, local_counterexample_evidence=evidence)
