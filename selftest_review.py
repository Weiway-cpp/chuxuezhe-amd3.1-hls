"""One optional specification-only review, before any candidate is cached.

This is a model hypothesis, not an oracle. The external judge stays independent.
"""
import hashlib
import json
from pathlib import Path
import re
import time

from agent.model import ModelError
from agent.runner import save_json


REVIEW_MAX_TOKENS = 1024


def review_model_config(config):
    """Bound only the optional edit reply, leaving design generation unchanged."""
    return dict(config, max_tokens=min(REVIEW_MAX_TOKENS, config['max_tokens']))


def apply_review_edits(source, response, *, design_repair=False):
    """Apply unambiguous edits to the ORIGINAL text; not a semantic proof.

    No execution, fuzzy matching, cascade replacements or arbitrary file paths.
    A malformed/truncated proposal leaves the original testbench untouched.
    """
    if not isinstance(response, str) or len(response) > (64000 if design_repair else 12000):
        raise ValueError('Review response must be a bounded JSON object')
    text = response.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    if fence:
        text = fence[1]
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate review JSON key: ' + key)
            result[key] = value
        return result
    try:
        proposal = json.loads(text, object_pairs_hook=unique_keys)
    except RecursionError as exc:
        raise ValueError('Review JSON nesting is too deep') from exc
    if not isinstance(proposal, dict) or set(proposal) != {'reason', 'edits'}:
        raise ValueError('Review must contain only reason and edits')
    reason, edits = proposal['reason'], proposal['edits']
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError('Review needs a brief specification-based reason')
    if not isinstance(edits, list) or len(edits) > (32 if design_repair else 6):
        raise ValueError('Review allows at most six local edits')
    spans = []
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict) or set(edit) != {'old', 'new'}:
            raise ValueError('Each review edit must contain only old and new')
        old, new = edit['old'], edit['new']
        if (not isinstance(old, str) or not old.strip() or
                not isinstance(new, str) or not new.strip() or old == new):
            raise ValueError('Review edits need nonempty, different text')
        if '\x00' in old or '\x00' in new:
            raise ValueError('Review snippets cannot contain NUL')
        start = source.find(old)
        if start < 0 or source.find(old, start + 1) != -1:
            if design_repair:
                raise ValueError('Design edit %d old snippet must match exactly once; found %d matches. '
                                 'Include unique surrounding function/context text; no replacement was applied.'
                                 % (index, len(list(re.finditer('(?='+re.escape(old)+')', source)))))
            raise ValueError('Review old text must match exactly once in the original')
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('Review edits overlap in the original testbench')
    if not design_repair and sum(end-start for start, end, _ in spans) > len(source) // 2:
        raise ValueError('Review must edit local snippets, not rewrite the testbench')
    revised = source
    for start, end, new in reversed(spans):
        revised = revised[:start] + new + revised[end:]
    if len(revised) > (max(len(source)*3, 64000) if design_repair else max(len(source) * 2, 4000)):
        raise ValueError('Review expanded the testbench too far')
    return revised, proposal


def review_messages(description, source, max_prompt_chars):
    messages = [
        {'role': 'system', 'content':
         'Review this generated C++ testbench against the immutable task specification. '
         'You have no design implementation or measured outputs. Independently recompute '
         'expected results from the specification, including signedness, finite-width '
         'overflow, reset/state, array bounds and the valid input domain. A wider reference '
         'accumulator does not make its result representable in a narrower output. '
         'Check every intermediate sum and the output range. Signed C++ overflow is not '
         'implicitly modulo arithmetic. If overflow behavior is unspecified, replace an '
         'overflowing input combination with one that keeps every partial sum representable '
         'while retaining boundary operands, signs and independent checks. If the task '
         'specifies wrap or saturation, implement exactly that rule. Prefer a small '
         'reference calculation over hand-computed constants. Change only a demonstrable '
         'testbench error. Preserve valid test inputs, independent checks and coverage; '
         'never delete a failing case, weaken assertions or make success unconditional. '
         'If no error is established, return the no-change object below, including its reason. '
         'Include the supplied header, call the exact top directly in int main(), and '
         'keep explicit failure checks in main. Do not implement or replace the design. '
         'Do not use lambdas, local classes, conditional preprocessing or external files. '
         'Return exactly one JSON object with exactly two keys: reason and edits. '
         'reason must be a nonempty string; edits must be a list, even when empty. '
         'Do not return {}, [], {"edits":[]}, a status key, Markdown, or C++ source.\n'
         'No-change response:\n'
         '{"reason":"No specification-based testbench error established.","edits":[]}\n'
         'Edit response example (syntax illustration only; use only if the actual '
         'source has this error and the specification justifies the correction):\n'
         '{"reason":"Use comparison rather than assignment in the assertion.",'
         '"edits":[{"old":"assert(got = expected);","new":"assert(got == expected);"}]}\n'
         'Use at most six non-overlapping edits; each old snippet must occur exactly once '
         'in the original. Keep snippets short, with just enough context to be unique. '
         'Do not rewrite the whole file, output a diff, or include explanatory prose. '
         'Escape newlines and quotes as JSON strings. Preserve every valid assertion.'},
        {'role': 'user', 'content': description},
        {'role': 'user', 'content': 'Generated testbench to review:\n' + source},
    ]
    if sum(len(m['content']) for m in messages) > max_prompt_chars:
        raise ValueError('Complete task and self-test exceed review input budget')
    return messages


class SelftestReviewVerifier:
    def __init__(self, verifier, *, task, directory, enabled, deadline,
                 generate, screen, max_prompt_chars, trace):
        self.verifier = verifier
        self.task = task
        self.directory = Path(directory)
        self.enabled = enabled
        self.deadline = deadline
        self.generate = generate
        self.screen = screen
        self.max_prompt_chars = max_prompt_chars
        self.trace = trace
        self.calls = 0
        self.first = True
        self.summary = {'enabled': enabled, 'status': 'not_triggered' if enabled else 'disabled',
                        'scope': 'Specification-only model review; not correctness proof'}

    def _run(self, kwargs):
        self.calls += 1
        return self.verifier(**kwargs)

    def __call__(self, **kwargs):
        first, self.first = self.first, False
        started = time.monotonic()
        local_deadline = min(self.deadline, started + kwargs['timeout'])
        original = self._run(kwargs)
        initial_seconds = time.monotonic() - started
        origins = original.get('compile_errors', {}).get('origins', [])
        eligible = ((original.get('status') == 'test_failed' and original.get('compile_pass') is True)
                    or (original.get('status') == 'compile_failed' and set(origins) == {'selftest'}))
        if not first or not self.enabled or not eligible:
            return original
        # Only the first verification can change the testbench: run_policy has
        # no cached evaluations or best candidate yet. Later comparisons share
        # the committed testbench, including after a design repair.
        self.summary.update(status='skipped', reason='insufficient_budget')
        reserve = max(20.0, initial_seconds * 1.2)
        repair_reserve = 20.0
        room = local_deadline - time.monotonic() - reserve - repair_reserve
        review_seconds = min(75.0, max(0.0, room),
                             max(0.0, (self.deadline - time.monotonic()) * .3))
        self.summary.update(protocol='exact_edits_v1', model_timeout_s=round(review_seconds, 3))
        if review_seconds < 5:
            return original
        review_started = time.monotonic()
        recheck, revised_path = None, None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            source = Path(kwargs['testbench']).read_bytes().decode('utf-8')
            self.summary['original_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            save_json(self.directory/'original_evaluation.json', original)
            recheck, revised_path = self._propose(
                kwargs, source, review_started, review_seconds, local_deadline,
                reserve, repair_reserve)
        except (ModelError, ValueError, OSError) as exc:
            self.summary.update(status='rejected', reason=type(exc).__name__ + ': ' + str(exc))
        self.summary['wall_seconds'] = round(time.monotonic() - review_started, 3)
        if not self._save_summary():
            self.summary.update(status='rejected', reason='review_evidence_unavailable')
            revised_path = None
        try:
            self.trace(tool='selftest', event='review_finished', **self.summary)
        except OSError as exc:
            self.summary.update(status='rejected', reason='review_trace_unavailable',
                                evidence_error=type(exc).__name__ + ': ' + str(exc))
            revised_path = None
            self._save_summary()
        # Evidence and trace writes are part of the deadline too. Commit only
        # after those writes, so a late recheck cannot replace the earlier result.
        if revised_path is not None:
            if time.monotonic() >= local_deadline:
                self.summary.update(status='rejected', reason='review_finalization_timeout',
                                    wall_seconds=round(time.monotonic() - review_started, 3))
                self._save_summary()
                self.trace(tool='selftest', event='review_rejected_late', **self.summary)
            else:
                self.task['_testbench'] = str(revised_path)
                return recheck
        return original

    def _save_summary(self):
        try:
            save_json(self.directory/'review.json', self.summary)
            return True
        except OSError as exc:
            self.summary['evidence_error'] = type(exc).__name__ + ': ' + str(exc)
            return False

    def _propose(self, kwargs, source, review_started, review_seconds,
                 local_deadline, reserve, repair_reserve):
        messages = review_messages(self.task['description'], source, self.max_prompt_chars)
        save_json(self.directory/'request.json', {'messages': messages})
        review_deadline = min(review_started + review_seconds,
                              local_deadline - reserve - repair_reserve)
        response = self.generate(messages, review_seconds, deadline=review_deadline,
                                 response_path=self.directory/'response.json')
        if time.monotonic() >= review_deadline:
            raise ModelError('Review response arrived after its reserved deadline')
        revised, proposal = apply_review_edits(source, response)
        save_json(self.directory/'edits.json', proposal)
        self.summary.update(edit_count=len(proposal['edits']), model_reason=proposal['reason'])
        self.screen(revised)
        if revised == source:
            self.summary.update(status='unchanged', reason='review_kept_original')
            return None, None
        revised_path = self.directory/'reviewed_testbench.cpp'
        revised_path.write_bytes(revised.encode('utf-8'))
        self.summary['revised_sha256'] = hashlib.sha256(revised.encode()).hexdigest()
        remaining = local_deadline - time.monotonic() - repair_reserve
        if remaining < reserve:
            self.summary.update(status='skipped', reason='insufficient_reverification_budget')
            return None, None
        updated = dict(kwargs, testbench=revised_path,
                       output_root=self.directory/'verification', timeout=remaining)
        recheck = self._run(updated)
        save_json(self.directory/'revised_evaluation.json', recheck)
        self.trace(tool='selftest', event='reverification', status=recheck.get('status'),
                   run_dir=recheck.get('run_dir'), sec=recheck.get('wall_seconds'))
        # A completed failure is still a usable verdict. Fixing a testbench can
        # expose a DUT compile error which should enter the ordinary repair loop.
        # Never choose the old/new test merely because one gives a pass.
        design_compile_error = (recheck.get('status') == 'compile_failed' and
            set(recheck.get('compile_errors', {}).get('origins', [])) == {'design'})
        usable = design_compile_error or (recheck.get('compile_pass') is True and
            recheck.get('status') in ('passed', 'test_failed', 'csim_failed', 'synthesis_failed'))
        if time.monotonic() >= local_deadline or not usable:
            self.summary.update(status='rejected', reason='reverification_unavailable')
            return None, None
        self.summary.update(status='applied', reason='reviewed_testbench_reverified')
        return recheck, revised_path
