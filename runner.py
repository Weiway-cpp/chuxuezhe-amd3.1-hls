"""Bounded generation/verification loop with separate baseline evaluation."""
import hashlib
from functools import partial
import json
import math
import shutil
from pathlib import Path
import re
import time
import uuid

from agent.model import ChatClient, ModelError
from agent.policy import prepare_messages, extract_code, result_rank, compact_diagnostics
from scripts.verify_hls import verify, validate_tool_version
from scripts.task_files import relative_file, validate_support, validate_oracle_snapshot


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def model_attempts(attempts):
    """Count dispatched model calls, including failures, not preparation records.

    Older records lacked model_called; retain their prior interpretation except
    for statuses which explicitly establish that no call occurred.
    """
    return [attempt for attempt in attempts if attempt.get('model_called',
        attempt.get('status') not in ('context_error', 'not_generated', 'generating'))]


def load_task(path, require_testbench=True, *, load_expected_stdout=True):
    path = Path(path).resolve()
    task = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(task, dict):
        raise ValueError("Task must be a JSON object")
    for key in ("id", "top", "description") + (("testbench",) if require_testbench else ()):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError("Task requires nonempty " + key)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task["top"]):
        raise ValueError("Invalid top function name")
    if require_testbench:
        task["_testbench"] = str(relative_file(path.parent, task['testbench']))
        names = task.get('support_files', [])
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError('support_files must list unique relative paths')
        task['_support_files'] = validate_support({n: relative_file(path.parent, n) for n in names})
        if load_expected_stdout and task.get('expected_stdout'):
            task['_expected_stdout'] = str(relative_file(path.parent, task['expected_stdout']))
        if load_expected_stdout and task.get('expected_stderr'):
            task['_expected_stderr'] = str(relative_file(path.parent, task['expected_stderr']))
    return task


def run_policy(task, client, tool, directory, *, baseline, max_attempts,
               total_seconds, tool_timeout, generation_timeout, reserve_seconds,
               verifier=verify, generation_only=False, use_skills=True,
               prompt_char_limit=None, trace=None, deadline_monotonic=None,
               initial_request=None, enable_patches=False, independent_final_repair=False,
               seed_source=None):
    # ``seed_source`` starts the attempt budget from a candidate produced
    # outside this loop (the opt-in best-of-K portfolio). The seed occupies
    # attempt 1 and is verified exactly like a generated candidate; the first
    # model call then buys a repair instead of the initial sample, so the
    # budget is unchanged while the starting point is stronger.
    if seed_source is not None and (baseline or generation_only):
        seed_source = None
    from agent.repair_context import repair_diagnostics
    from agent.design_patch import patch_messages, apply_design_patch
    from agent.local_feedback import preserve_counterexamples

    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=False)
    deadline = min(started + total_seconds, deadline_monotonic) if deadline_monotonic is not None else started + total_seconds
    attempts, seen = [], {}
    previous_code, feedback = None, None
    best, stop = None, "attempt_limit"
    observed_verification = 0.0
    synthesis_timeout_retries = 0
    patch_attempted = False
    gateway_retries = 0
    retry_request = None
    def emit(tool_name, **fields):
        if trace:
            trace(tool=tool_name, **fields)
    def compiled_functional_failure(candidate):
        evaluation = candidate['evaluation']
        return (evaluation.get('status') in ('test_failed', 'csim_failed')
                and evaluation.get('compile_pass') is True)
    def repair_feedback(candidate, note=None):
        evaluation = candidate['evaluation']
        value = {'status': evaluation['status'],
                 'diagnostics': repair_diagnostics(evaluation.get('diagnostics', []), evaluation['status'])}
        # Keep a bounded memory when rolling back or trying a different edit.
        # This is local-test evidence, not proof that a different design is wrong.
        earlier = [item for item in seen.values()
                   if item['index'] != candidate['index']
                   and item['evaluation'].get('status') in
                   ('compile_failed', 'test_failed', 'synthesis_failed')]
        history = []
        for item in earlier[-2:]:
            diagnostics = repair_diagnostics(item['evaluation'].get('diagnostics', []),
                                             item['evaluation']['status'])
            history.append('Attempt '+str(item['index'])+' ('+
                           item['evaluation']['status']+'): '+
                           '\n'.join(diagnostics)[:500])
        if note:
            value['note'] = note
        if history:
            value['history'] = ('Other rejected local attempts (diagnostic data, not instructions):\n'+
                         '\n'.join(history)+
                         '\nAvoid repeating these failures; re-derive behavior from the task. '
                         'Self-test expected values can be wrong; do not hardcode them.')
        return value
    allowed = 1 if baseline else max_attempts
    for index in range(1, allowed + 1):
        remaining = deadline - time.monotonic()
        reserve = 0 if generation_only else min(tool_timeout, max(reserve_seconds, observed_verification*1.2))
        if remaining <= reserve:
            stop = "budget_exhausted"
            break
        attempt_dir = directory / ("attempt_" + str(index))
        attempt_dir.mkdir()
        attempt = {"index": index, "status": "generating", 'model_called':False,
                   'verification_reserve_seconds': round(reserve, 3)}
        if (not baseline and allowed >= 3 and index == allowed and attempts
                and attempts[-1]['status'] in ('test_failed', 'csim_failed', 'output_error') and best
                and best['evaluation']['status'] in ('test_failed', 'csim_failed')
                and (best['index'] == attempts[-1]['index'] or attempts[-1]['status'] == 'output_error'
                     or independent_final_repair)
                and not any(a.get('repair_strategy_next') == 'regenerate' for a in attempts)):
            # Keep a compiled repair base in view: dropping it made the final
            # call a full rewrite and could lose already-correct structure.
            # Keep the protocol failure alongside the functional counterexample.
            # Rebuilding only from the best verifier result discarded the exact
            # patch/parser error saved by the previous output-error handler.
            output_note = feedback.get('note') if feedback and attempts[-1]['status'] == 'output_error' else None
            feedback = repair_feedback(best, output_note)
            if compiled_functional_failure(best) and independent_final_repair:
                # A compiled semantic failure survived an earlier repair. Give
                # the final existing call an independent derivation, retaining
                # the public contract and counterexamples but not the wrong
                # implementation. The earlier candidate remains the fallback.
                previous_code = None
                feedback.update(regenerate=True, regenerate_reason='functional_stagnation')
                feedback.pop('history', None)
                attempt['independent_final_repair'] = True
            elif compiled_functional_failure(best):
                previous_code = best['source'].read_bytes().decode('utf-8')
                feedback.update(focused_repair=True, focused_repair_reason='final_semantic_recheck')
            else:
                previous_code = None
                feedback.update(regenerate=True, regenerate_reason='final_semantic_recheck')
            if attempts[-1]['status'] == 'output_error':
                feedback['output_retry'] = True
            attempt['generation_strategy'] = 'final_semantic_recheck'
        if feedback and feedback.get('focused_repair') is True:
            attempt['focused_repair_reason'] = feedback['focused_repair_reason']
        attempts.append(attempt)
        patch_base = None
        try:
            initial = initial_request if index == 1 and not baseline else None
            if retry_request is not None:
                messages, skills, context, initial, retry_generate = retry_request
                retry_request = None
                attempt['generation_strategy'] = 'retry_gateway_request'
                if initial is not None:
                    attempt['generation_profile'] = initial['generation_profile']
            elif initial is not None:
                messages = initial['messages']
                size = sum(len(row['content']) for row in messages)
                if prompt_char_limit is not None and size > prompt_char_limit:
                    raise ValueError('Complete initial task exceeds max_prompt_chars; task was not truncated')
                skills, context = {}, {'characters':size, 'original_characters':size,
                    'complete_task_preserved':True, 'previous_code_included':False, 'actions':[]}
                attempt['generation_profile'] = initial['generation_profile']
            else:
                if (enable_patches and not baseline and not patch_attempted and previous_code
                        and feedback and feedback.get('status') in
                        ('compile_failed', 'test_failed', 'csim_failed', 'synthesis_failed')
                        and not feedback.get('output_retry') and not feedback.get('regenerate')):
                    try:
                        messages, skills, context = patch_messages(task['description'], previous_code,
                            feedback, max_prompt_chars=prompt_char_limit, use_skills=use_skills)
                        patch_base = previous_code
                        attempt.update(output_protocol='exact_edits_v1', patch_base_sha256=context['base_sha256'])
                    except ValueError:
                        attempt['patch_skipped'] = 'complete_base_exceeds_prompt_budget'
                if patch_base is None:
                    messages, skills, context = prepare_messages(
                        task["description"], baseline=baseline, code=previous_code, feedback=feedback,
                        use_skills=use_skills, max_prompt_chars=prompt_char_limit)
        except ValueError as exc:
            attempt.update(status='context_error', diagnostics=[str(exc)])
            stop = 'context_error'
            save_json(attempt_dir/'attempt.json', attempt)
            emit('context', rc=1, excerpt=str(exc), round=index)
            break
        save_json(attempt_dir/"request.json", {
            "messages": messages,
            'context': context,
            "skill_sha256": {name: hashlib.sha256(body.encode()).hexdigest()
                             for name, body in skills.items()},
            **({'generation_profile':initial['generation_profile'], 'sampling':initial['sampling']}
               if initial is not None else {}),
        })
        save_json(attempt_dir/"attempt.json", attempt)
        # Prompt preparation and evidence writes also consume the wall budget.
        # Never start a model worker using a stale pre-preparation timeout.
        remaining = deadline - time.monotonic()
        if remaining <= reserve:
            attempt['status'] = 'not_generated'
            stop = 'budget_exhausted'
            save_json(attempt_dir/'attempt.json', attempt)
            break
        generation_started = time.time()
        call_timeout = min(generation_timeout, remaining - reserve)
        if initial is not None:
            call_timeout = min(call_timeout, initial['timeout_cap_s'])
        try:
            if seed_source is not None and index == 1 and not baseline:
                # Portfolio seed: no model call at all. The response record is
                # synthetic and marked as such so evidence never implies a
                # generation that did not happen.
                attempt.update(model_called=False, generation_strategy='portfolio_seed',
                               generation_profile='portfolio_best_of_k',
                               generation_seconds=0.0, usage={})
                response = {'content': seed_source, 'finish_reason': 'stop',
                            'wall_seconds': 0.0, 'usage': {}, 'synthetic': 'portfolio_seed'}
                save_json(attempt_dir/'response.json', response)
                emit('portfolio', event='seed_adopted', round=index,
                     chars=len(seed_source),
                     response_sha256=hashlib.sha256(seed_source.encode()).hexdigest())
            else:
                attempt['model_called'] = True
                generate = client.generate
                if attempt.get('generation_strategy') == 'retry_gateway_request':
                    generate = retry_generate
                elif patch_base is not None:
                    patch_attempted = True
                    generate = getattr(client, 'generate_patch', client.generate)
                elif initial is not None and callable(getattr(client, 'generate_initial', None)):
                    generate = client.generate_initial
                elif not baseline and feedback and feedback.get('regenerate') is True:
                    diversified = getattr(client, 'generate_regenerated', None)
                    if callable(diversified):
                        generate = diversified
                        attempt['sampling_strategy'] = 'regeneration'
                response = generate(messages, timeout=call_timeout)
                save_json(attempt_dir/"response.json", response)
                attempt["generation_seconds"] = response.get("wall_seconds", 0)
                attempt["usage"] = response.get("usage", {})
                emit('llm', ts=generation_started, round=index, rc=0,
                     tokens_in=attempt['usage'].get('prompt_tokens'),
                     tokens_out=attempt['usage'].get('completion_tokens'),
                     finish=response.get('finish_reason'), sec=attempt['generation_seconds'],
                     timeout_s=call_timeout, response_path=str(attempt_dir/'response.json'),
                     response_sha256=hashlib.sha256(response['content'].encode()).hexdigest())
        except ModelError as exc:
            attempt["status"] = "model_error"
            attempt["diagnostics"] = [str(exc)]
            attempt['failure_kind'] = exc.failure_kind
            stop = 'budget_exhausted' if exc.failure_kind == 'budget_exhausted' else 'model_error'
            save_json(attempt_dir/"attempt.json", attempt)
            emit('llm', ts=generation_started, round=index, rc=1, excerpt=str(exc),
                 sec=time.time()-generation_started, timeout_s=call_timeout,
                 failure_kind=exc.failure_kind, diagnostic=exc.diagnostic)
            diagnostic = exc.diagnostic if isinstance(exc.diagnostic, dict) else {}
            categories = diagnostic.get('categories', [])
            retryable = (exc.failure_kind is None
                         and type(diagnostic.get('http_status')) is int
                         and diagnostic.get('http_status') in (502, 503, 504)
                         and isinstance(categories, list) and all(isinstance(item, str) for item in categories)
                         and set(categories) <= {'connection_failed', 'backend_timeout'}
                         and diagnostic.get('error_type') in (None, 'server_error', 'internal_server_error')
                         and diagnostic.get('error_code') in (None, 'server_error', 'internal_server_error')
                         and 'process_exit_code' not in diagnostic)
            # One structured gateway recovery consumes the next existing design
            # call. Preserve the exact request/profile; no hidden HTTP retries,
            # no extension of the deadline, and never a second patch dispatch.
            if (retryable and not baseline and patch_base is None and gateway_retries == 0
                    and attempt.get('sampling_strategy') != 'regeneration'
                    and index < allowed and deadline-time.monotonic() > reserve+min(30, generation_timeout)):
                gateway_retries += 1
                retry_request = (messages, skills, context, initial, generate)
                attempt['recovery_next'] = 'same_request_within_existing_attempt_budget'
                save_json(attempt_dir/'attempt.json', attempt)
                stop = 'attempt_limit'
                continue
            break
        if time.monotonic() >= deadline:
            attempt.update(status='timeout', diagnostics=['Model response arrived after the policy deadline'])
            stop = 'budget_exhausted'
            save_json(attempt_dir/'attempt.json', attempt)
            break
        try:
            if response.get("finish_reason") == "length":
                (attempt_dir/'truncated.txt').write_bytes(response['content'].encode('utf-8'))
                raise ValueError("Generation reached max_tokens; source may be incomplete")
            if patch_base is not None:
                code, proposal = apply_design_patch(patch_base, response['content'], attempt['patch_base_sha256'])
                save_json(attempt_dir/'patch.json', proposal)
                attempt['patch_applied'] = True
            else:
                code = extract_code(response["content"])
        except ValueError as exc:
            attempt.update(status="output_error", diagnostics=[str(exc)])
            if best:
                previous_code = best['source'].read_bytes().decode('utf-8')
                feedback = repair_feedback(best, 'The last response was unusable: ' + str(exc) +
                    '. Continue repairing this earlier candidate and its recorded failure; '
                    'return one complete source file.')
                feedback['output_retry'] = True
                attempt['repair_base_next'] = best['index']
            else:
                feedback = dict(attempt)
                previous_code = None
            save_json(attempt_dir/"attempt.json", attempt)
            continue
        previous_code = code
        source = attempt_dir/"candidate.cpp"
        # Hash, compile and deliver the same UTF-8 bytes on every host; text-mode
        # writes on Windows would translate LF (and turn CRLF into CRCRLF).
        source.write_bytes(code.encode('utf-8'))
        digest = hashlib.sha256(code.encode()).hexdigest()
        attempt["source_sha256"] = digest
        attempt['output_format_pass'] = True
        if time.monotonic() >= deadline:
            attempt.update(status='not_verified', diagnostics=['Source preparation exhausted the policy budget'])
            stop = 'budget_exhausted'
            save_json(attempt_dir/'attempt.json', attempt)
            break
        if generation_only:
            attempt['status'] = 'generated'
            best = {'index': index, 'source': source,
                    'evaluation': {'status': 'not_evaluated'}}
            stop = 'single_generation'
            save_json(attempt_dir/'attempt.json', attempt)
            break
        if digest in seen:
            cached = seen[digest]
            attempt.update(status="duplicate_candidate", duplicate_of=cached['index'],
                           evaluation=cached['evaluation'], verification_cache_hit=True,
                           selection_decision='duplicate_not_reverified')
            repair = best if best and result_rank(best['evaluation']) > result_rank(cached['evaluation']) else cached
            feedback = repair_feedback(repair,
                'The last response repeated an already rejected implementation. Make a substantive correction; do not repeat it.')
            # Change the next prompt within the existing call/time budget. Keep
            # a strictly better repair base when the duplicate was a regression.
            focused = repair is cached and compiled_functional_failure(repair)
            regenerate = repair is cached and not focused
            previous_code = None if regenerate else repair['source'].read_bytes().decode('utf-8')
            if regenerate:
                feedback['regenerate'] = True
            elif focused:
                feedback.update(focused_repair=True, focused_repair_reason='duplicate_functional')
            attempt.update(repair_base_next=repair['index'],
                           repair_strategy_next=('regenerate' if regenerate else
                                                 'focused_repair' if focused else 'repair_best'))
            save_json(attempt_dir/"attempt.json", attempt)
            emit('cache', round=index, duplicate_of=cached['index'], source_sha256=digest,
                 repair_strategy_next=attempt['repair_strategy_next'])
            stop = 'attempt_limit'
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            attempt["status"] = "not_verified"
            stop = "budget_exhausted"
            save_json(attempt_dir/"attempt.json", attempt)
            break
        try:
            inputs = {}
            if task.get('_support_files'):
                inputs['support_files'] = task['_support_files']
            if task.get('_expected_stdout'):
                inputs['expected_stdout'] = task['_expected_stdout']
            if task.get('_expected_stderr'):
                inputs['expected_stderr'] = task['_expected_stderr']
            attempt['verification_called'] = True
            evaluation = verifier(
                source=source, testbench=task["_testbench"], top=task["top"], tool=tool,
                output_root=attempt_dir/"verification",
                timeout=min(tool_timeout, remaining), **inputs)
        except (OSError, ValueError) as exc:
            evaluation = {"status": "tool_error", "failure_kind": "infrastructure",
                          "diagnostics": ["Verifier error: " + str(exc)]}
        evaluation = preserve_counterexamples(evaluation)
        late_verification = time.monotonic() >= deadline
        if late_verification:
            evaluation = dict(evaluation, status='timeout', failure_kind='timeout',
                local_verifier_status=evaluation['status'], timeout_phase='controller_receipt',
                unaccepted_verifier_result=evaluation,
                compile_pass=False, csim_pass=False, synthesis_pass=False,
                diagnostics=['Verification result arrived after the policy deadline'])
        attempt.update(status=evaluation["status"], evaluation=evaluation)
        rank = result_rank(evaluation)
        previous_rank = result_rank(best['evaluation']) if best else None
        attempt.update(selection_rank=rank, previous_best_rank=previous_rank,
            selection_decision=('initial_recovery_candidate' if best is None else
                'higher_verified_stage' if rank > previous_rank else
                'same_grade_keep_earlier' if rank == previous_rank else 'lower_grade_keep_earlier'))
        seen[digest] = {'index': index, 'source': source, 'evaluation': evaluation}
        measured = evaluation.get('wall_seconds', 0)
        if (evaluation['status'] != 'timeout' and isinstance(measured, (int, float))
                and math.isfinite(measured) and measured > 0):
            observed_verification = max(observed_verification, measured)
        emit('vitis', round=index, rc=evaluation.get('returncode'), status=evaluation['status'],
             stages=evaluation.get('stages', {}), sec=measured,
             source_sha256=digest, run_dir=evaluation.get('run_dir'),
             excerpt='\n'.join(evaluation.get('diagnostics', []))[:2000])
        save_json(attempt_dir/"attempt.json", attempt)
        if best is None or result_rank(evaluation) > result_rank(best["evaluation"]):
            best = {"index": index, "source": source, "evaluation": evaluation}
        feedback = repair_feedback(seen[digest])
        if late_verification:
            stop = 'budget_exhausted'
            break
        if evaluation["status"] in ('passed', 'compile_only_passed'):
            stop = evaluation['status']
            break
        if (evaluation['status'] == 'timeout' and synthesis_timeout_retries == 0
                and evaluation.get('timeout_stage') == 'synthesis'
                and evaluation.get('timeout_phase') == 'execution'
                and evaluation.get('compile_pass') is True and evaluation.get('csim_pass') is True
                and evaluation.get('required_tool_version')
                and evaluation.get('tool_version') == evaluation.get('required_tool_version')
                and not evaluation.get('cleanup_errors') and not evaluation.get('stage_evidence_error')
                and not evaluation.get('infrastructure_error_detected') and not evaluation.get('launch_error')):
            # An execution timeout is not a synthesis verdict. Keep that evidence
            # intact, but permit one model-authored simplification of the design.
            # Preparation/cleanup/tool failures must still stop immediately.
            synthesis_timeout_retries += 1
            feedback = repair_feedback(seen[digest],
                'Local synthesis exceeded its time allocation; this does not prove a design error. '
                'Preserve the behavior that passed local C simulation. Remove unnecessary full '
                'array partitioning, full unrolling or forced pipeline constraints unless the '
                'public task requires them. Return a complete simpler implementation.')
            feedback.update(status='synthesis_failed', local_status='timeout')
            attempt['repair_strategy_next'] = 'simplify_synthesis'
            save_json(attempt_dir/'attempt.json', attempt)
            stop = 'timeout'
            continue
        if evaluation["status"] in ("tool_error", "timeout", "selftest_unusable"):
            stop = evaluation["status"]
            break
        if best and result_rank(evaluation) < result_rank(best['evaluation']):
            previous_code = best['source'].read_bytes().decode('utf-8')
            feedback = repair_feedback(best, 'The last edit regressed to ' + evaluation['status'] +
                                      '. Continue from this earlier candidate which passed more checks.')
            attempt['repair_base_next'] = best['index']
            save_json(attempt_dir/'attempt.json', attempt)
    result = {
        "policy": "baseline" if baseline else "agent",
        "status": "passed" if best and best["evaluation"]["status"] == "passed" else "failed",
        "stop_reason": stop, "wall_seconds": round(time.monotonic()-started, 3),
        "attempts": attempts, "selected_attempt": best["index"] if best else None,
        "selected_status": best["evaluation"]["status"] if best else None,
        "passed": bool(best and best["evaluation"]["status"] == "passed"),
        'verification_calls': sum(bool(a.get('verification_called')) for a in attempts),
        'duplicate_candidates': sum(a['status'] == 'duplicate_candidate' for a in attempts),
        'generation_count': len(model_attempts(attempts)),
        'use_skills': use_skills and not baseline,
        'selection_scope': 'Local checks only; equal grades do not establish functional improvement',
    }
    if best:
        final = directory/"final.cpp"
        final.write_bytes(best["source"].read_bytes())
        result["final_source"] = str(final)
    if generation_only:
        result.update(status='generated' if best else 'failed', passed=None,
                      evaluation_scope='Generation only; no tool invoked')
    save_json(directory/"result.json", result)
    return result


def score_baseline(result, task, tool, directory, total_seconds, tool_timeout, verifier):
    """External evaluator: called only AFTER the bare baseline has terminated."""
    started = time.monotonic()
    generation_seconds = result['wall_seconds']
    result['generation_wall_seconds'] = generation_seconds
    result['evaluation_scope'] = 'External score; no feedback or further baseline generation'
    result['passed'] = False
    result['status'] = 'failed'
    if result.get('final_source'):
        attempt = result['attempts'][0]
        attempt['verification_called'] = False
        remaining = total_seconds - generation_seconds
        if remaining <= 0:
            evaluation = {'status': 'timeout', 'diagnostics': ['No verification budget remains']}
        else:
            inputs = {}
            if task.get('_support_files'):
                inputs['support_files'] = task['_support_files']
            if task.get('_expected_stdout'):
                inputs['expected_stdout'] = task['_expected_stdout']
            if task.get('_expected_stderr'):
                inputs['expected_stderr'] = task['_expected_stderr']
            try:
                attempt['verification_called'] = True
                evaluation = verifier(source=result['final_source'], testbench=task['_testbench'],
                                      top=task['top'], tool=tool,
                                      output_root=directory/'attempt_1'/'verification',
                                      timeout=min(tool_timeout, remaining), **inputs)
            except (OSError, ValueError) as exc:
                evaluation = {'status': 'tool_error', 'diagnostics': ['Verifier error: ' + str(exc)]}
        attempt.update(status=evaluation['status'], evaluation=evaluation)
        save_json(directory/'attempt_1'/'attempt.json', attempt)
        result.update(passed=evaluation['status'] == 'passed',
                      selected_status=evaluation['status'], stop_reason=evaluation['status'])
        result['status'] = 'passed' if result['passed'] else 'failed'
    result['verification_calls'] = sum(bool(a.get('verification_called')) for a in result['attempts'])
    result['evaluation_seconds'] = round(time.monotonic()-started, 3)
    result['wall_seconds'] = round(generation_seconds + result['evaluation_seconds'], 3)
    save_json(directory/'result.json', result)
    return result


def run_session(task, config, tool, output_root, *, mode="compare", max_attempts=3,
                total_seconds=600, tool_timeout=180, generation_timeout=120,
                reserve_seconds=15, client=None, verifier=verify, generation_only=False,
                policy_order=('baseline', 'agent'), use_skills=True,
                required_tool_version='2025.2'):
    validate_tool_version(required_tool_version)
    verifier = partial(verifier, required_tool_version=required_tool_version)
    if mode not in ("baseline", "agent", "compare"):
        raise ValueError("Invalid mode")
    if tuple(policy_order) not in (('baseline', 'agent'), ('agent', 'baseline')):
        raise ValueError('policy_order must contain baseline and agent exactly once')
    if generation_only and mode != 'baseline':
        raise ValueError('generation_only is available only for the bare baseline')
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (
            total_seconds, tool_timeout, generation_timeout, reserve_seconds)):
        raise ValueError("Time budgets must be positive and finite")
    if reserve_seconds >= total_seconds:
        raise ValueError("Verification reserve must be less than total budget")
    run = Path(output_root).resolve()/(
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    task = dict(task)
    (run/"inputs").mkdir()
    manifest = {}
    if not generation_only:
        snapshot = run/'inputs'/'testbench.cpp'
        shutil.copy2(task['_testbench'], snapshot)
        task['_testbench'] = str(snapshot)
        manifest['testbench.cpp'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        supports = {}
        for name, path in task.get('_support_files', {}).items():
            dest = run/'inputs'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            supports[name] = str(dest)
            manifest[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        task['_support_files'] = supports
        if task.get('_expected_stdout'):
            dest = run/'inputs'/'expected.stdout.txt'
            shutil.copy2(task['_expected_stdout'], dest)
            task['_expected_stdout'] = str(dest)
            manifest['expected.stdout.txt'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if task.get('_expected_stderr'):
            dest = run/'inputs'/'expected.stderr.txt'
            shutil.copy2(task['_expected_stderr'], dest)
            task['_expected_stderr'] = str(dest)
            manifest['expected.stderr.txt'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        validate_oracle_snapshot(task, required_tool_version)
    save_json(run/"task.json", {k:v for k,v in task.items() if not k.startswith("_")})
    save_json(run/"inputs"/"manifest.json", manifest)
    public_config = {k:config[k] for k in (
        "base_url", "model", "max_tokens", "temperature", "context_tokens",
        "max_prompt_chars", "seed", "api_key_env") if k in config}
    save_json(run/"model_config.json", public_config)
    save_json(run/"budget.json", {
        "required_tool_version": required_tool_version,
        "mode": mode, "max_attempts": max_attempts,
        "total_seconds_per_policy": total_seconds, "tool_timeout": tool_timeout,
        "generation_timeout": generation_timeout, "reserve_seconds": reserve_seconds,
    })
    client = client or ChatClient(config)
    summary = {"task_id": task["id"], "run_dir": str(run), "policies": {},
               'required_tool_version': required_tool_version,
               'baseline_kind': 'legacy_development_baseline_not_official_script',
               'policy_order': list(policy_order) if mode == 'compare' else [mode],
               'use_skills': use_skills,
               "scope": "Local visible-test evaluation; not official scoring or pass@k."}
    policies = tuple(policy_order) if mode == "compare" else (mode,)
    for policy in policies:
        result = run_policy(
            task, client, tool, run/policy, baseline=policy == "baseline",
            max_attempts=max_attempts, total_seconds=total_seconds,
            tool_timeout=tool_timeout, generation_timeout=generation_timeout,
            reserve_seconds=reserve_seconds, verifier=verifier,
            generation_only=policy == 'baseline', use_skills=use_skills,
            prompt_char_limit=config.get('max_prompt_chars'))
        if policy == 'baseline' and not generation_only:
            result = score_baseline(result, task, tool, run/policy, total_seconds, tool_timeout, verifier)
        summary["policies"][policy] = {
            key:result[key] for key in (
                "status", "passed", "stop_reason", "wall_seconds",
                "selected_attempt", "selected_status")}
        summary["policies"][policy]["generation_count"] = result['generation_count']
        save_json(run/"summary.json", summary)
    if mode == "compare":
        summary["local_success_delta"] = (
            int(summary["policies"]["agent"]["passed"])
            - int(summary["policies"]["baseline"]["passed"]))
    summary["wall_seconds"] = round(time.monotonic()-started, 3)
    save_json(run/"summary.json", summary)
    return summary
