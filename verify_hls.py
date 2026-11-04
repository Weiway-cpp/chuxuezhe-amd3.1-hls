"""Local Vitis HLS development verifier, not the official competition scorer.

Stage evidence comes from the controlling Tcl process rather than candidate
stdout. Candidate code still executes with the verifier's OS privileges: this
module is not a security sandbox and should be run inside an isolated worker.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
try:
    from scripts.task_files import validate_support
    from scripts.diagnostics import has_infrastructure_error, select_diagnostics, compiler_error_locations
except ModuleNotFoundError:
    from task_files import validate_support
    from diagnostics import has_infrastructure_error, select_diagnostics, compiler_error_locations


PART = 'xczu3eg-sbva484-1-e'
CLOCK_NS = 5
CXX_FLAGS = '-I. -std=c++14'
# Vitis 2025.2 and 2026.1 Windows can omit user flags on the preprocessed-source
# syntax check. A diagnostic pragma survives preprocessing; command-line
# -Wno-error=register alone does not. Clang 16's pragma 'warning' also does
# not demote this default-error diagnostic; ignore only -Wregister instead.
# Keep all other diagnostics unchanged; original source bytes remain intact.
REGISTER_COMPAT_VERSIONS = frozenset({'2025.2', '2026.1'})
REGISTER_COMPAT = ('#if defined(__clang__) && defined(__SYNTHESIS__)\n'
                   '#pragma clang diagnostic ignored "-Wregister"\n'
                   '#endif\n')
REGISTER_WRAPPER = REGISTER_COMPAT + '#include "../candidate.cpp"\n'
# M_PI is a platform extension, not a C++14 constant. Only this reviewed,
# byte-identical upstream CORDIC testbench receives a portable fallback.
# Include math.h first so an implementation-provided M_PI remains unchanged.
CORDIC_TESTBENCH_SHA256 = 'dc283122ad24f651f39e4963928dd56d2bd92385ea7a0562fbbbc8386eb9799a'
CORDIC_HEADER_SHA256 = '4a979e06b131926c08d0a69fd2659f56b75286be9f0b4c688618ec9e47476bf9'
CORDIC_TESTBENCH_WRAPPER = ('#include <math.h>\n'
                            '#ifndef M_PI\n'
                            '#define M_PI 3.14159265358979323846264338327950288\n'
                            '#endif\n'
                            '#include "../testbench.cpp"\n')
# Report filenames can themselves be long. These private directory names
# reduce the Windows path burden without relocating or deleting any evidence.
PROJECT_DIR = 'p'
SOLUTION_DIR = 's'
STAGE_NAMES = ('setup', 'compile', 'csim', 'synthesis')


def tcl_path(path):
    value = Path(path).resolve().as_posix()
    if any(c in value for c in '{}\n\r'):
        raise ValueError('Unsupported character in Tcl path')
    return '{' + value + '}'


def _make_script(run, top, support_files=None, expected_stdout=False, expected_stderr=False,
                 required_tool_version='2025.2', testbench_compat=False):
    support_files = support_files or {}
    cflags = '-cflags {' + CXX_FLAGS + '} '
    # Keep compatibility in a source wrapper, not an absolute -include flag:
    # Vitis 2025.2's Windows shell mangled that flag into E:E:/... in real CSIM.
    # Quoted source includes resolve relative to this wrapper in every cwd.
    design = (run/'evidence/design_compat.cpp' if required_tool_version in REGISTER_COMPAT_VERSIONS
              else run/'candidate.cpp')
    testbench = run/('evidence/testbench_compat.cpp' if testbench_compat else 'testbench.cpp')
    extra = []
    for name in support_files:
        flag = '' if Path(name).suffix.lower() in ('.h', '.hh', '.hpp', '.hxx') else '-tb '
        extra.append('add_files ' + flag + cflags + tcl_path(run / name))
    setup = '\n'.join([
        'open_project ' + tcl_path(run / PROJECT_DIR),
        'set_top ' + top,
        'add_files ' + cflags + tcl_path(design),
        'add_files -tb ' + cflags + tcl_path(testbench),
        *extra,
        'open_solution ' + SOLUTION_DIR + ' -flow_target vivado',
        'set_part {' + PART + '}',
        'create_clock -period 5 -name default',
    ])
    # Each stage closes its evidence file before executing the next operation.
    # stdout is deliberately not used to infer progress or success.
    # Execute the exact Vitis-built C simulation binary, as HLS-Eval does.
    # Separating its stdout avoids comparing timestamps and compiler messages.
    simulation = [
        'set agent_old_dir [pwd]',
        'cd ' + tcl_path(run / PROJECT_DIR / SOLUTION_DIR / 'csim' / 'build'),
        'exec ./csim.exe > ' + tcl_path(run/'evidence'/'csim.stdout.txt')
        + ' 2> ' + tcl_path(run/'evidence'/'csim.stderr.txt'),
        'cd $agent_old_dir',
    ]
    data_setup = []
    for name in support_files:
        if Path(name).suffix.lower() not in ('.h', '.hh', '.hpp', '.hxx'):
            target = run/PROJECT_DIR/SOLUTION_DIR/'csim'/'build'/name
            data_setup += ['file mkdir ' + tcl_path(target.parent),
                           'file copy -force ' + tcl_path(run/name) + ' ' + tcl_path(target)]
    simulation = data_setup + simulation
    for stream, enabled in (('stdout', expected_stdout), ('stderr', expected_stderr)):
        if not enabled:
            continue
        simulation += [
            # Match the recorded byte hashes. Tcl text mode translates CR/LF
            # and treats Ctrl+Z as EOF on Windows, hiding unequal output bytes.
            'set actual_file [open ' + tcl_path(run/'evidence'/('csim.'+stream+'.txt')) + ' rb]',
            'set actual [read $actual_file]', 'close $actual_file',
            'set expected_file [open ' + tcl_path(run/('expected.'+stream+'.txt')) + ' rb]',
            'set expected [read $expected_file]', 'close $expected_file',
            'if {$actual ne $expected} {error "Test output differs from reference '+stream+'"}',
        ]
    return '\n'.join([
        'set agent_evidence_dir ' + tcl_path(run / 'evidence'),
        'proc agent_record {stage event} {',
        '    global agent_evidence_dir',
        '    set f [open [file join $agent_evidence_dir stages.tsv] a]',
        '    puts $f "$stage\\t$event\\t[clock milliseconds]"',
        '    close $f',
        '}',
        'proc agent_stage {stage body} {',
        '    global agent_evidence_dir',
        '    agent_record $stage start',
        '    if {[catch {uplevel 1 $body} message options]} {',
        '        set f [open [file join $agent_evidence_dir "$stage.error.txt"] w]',
        '        puts $f $message',
        '        if {[dict exists $options -errorinfo]} {',
        '            puts $f [dict get $options -errorinfo]',
        '        }',
        '        close $f',
        '        agent_record $stage fail',
        '        puts stderr "AGENT_STAGE_FAILED: $stage: $message"',
        '        exit 1',
        '    }',
        '    agent_record $stage pass',
        '}',
        'agent_stage setup {\n' + setup + '\n}',
        'agent_stage compile {csim_design -setup}',
        'agent_stage csim {\n' + '\n'.join(simulation) + '\n}',
        'agent_stage synthesis {csynth_design}',
        'exit 0',
        '',
    ])


def _read_stages(run):
    stages = {name: {'status': 'not_run'} for name in STAGE_NAMES}
    path = run / 'evidence' / 'stages.tsv'
    if not path.exists():
        return stages, None
    current = 0
    awaiting_end = False
    try:
        lines = path.read_text(encoding='utf-8', errors='strict').splitlines()
        for line in lines:
            fields = line.split('\t')
            if len(fields) != 3 or current >= len(STAGE_NAMES):
                raise ValueError('Unexpected stage evidence record')
            name, event, timestamp = fields
            stamp = int(timestamp)
            if name != STAGE_NAMES[current] or stamp < 0:
                raise ValueError('Invalid stage evidence order or timestamp')
            state = stages[name]
            if not awaiting_end and event == 'start':
                state.update(status='running', started_ms=stamp)
                awaiting_end = True
            elif awaiting_end and event in ('pass', 'fail'):
                state.update(status='passed' if event == 'pass' else 'failed',
                             ended_ms=stamp,
                             wall_seconds=max(0, stamp-state['started_ms']) / 1000)
                awaiting_end = False
                current += 1
                if event == 'fail':
                    # Failed stage must be the last record.
                    current = len(STAGE_NAMES)
            else:
                raise ValueError('Invalid stage evidence event sequence')
        return stages, None
    except (OSError, UnicodeError, ValueError) as exc:
        return stages, str(exc)


def _check_synthesis_evidence(run, top):
    report = run / PROJECT_DIR / SOLUTION_DIR / 'syn' / 'report' / (top + '_csynth.xml')
    rtl = [run / PROJECT_DIR / SOLUTION_DIR / 'syn' / lang / (top + ext)
           for lang, ext in (('verilog', '.v'), ('vhdl', '.vhd'))]
    evidence = {'reports': [], 'rtl_files': [], 'report_metadata': {}}
    errors = []
    try:
        if not report.is_file():
            errors.append('Missing exact top synthesis report: ' + report.name)
        else:
            evidence['reports'] = [str(report.relative_to(run))]
            root = ET.parse(report).getroot()
            values = {key: root.findtext('./UserAssignments/' + key)
                      for key in ('Part', 'TopModelName', 'TargetClockPeriod')}
            evidence['report_metadata'] = values
            if values['Part'] != PART:
                errors.append('Synthesis report part does not match ' + PART)
            if values['TopModelName'] != top:
                errors.append('Synthesis report top does not match ' + top)
            try:
                clock = float(values['TargetClockPeriod'])
                if not math.isfinite(clock) or abs(clock-CLOCK_NS) > 1e-6:
                    raise ValueError()
            except (TypeError, ValueError):
                errors.append('Synthesis report target clock does not match 5 ns')
            if root.tag != 'profile':
                errors.append('Unexpected synthesis report root element')
        evidence['rtl_files'] = [str(p.relative_to(run)) for p in rtl
                                 if p.is_file() and p.stat().st_size > 0]
        if not evidence['rtl_files']:
            errors.append('Missing non-empty RTL file for top ' + top)
    except (OSError, ET.ParseError) as exc:
        errors.append('Cannot read synthesis evidence: ' + str(exc))
    evidence['evidence_errors'] = errors
    return evidence


def _launch_process(command, cwd, stdout, env):
    if os.name == 'nt':
        try:
            from scripts import process_tree
        except ModuleNotFoundError:
            import process_tree
        return process_tree.launch(command, cwd, stdout, env)
    return subprocess.Popen(command, cwd=cwd, stdout=stdout,
                            stderr=subprocess.STDOUT, env=env,
                            start_new_session=True), None


def _close_process_control(control):
    if control is not None:
        try:
            from scripts import process_tree
        except ModuleNotFoundError:
            import process_tree
        return process_tree.close(control)
    return []


def _stop_process_tree(proc, control=None):
    """Best-effort tree termination with a strict bound on cleanup waits."""
    if control is not None:
        try:
            from scripts import process_tree
        except ModuleNotFoundError:
            import process_tree
        return process_tree.stop(proc, control)
    errors = []
    try:
        if os.name == 'nt':
            stopped = subprocess.run(
                ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                capture_output=True, timeout=5)
            if stopped.returncode:
                errors.append('taskkill exit code ' + str(stopped.returncode))
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(str(exc))
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append('Process termination incomplete: ' + str(exc))
    return errors


def _diagnostic_evidence(output, run, stages):
    selected = []
    infrastructure_error = has_infrastructure_error(output)
    for name in STAGE_NAMES:
        if stages[name]['status'] == 'failed':
            path = run / 'evidence' / (name + '.error.txt')
            if path.is_file():
                raw = path.read_text(encoding='utf-8', errors='replace')
                infrastructure_error |= has_infrastructure_error(raw)
                selected.extend(raw.splitlines())
    if stages['csim']['status'] in ('failed', 'running'):
        for name in ('csim.stderr.txt', 'csim.stdout.txt'):
            path = run / 'evidence' / name
            if path.is_file():
                selected.extend(path.read_text(encoding='utf-8', errors='replace').splitlines())
    selected.extend(output.splitlines())
    # Candidate stdout/stderr is feedback, not infrastructure/success evidence.
    return selected, infrastructure_error


def _diagnostics(output, run, stages):
    return select_diagnostics(_diagnostic_evidence(output, run, stages)[0],
                              max_lines=30, max_chars=12000)


def validate_tool_version(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}\.\d+', value):
        raise ValueError('Required tool version must have YYYY.N form, for example 2026.1')
    return value


def _needs_cordic_testbench_compat(top, testbench, support_files):
    header = support_files.get('cordic.h')
    return (top == 'cordic' and header is not None
            and hashlib.sha256(Path(testbench).read_bytes()).hexdigest() == CORDIC_TESTBENCH_SHA256
            and hashlib.sha256(Path(header).read_bytes()).hexdigest() == CORDIC_HEADER_SHA256)


def verify(source, testbench, top, tool, output_root, timeout=300,
           support_files=None, expected_stdout=None, required_tool_version='2025.2',
           expected_stderr=None):
    validate_tool_version(required_tool_version)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', top):
        raise ValueError('Invalid top function name')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    started = time.monotonic()
    deadline = started + timeout
    source, testbench = Path(source).resolve(), Path(testbench).resolve()
    if not source.is_file() or not testbench.is_file():
        raise FileNotFoundError('Source and testbench must exist')
    support_files = validate_support(support_files or {})
    for stream, path in (('stdout', expected_stdout), ('stderr', expected_stderr)):
        if path is not None and not Path(path).is_file():
            raise ValueError('Missing expected ' + stream + ' file')
    run = Path(output_root).resolve() / ('r' + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, run / 'candidate.cpp')
    shutil.copy2(testbench, run / 'testbench.cpp')
    for name, path in support_files.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, run / name)
    if expected_stdout is not None:
        shutil.copy2(expected_stdout, run / 'expected.stdout.txt')
    if expected_stderr is not None:
        shutil.copy2(expected_stderr, run / 'expected.stderr.txt')
    (run / 'evidence').mkdir()
    if required_tool_version in REGISTER_COMPAT_VERSIONS:
        (run/'evidence/design_compat.cpp').write_bytes(REGISTER_WRAPPER.encode('ascii'))
    testbench_compat = _needs_cordic_testbench_compat(top, run/'testbench.cpp',
                                                   {name: run/name for name in support_files})
    if testbench_compat:
        (run/'evidence/testbench_compat.cpp').write_bytes(CORDIC_TESTBENCH_WRAPPER.encode('ascii'))
    script = run / 'verify.tcl'
    script.write_text(_make_script(run, top, support_files, expected_stdout is not None,
                                  expected_stderr is not None, required_tool_version,
                                  testbench_compat), encoding='utf-8')
    command = [str(tool), '--mode', 'hls', '--tcl', str(script)]
    result = {
        'schema_version': 4,
        'cxx_flags': CXX_FLAGS,
        'project_layout': {'project': PROJECT_DIR, 'solution': SOLUTION_DIR,
                           'run_name': run.name},
        'compiler_compatibility': ({
            'kind': 'vitis_' + required_tool_version.replace('.', '_') + '_register_extension_wrapper',
            'ignored_diagnostic': '-Wregister',
            'translation_unit': 'evidence/design_compat.cpp',
            'translation_unit_sha256': hashlib.sha256(REGISTER_WRAPPER.encode('ascii')).hexdigest(),
            'source_rewritten': False,
        } if required_tool_version in REGISTER_COMPAT_VERSIONS else None),
        'testbench_compatibility': ({
            'kind': 'pinned_pp4fpga_cordic_portable_pi',
            'translation_unit': 'evidence/testbench_compat.cpp',
            'translation_unit_sha256': hashlib.sha256(CORDIC_TESTBENCH_WRAPPER.encode('ascii')).hexdigest(),
            'testbench_sha256': CORDIC_TESTBENCH_SHA256,
            'header_sha256': CORDIC_HEADER_SHA256,
            'testbench_rewritten': False,
        } if testbench_compat else None),
        'required_tool_version': required_tool_version,
        'source_sha256': hashlib.sha256((run/'candidate.cpp').read_bytes()).hexdigest(),
        'testbench_sha256': hashlib.sha256((run/'testbench.cpp').read_bytes()).hexdigest(),
        'part': PART, 'clock_ns': CLOCK_NS,
        'top': top, 'command': command, 'run_dir': str(run),
        'support_sha256': {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                           for name in support_files},
        'functional_check': ('reference_stdout_stderr_and_exit' if expected_stdout is not None and expected_stderr is not None
                             else 'reference_stdout_and_exit' if expected_stdout is not None
                             else 'reference_stderr_and_exit' if expected_stderr is not None else 'testbench_exit'),
        'expected_stdout_sha256': hashlib.sha256((run/'expected.stdout.txt').read_bytes()).hexdigest()
                                  if expected_stdout is not None else None,
        'expected_stderr_sha256': hashlib.sha256((run/'expected.stderr.txt').read_bytes()).hexdigest()
                                  if expected_stderr is not None else None,
        'parse_pass': None,
        'parse_note': 'This verifier does not establish a separate Vitis L1 parse result; code extraction is not parsing evidence.',
        'compile_pass': False, 'csim_pass': False, 'synthesis_pass': False,
        'status': 'tool_error', 'failure_kind': 'infrastructure',
        'failure_stage': None, 'returncode': None,
        'timeout_seconds': timeout, 'timeout_phase': None,
    }
    env = os.environ.copy()
    temp = run / 'temp'
    temp.mkdir()
    env.update({'TEMP': str(temp), 'TMP': str(temp), 'TMPDIR': str(temp)})
    timed_out = False
    proc = None
    control = None
    with (run/'tool.log').open('w', encoding='utf-8') as log:
        try:
            if time.monotonic() >= deadline:
                timed_out = True
                result['timeout_phase'] = 'preparation'
            else:
                proc, control = _launch_process(command, cwd=run, stdout=log, env=env)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    result['timeout_phase'] = 'launch'
                else:
                    try:
                        result['returncode'] = proc.wait(timeout=remaining)
                        timed_out = time.monotonic() >= deadline
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    if timed_out:
                        result['timeout_phase'] = 'execution'
                if timed_out:
                    result['cleanup_errors'] = _stop_process_tree(proc, control)
                    result['returncode'] = proc.poll()
        except OSError as exc:
            result['launch_error'] = str(exc)
        finally:
            if proc is not None and control is None and not timed_out:
                result['cleanup_errors'] = _stop_process_tree(proc, control)
            close_errors = _close_process_control(control)
            if control is not None:
                result['process_control'] = {
                    'kind': 'windows_job', 'assigned': control.assigned,
                    'cleanup_confirmed': control.cleanup_confirmed,
                    'remaining_processes': control.last_active_processes,
                    'closed': control.closed,
                }
                result['cleanup_errors'] = list(dict.fromkeys(
                    result.get('cleanup_errors', []) + control.errors + close_errors))
            if sys.exc_info()[0] is not None and result.get('cleanup_errors'):
                # Do not acknowledge SIGTERM (exit 143) when nested EDA cleanup
                # failed while unwinding the worker. The outer API must fence it.
                raise RuntimeError('EDA cleanup failed during cancellation: ' +
                                   '; '.join(result['cleanup_errors']))
    output = (run/'tool.log').read_text(encoding='utf-8', errors='replace')
    version = re.search(r'^\*+\s+vitis-run v(\d+\.\d+)\b', output, re.M)
    result['tool_version'] = version.group(1) if version else None
    stages, stage_error = _read_stages(run)
    result['stages'] = stages
    result['stage_evidence_error'] = stage_error
    if stage_error is None:
        result['compile_pass'] = stages['compile']['status'] == 'passed'
        result['csim_pass'] = result['compile_pass'] and stages['csim']['status'] == 'passed'
    diagnostic_lines, infrastructure_error = _diagnostic_evidence(output, run, stages)
    result['diagnostics'] = select_diagnostics(diagnostic_lines, max_lines=30, max_chars=12000)
    # Only a failed compile stage supplies compiler-location evidence. Runtime
    # output may imitate compiler messages and must never drive this decision.
    compile_lines = []
    if stage_error is None and stages['compile']['status'] == 'failed':
        compile_lines.append(output)
        compile_error = run/'evidence'/'compile.error.txt'
        if compile_error.is_file():
            compile_lines.append(compile_error.read_text(encoding='utf-8', errors='replace'))
    result['compile_errors'] = compiler_error_locations(compile_lines)
    result['infrastructure_error_detected'] = infrastructure_error
    evidence = _check_synthesis_evidence(run, top)
    result.update(evidence)
    active = next((name for name in STAGE_NAMES
                   if stages[name]['status'] in ('failed', 'running')), None)
    result['failure_stage'] = active
    result['diagnostic_artifacts'] = {
        p.relative_to(run).as_posix(): {'bytes': p.stat().st_size}
        for p in (run/'tool.log', run/'verify.tcl', *(run/'evidence').iterdir())
        if p.is_file()
    }
    if not timed_out and time.monotonic() >= deadline:
        timed_out = True
        result['timeout_phase'] = 'cleanup_or_evidence'
    if timed_out:
        result.update(status='timeout', failure_kind='timeout')
        result['timeout_stage'] = active
        result['diagnostics'].insert(0, 'Verifier deadline exceeded after ' + str(timeout)
                                     + ' seconds; active stage: ' + (active or 'none')
                                     + '; phase: ' + str(result['timeout_phase']))
    elif result.get('cleanup_errors') or (control is not None and not control.cleanup_confirmed):
        result['diagnostics'].extend(result.get('cleanup_errors', []))
        result['diagnostics'].append('Process tree cleanup was not fully confirmed')
    elif stage_error:
        result['diagnostics'].insert(0, stage_error)
    elif 'launch_error' in result:
        result['diagnostics'].insert(0, result['launch_error'])
    elif result['tool_version'] != required_tool_version:
        result.update(compile_pass=False, csim_pass=False)
        result['diagnostics'].insert(0, 'Expected Vitis ' + required_tool_version + '; tool banner is missing or mismatched')
    elif active and stages[active]['status'] == 'failed':
        if active != 'setup' and not infrastructure_error:
            result.update({
                'compile': {'status': 'compile_failed', 'failure_kind': 'compile'},
                'csim': {'status': 'test_failed', 'failure_kind': 'functional'},
                'synthesis': {'status': 'synthesis_failed', 'failure_kind': 'synthesizability'},
            }[active])
    elif (result['returncode'] == 0 and result['csim_pass']
          and stages['synthesis']['status'] == 'passed' and not evidence['evidence_errors']):
        result.update(status='passed', failure_kind=None, failure_stage=None,
                      synthesis_pass=True, diagnostics=[])
    else:
        result['diagnostics'].extend(evidence['evidence_errors'])
        result['diagnostics'].append('HLS exited without complete successful stage evidence'
                                     if active or not result['csim_pass'] else
                                     'HLS exit status or synthesis artifacts are inconsistent')
    result['wall_seconds'] = round(time.monotonic()-started, 3)
    (run/'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    finished = time.monotonic()
    if not timed_out and finished >= deadline:
        # Disk writes are not preemptible here. Never return a late success just
        # because serialization completed after the preceding budget check.
        result.update(status='timeout', failure_kind='timeout', synthesis_pass=False,
                      timeout_phase='result_write', wall_seconds=round(finished-started, 3))
        (run/'result.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--testbench', required=True)
    parser.add_argument('--top', required=True)
    parser.add_argument('--tool', default=os.environ.get('VITIS_RUN', 'vitis-run'))
    parser.add_argument('--output-root', default='runs')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--required-tool-version', default=os.environ.get('HLS_TOOL_VERSION', '2025.2'))
    result = verify(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
