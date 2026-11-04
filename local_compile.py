"""Vitis compile/link probe with no executed testbench and no functional verdict."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from scripts import verify_hls as hls
from scripts.task_files import validate_support
from scripts.diagnostics import compiler_error_locations, select_diagnostics
from agent.public_interface import MAX_HEADER_BYTES, public_functions


def link_entry(top, supports, *, discovered=None):
    """Link the exact public declaration, without inventing argument values.

    A volatile function pointer keeps the relocation even under optimization.
    This catches an absent definition or different parameter ABI, including
    discovered ordinary public helpers. It does not prove behavior, synthesis,
    completeness of declaration discovery, or every cross-TU type discrepancy.
    """
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', top):
        raise ValueError('Invalid public top name')
    names = list(supports)
    if not names or any(not re.fullmatch(r'[A-Za-z0-9_./-]+\.(?:h|hh|hpp|hxx)', n)
                        for n in names):
        raise ValueError('Compile probe requires safely named public headers')
    includes = ''.join('#include "'+name+'"\n' for name in names)
    functions = public_functions(supports)[0] if discovered is None else discovered
    identifiers = set([top, *functions])
    for path in supports.values():
        if path is None:
            continue
        try:
            with Path(path).open('rb') as stream:
                raw = stream.read(MAX_HEADER_BYTES+1)
            if len(raw) <= MAX_HEADER_BYTES:
                text = re.sub(r'\\\r?\n', '', raw.decode('utf-8-sig'))
                identifiers.update(re.findall(r'[A-Za-z_]\w*', text))
        except (OSError, UnicodeError):
            pass  # Discovery reports the omission; the compiler still checks includes.
    def fresh(preferred):
        name, index = preferred, 0
        while name in identifiers:
            index += 1
            name = preferred+'_'+str(index)
        identifiers.add(name)
        return name
    entry_name = fresh('public_entry')
    entry = includes+'int main() {\n    auto volatile '+entry_name+' = &'+top+';\n'
    for index, name in enumerate(functions):
        if name != top:
            local_name = fresh('public_helper_'+str(index))
            entry += ('    auto volatile '+local_name+' = &'+name+';\n'
                      '    if ('+local_name+' == nullptr) return 1;\n')
    return entry+'    return '+entry_name+' == nullptr;\n}\n'


def compile_script(run, top, supports, version):
    # Reuse the already exercised Vitis setup/compile stage verbatim. This
    # script ends before csim execution and before synthesis.
    full = hls._make_script(run, top, supports, required_tool_version=version)
    marker = 'agent_stage csim {\n'
    if full.count(marker) != 1:
        raise ValueError('Unrecognized Vitis stage script; refuse compile probe')
    return full.split(marker)[0] + 'exit 0\n'


def verify_compile(*, source, top, tool, output_root, timeout, support_files=None,
                   required_tool_version='2025.2', **unused):
    started = time.monotonic()
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Positive finite compile timeout required')
    deadline = started + timeout
    hls.validate_tool_version(required_tool_version)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', top):
        raise ValueError('Invalid public top name')
    supports = validate_support(support_files or {})
    if any(Path(n).suffix.lower() not in ('.h','.hh','.hpp','.hxx') for n in supports):
        raise ValueError('Compile probe accepts public headers only')
    functions, omissions = public_functions(supports)
    entry = link_entry(top, supports, discovered=functions)
    run = Path(output_root).resolve()/('r'+uuid.uuid4().hex[:12])
    (run/'evidence').mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, run/'candidate.cpp')
    for name, path in supports.items():
        (run/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, run/name)
    # Link entry only. NEVER execute this program or use it as a functional test.
    (run/'testbench.cpp').write_bytes(entry.encode('utf-8'))
    if required_tool_version in hls.REGISTER_COMPAT_VERSIONS:
        (run/'evidence/design_compat.cpp').write_bytes(hls.REGISTER_WRAPPER.encode('ascii'))
    script = run/'verify.tcl'
    script.write_text(compile_script(run, top, supports, required_tool_version), encoding='utf-8')
    command = [str(tool), '--mode', 'hls', '--tcl', str(script)]
    result = {'status':'tool_error','failure_kind':'infrastructure','compile_pass':False,
        'csim_pass':False,'synthesis_pass':False,'passed_official_tests':None,
        'evaluation_scope':'compile_link_only_no_functional_test',
        'testbench_executed':False,'top_signature_with_official_testbench_verified':False,
        'public_top_linked':False,'link_entry_sha256':hashlib.sha256(entry.encode('utf-8')).hexdigest(),
        'public_function_link_check':{'symbols':list(dict.fromkeys([top, *functions])),
            'discovery_omissions':omissions, 'linked':False,
            'scope':'top_and_discovered_ordinary_public_functions'},
        'source_sha256':hashlib.sha256((run/'candidate.cpp').read_bytes()).hexdigest(),
        'support_sha256':{n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in supports},
        'required_tool_version':required_tool_version,'command':command,'run_dir':str(run),
        'returncode':None,'cleanup_errors':[],'timeout_seconds':timeout}
    env = os.environ.copy()
    (run/'temp').mkdir()
    env.update(TEMP=str(run/'temp'),TMP=str(run/'temp'),TMPDIR=str(run/'temp'))
    proc = control = None
    timed_out = False
    with (run/'tool.log').open('w',encoding='utf-8') as log:
        try:
            if time.monotonic() >= deadline:
                timed_out = True
            else:
                proc, control = hls._launch_process(command,cwd=run,stdout=log,env=env)
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    timed_out = True
                else:
                    try:
                        result['returncode'] = proc.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    timed_out = timed_out or time.monotonic() >= deadline
                if timed_out:
                    result['cleanup_errors'] = hls._stop_process_tree(proc,control)
                    result['returncode'] = proc.poll()
        except OSError as exc:
            result['launch_error'] = str(exc)
        finally:
            if proc is not None and control is None and not timed_out:
                result['cleanup_errors'] = hls._stop_process_tree(proc,control)
            errors = hls._close_process_control(control)
            if control is not None:
                result['process_control'] = {'kind':'windows_job','assigned':control.assigned,
                    'cleanup_confirmed':control.cleanup_confirmed,'closed':control.closed,
                    'remaining_processes':control.last_active_processes}
                result['cleanup_errors'] = list(dict.fromkeys(result['cleanup_errors']+control.errors+errors))
            if sys.exc_info()[0] is not None and result['cleanup_errors']:
                raise RuntimeError('EDA cleanup failed during compile-probe cancellation: '+
                                   '; '.join(result['cleanup_errors']))
    output = (run/'tool.log').read_text(encoding='utf-8',errors='replace')
    version = re.search(r'^\*+\s+vitis-run v(\d+\.\d+)\b',output,re.M)
    result['tool_version'] = version[1] if version else None
    stages, stage_error = hls._read_stages(run)
    result.update(stages=stages,stage_evidence_error=stage_error)
    lines, infrastructure = hls._diagnostic_evidence(output,run,stages)
    result['diagnostics'] = select_diagnostics(lines,max_lines=30,max_chars=12000)
    result['infrastructure_error_detected'] = infrastructure
    compiler_lines = [output]
    if (run/'evidence/compile.error.txt').exists():
        compiler_lines.append((run/'evidence/compile.error.txt').read_text(encoding='utf-8',errors='replace'))
    result['compile_errors'] = compiler_error_locations(compiler_lines if not stage_error and
        stages['compile']['status']=='failed' else [])
    executable = run/hls.PROJECT_DIR/hls.SOLUTION_DIR/'csim/build/csim.exe'
    safe = (not stage_error and not infrastructure and 'launch_error' not in result
            and not result['cleanup_errors'] and (control is None or control.cleanup_confirmed)
            and result['tool_version']==required_tool_version
            and stages['setup']['status']=='passed'
            and all(stages[n]['status']=='not_run' for n in ('csim','synthesis')))
    if safe and stages['compile']['status']=='failed' and result['returncode'] not in (None,0):
        result.update(status='compile_failed',failure_kind='compile',failure_stage='compile')
    elif (safe and stages['compile']['status']=='passed' and result['returncode']==0
          and executable.is_file() and executable.stat().st_size>0):
        result.update(status='compile_only_passed',failure_kind=None,compile_pass=True,public_top_linked=True)
        result['public_function_link_check']['linked'] = True
    if timed_out or time.monotonic() >= deadline:
        result.update(status='timeout',failure_kind='timeout',compile_pass=False,public_top_linked=False)
        result['public_function_link_check']['linked'] = False
    result['wall_seconds'] = round(time.monotonic()-started,3)
    path = run/'result.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if time.monotonic() >= deadline and result['status']!='timeout':
        result.update(status='timeout',failure_kind='timeout',compile_pass=False,public_top_linked=False,
                      wall_seconds=round(time.monotonic()-started,3))
        result['public_function_link_check']['linked'] = False
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
