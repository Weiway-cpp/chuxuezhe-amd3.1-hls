"""Validate immutable task inputs without exposing reference implementations."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re


# Reviews are bound to every immutable task input, not to a filename or a
# keyword heuristic. Changing a header, prompt, data file or testbench removes
# the exception. These describe harness behavior, never model performance.
REVIEWED_HARNESSES = {
    '824632aa0c6d7254a0b9cc8d434dd9cc1f32f1b0bab3f6a7e3f03875fd10a878': {
        'case': 'chstone/df_extractFloat64Exp', 'kind': 'silent_self_check',
        'reason': 'All ten expected exponents are compared; any mismatch returns 1. Native reference and zero-return mutant checked.'},
    'c1b89b87a9291f7b698256d6d5501dfebde6f16ebed8bc8fc3b8461d55fb8566': {
        'case': 'rosetta/digit_recognition__popcount', 'kind': 'silent_self_check',
        'reason': 'All seven expected counts are compared and any mismatch sets the returned fail_flag. Source reviewed; Vitis validation required.'},
    'ecd16abc7c4e034e51b48546b632c9a5c0202d9a3663191b655246ce9508aaf9': {
        'case': 'rosetta/rendering_3d__clockwise_vertices', 'kind': 'silent_self_check',
        'reason': 'Both supplied vertex-swap cases return 1 on mismatch. Source reviewed; Vitis validation required.'},
    'd1c28f2180de102594c887c9ba74496bbc33d3804d91a3cc4948210f146ff18c': {
        'case': 'rosetta/rendering_3d__check_clockwise', 'kind': 'silent_self_check',
        'reason': 'All five supplied orientation checks set the returned test_failed flag. Source reviewed; Vitis validation required.'},
    '06f19bf98ec852feb159327d84f6df6633d0d125a31409e7f1a0434ba9cd9054': {
        'case': 'gnnbuilder/compute_neighbor_tables', 'kind': 'invalid_testbench',
        'reason': 'edge_list has MAX_NODES=100 rows but the initialization loop writes num_edges=500 rows; output is also unchecked.'},
    'f50c5d947427cc4ce3e03087f4982abece283e57be0c72867fcd84d42b5e1bd3': {
        'case': 'gnnbuilder/gather_node_neighbors', 'kind': 'invalid_testbench',
        'reason': 'neighbor_table[7+i] is written for i<MAX_EDGES, overrunning its MAX_EDGES elements by seven; output is also unchecked.'},
    '625b6e061c1a59ce6f19d57dad1cd5dbd161dbe998dec37058b46eb1f8281ab3': {
        'case': 'gnnbuilder/global_add_pool', 'kind': 'unobserved_output',
        'reason': 'pooled_embedding is neither printed nor checked; native reference and no-op mutant both exit 0 with empty streams.'},
    '9094c761a1b736455300096ca394b3a670d21f683a3c0f5bf2a5b4caca93f259': {
        'case': 'flowgnn/fgnn_linear_input_stationary', 'kind': 'unobserved_output',
        'reason': 'The output array is neither printed nor checked after the kernel call.'},
    '174f7d1f40a15e005c32c4825101e9e40888ba6720f5a1408a7f16791426fde4': {
        'case': 'flowgnn/fgnn_linear_output_stationary', 'kind': 'unobserved_output',
        'reason': 'The output stream is drained, but its values are neither printed nor checked.'},
    'e8fc573415a4de0e4dd00b1b7cd3b8d3ddb7ec140432465e3c3da138bd4e06e2': {
        'case': 'c2hlsc/des', 'kind': 'invalid_reference',
        'reference_source_sha256': '1acd10e278df98bc9514071e12f7c84195e221224654c91a7fc4f91e1661e911',
        'reason': 'des_key_t *key points to one 16x6-byte schedule, but key[idx] and key[15] step across whole schedules instead of six-byte subkeys. The supplied reference reads beyond the testbench allocation; a successful run cannot establish a valid oracle.'},
    '5b3cb72b4dee6482a0a4758ef853fbb07069767297dabb1bbe9ff86f6289460a': {
        'case': 'pp4fpga/block_mm', 'kind': 'success_output', 'stdout': 'passed\n',
        'allow_hls_stream_depth_metadata': True,
        'reason': 'The testbench returns 0 even when its software comparison prints failed; only the explicit passed output is valid.'},
    '0383ee0abbf926ed168f2e029e38c394a88142ee582eb547a4db76fc7e9deec0': {
        'case': 'rosetta/rendering_3d__pixel_in_triangle', 'kind': 'success_output', 'stdout': 'Test passed!\n',
        'reason': 'Several mismatch paths print Error without setting fail_flag; only a clean success line, without errors, is valid.'},
}


def reviewed_harness(task):
    review = REVIEWED_HARNESSES.get(task_input_sha256(task))
    return dict(review) if review is not None else None


def validate_observed_output(task, stdout, stderr):
    """Do not confuse repeatable process output with a meaningful oracle."""
    review = reviewed_harness(task)
    if review and review['kind'] in ('invalid_testbench', 'unobserved_output', 'invalid_reference'):
        raise ValueError('Reviewed upstream harness requires repair: ' + review['reason'])
    if review and review['kind'] == 'success_output':
        # Recognize only the reviewed functional success line and, for the
        # stream-based block_mm harness, the exact form of the metadata line
        # observed in Vitis. Metadata alone never demonstrates correctness.
        # This validation does not transform either stored or compared bytes.
        success = review['stdout'].encode('ascii')
        accepted = False
        for ending in (b'\n', b'\r\n'):
            pattern = re.escape(success.replace(b'\n', ending))
            if review.get('allow_hls_stream_depth_metadata'):
                metadata = (re.escape(b'INFO [HLS SIM]: The maximum depth reached by any '
                                      b'hls::stream() instance in the design is ')
                            + rb'[0-9]+' + re.escape(ending))
                pattern += b'(?:' + metadata + b')?'
            accepted = accepted or re.fullmatch(pattern, stdout) is not None
        if not accepted or stderr:
            raise ValueError('Reference testbench did not report clean success: ' + review['reason'])
    if not stdout and not stderr and not (review and review['kind'] == 'silent_self_check'):
        raise ValueError('Empty output has no reviewed self-checking testbench')
    return review


def relative_file(root, value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Input paths must use nonempty relative POSIX paths')
    rel = PurePosixPath(value)
    if rel.is_absolute() or any(x in ('..', '.') for x in rel.parts) or ':' in value:
        raise ValueError('Input path must stay inside the task directory')
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Missing or escaping task input: ' + value)
    return path


def validate_support(files, source_name='candidate.cpp', testbench_name='testbench.cpp'):
    """Normalize destination names; reject aliases, file/dir collisions and code."""
    if not isinstance(files, dict):
        raise ValueError('support_files must be a name-to-path mapping')
    result, seen = {}, {source_name.casefold(), testbench_name.casefold()}
    reserved = {'project', 'p', 'evidence', 'temp', 'tool.log', 'result.json', 'manifest.json',
                'verify.tcl', 'expected.stdout.txt', 'expected.stderr.txt'}
    for name, path in files.items():
        if not isinstance(name, str):
            raise ValueError('Support filenames must be strings')
        rel = PurePosixPath(name)
        canonical = rel.as_posix()
        folded = canonical.casefold()
        if (not name or not rel.parts or rel.is_absolute() or '\\' in name or ':' in name
                or any(x in ('..', '.') for x in rel.parts)
                or rel.parts[0].casefold() in reserved
                or any(folded == other or folded.startswith(other+'/')
                       or other.startswith(folded+'/') for other in seen)):
            raise ValueError('Invalid or colliding support filename: ' + name)
        if rel.suffix.lower() in ('.c', '.cpp', '.cc', '.cxx'):
            raise ValueError('Support files cannot contain extra implementations')
        seen.add(folded)
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError('Missing support file: ' + str(path))
        result[canonical] = path
    return result


def task_input_sha256(task):
    """Fingerprint the task/oracle inputs, excluding paths, answers and models."""
    payload = {'top':task['top'], 'description':task['description'],
               'testbench':hashlib.sha256(Path(task['_testbench']).read_bytes()).hexdigest(),
               'support':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name, path in task.get('_support_files', {}).items()}}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(raw).hexdigest()


def validate_task_oracle(task_file, task, required_tool_version='2025.2'):
    """Require a current calibration; legacy evidence stays historical."""
    oracle_path = Path(task_file).parent/'oracle.json'
    if not task.get('_expected_stdout') or not task.get('_expected_stderr') or not oracle_path.is_file():
        raise ValueError('Calibrate imported task before benchmark: '+task['id'])
    evidence = json.loads(oracle_path.read_text(encoding='utf-8'))
    if not isinstance(evidence, dict) or evidence.get('schema_version') != 5:
        raise ValueError('Legacy or invalid oracle; recalibrate imported task: '+task['id'])
    if evidence.get('stdout_comparison') != 'byte_exact':
        raise ValueError('Oracle stdout comparison changed; recalibrate imported task: '+task['id'])
    if evidence.get('stderr_comparison') != 'byte_exact':
        raise ValueError('Oracle stderr comparison changed; recalibrate imported task: '+task['id'])
    manifest_bytes = (Path(task_file).parent/'source_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if evidence.get('tool_version') != required_tool_version:
        raise ValueError('Oracle tool version mismatch; recalibrate imported task: '+task['id'])
    expected = {'task_input_sha256':task_input_sha256(task),
                'stdout_comparison':'byte_exact',
                'stderr_comparison':'byte_exact',
                'tool_version':required_tool_version,
                'expected_stdout_sha256':hashlib.sha256(Path(task['_expected_stdout']).read_bytes()).hexdigest(),
                'expected_stderr_sha256':hashlib.sha256(Path(task['_expected_stderr']).read_bytes()).hexdigest(),
                'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
                'upstream_revision':task.get('provenance',{}).get('revision')}
    if not isinstance(manifest,dict) or manifest.get('revision') != expected['upstream_revision']:
        raise ValueError('Calibrated task provenance changed: '+task['id'])
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError('Calibrated task inputs or expected stdout changed: '+task['id'])
    review = validate_observed_output(task, Path(task['_expected_stdout']).read_bytes(),
                                     Path(task['_expected_stderr']).read_bytes())
    if review and evidence.get('harness_review') != review:
        raise ValueError('Oracle lacks the current harness review; recalibrate imported task: '+task['id'])
    return expected


def validate_oracle_snapshot(task, required_tool_version='2025.2'):
    """Recheck copied inputs before any model call, if calibration was required."""
    evidence = task.get('_oracle_validation')
    if evidence is None:
        return
    if evidence.get('stdout_comparison') != 'byte_exact':
        raise ValueError('Calibrated stdout comparison changed before the trial snapshot: '+task['id'])
    if evidence.get('stderr_comparison') != 'byte_exact':
        raise ValueError('Calibrated stderr comparison changed before the trial snapshot: '+task['id'])
    if evidence.get('tool_version') != required_tool_version:
        raise ValueError('Calibrated tool version changed before the trial snapshot: '+task['id'])
    if (task_input_sha256(task) != evidence['task_input_sha256'] or
            hashlib.sha256(Path(task['_expected_stdout']).read_bytes()).hexdigest() != evidence['expected_stdout_sha256'] or
            hashlib.sha256(Path(task['_expected_stderr']).read_bytes()).hexdigest() != evidence['expected_stderr_sha256']):
        raise ValueError('Calibrated inputs changed before the trial snapshot: '+task['id'])
    validate_observed_output(task, Path(task['_expected_stdout']).read_bytes(),
                             Path(task['_expected_stderr']).read_bytes())
