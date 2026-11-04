"""Specification-only reference testbenches and bounded structural screening.

The reference is another model hypothesis, not an oracle. Screening verifies
the requested shape and rejects obvious DUT-derived expectations; it cannot
prove reference semantics, domain validity, alias independence or coverage.
Compilation/execution and the independent external judge remain necessary.
"""
import re

from agent.interface_guidance import interface_guidance
from agent.reference_integrity import validate_reference_integrity

# Records reference-integrity findings that were downgraded from "reject the
# testbench" to "run it anyway". The dataflow screen is a static heuristic over
# arbitrary C++: it refuses many testbenches whose reference outputs do reach a
# failure check through forms it does not model (helper comparison functions,
# stacked arrays, loop-carried expectations). Rejecting those throws away the
# only functional signal this host can obtain, so the caller may keep the
# testbench and let the real compiler/simulator decide.
INTEGRITY_DOWNGRADES = []


def _reference_name(top):
    if not isinstance(top, str) or not re.fullmatch(r'[A-Za-z_]\w*', top, re.ASCII):
        raise ValueError('Reference self-test needs a plain public top identifier')
    return 'reference_' + top


def build_reference_messages(description, top, header, max_prompt_chars=None):
    """Build one complete prompt using public inputs, never a candidate design."""
    reference = _reference_name(top)
    if not isinstance(header, str) or not re.fullmatch(
            r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:h|hh|hpp|hxx)', header):
        raise ValueError('Reference self-test needs a safe public header filename')
    system = (
        'Write one compact C++14 testbench for the supplied public specification and immutable interface. '
        'You have no design implementation or judge tests. Include "'+header+'" and define int main(). '
        'First derive a plain, separately named reference function '+reference+' from the specification. '
        'Implement its computation, not a declaration, a table of guessed answers or a call to the DUT. '
        'Prefer a simple independent mathematical algorithm over copying the described hardware '
        'pipeline, lookup-table architecture or scratch-buffer organization. Reference code need '
        'not be synthesizable. For bit-preserving conversions use memcpy with equal-size types, '
        'never type-punning through reinterpret_cast references. '
        'Additional ordinary named reference helpers are allowed. The reference and every helper must '
        'be independent: never call, take the address of, alias or use '+top+' outside main. '
        'Do not define the DUT: never give any function, functor, alias or macro the exact name '
        +top+'; that name may appear only in calls inside main. Name your model of the expected '
        'behavior exactly '+reference+' and call that. Inside main, do not redeclare, shadow or '
        'replace the DUT or reference. '
        'Inside main, call '+reference+' and '+top+' directly on independent copies of identical inputs. '
        'Immediately before each reference/DUT call, write one short std::cerr line '
        'SELFTEST_PHASE=reference or SELFTEST_PHASE=dut with the case index and flush. '
        'These stage markers help locate a crash; they are not correctness checks. '
        'Compare actual return values and every specified output against the calculated reference. '
        'Identify API output and inout objects separately from scratch buffers; checking a temporary '
        'does not substitute for checking the specified mutated output. '
        'Derive output obligations from the public postconditions, not merely from a writable '
        'pointer type or parameter name. Do not compare final scratch-buffer bytes when the '
        'specification leaves their contents unspecified: valid algorithms may use different '
        'temporary layouts. If a buffer is a specified output, check it even when also used '
        'as scratch. Never invent extra output requirements from reference internals. '
        'Use an explicit nonconstant assert or conditional return 1 inside main after the calls; '
        'return 0 only after every check. Merely calling the reference without checking its outputs, '
        'checking inputs, self-comparisons and unconditional failures are not evidence. '
        'Do not feed DUT results to the reference or overwrite expected outputs with actual outputs. '
        'Keep reference output storage separate from every DUT argument; compare before reusing buffers. '
        'Reset independent state and copy mutable inputs '
        'before either call, preserving state across calls only as specified. '
        'Cover the domain with several cases, not just one: zero or sparse, small distinct values, '
        'one relevant boundary, and, when the input domain is small enough to enumerate '
        'independently, 8 to 12 pseudorandom cases from a small fixed-seed unsigned generator. '
        'A single easy case hides off-by-one, ordering, sign and boundary bugs that the graded '
        'tests will find. Compare every specified output element, and on the first mismatch '
        'print the failing element index with expected and actual values. No time-based seed, '
        'exhaustive enumeration or large literal tables. Reuse loops and reference calculations '
        'across cases. '
        'Resolve every public typedef before allocation. Preserve the exact multidimensional shape, '
        'extent and pointer-to-array type for each input, intermediate and output independently; '
        'derive each loop bound from the array being accessed, never from a different output array. '
        'Do not flatten or reinterpret multidimensional storage or redeclare supplied aliases. '
        'Initialize every value that will be read. No undefined behavior or unbounded loops. '
        'Match specified signedness, bit widths, encoding, reset behavior and operation order. '
        'For encoded integer bit fields, extract the requested raw fields: do not subtract a bias, '
        'normalize or reinterpret as floating point unless explicitly required. '
        'Calculate expected values, including bit boundaries, instead of guessing constants. '
        'Bound each product, partial sum, shift and final output in the declared type. Signed C++ '
        'overflow is not wrapping. When overflow behavior is unspecified, use representable operands '
        'and intermediates; a wider reference accumulator does not justify an out-of-range output. '
        'For specified wrap, saturation or fixed-point quantization, explicitly reproduce those rules '
        'at the stated stages using independently derived arithmetic. '
        'For floating point, obey the task precision and accuracy requirements, including operation '
        'order when specified. Do not impose a universal epsilon, ULP bound or bit-equality rule. '
        'Use small finite inputs unless exceptional behavior is specified. Check classification before '
        'numeric error: unexpected NaNs or infinities must fail; fabs(actual-expected)>tol alone '
        'does not reject NaN. For finite approximate results with a task-justified tolerance, '
        'use a negated acceptance check such as !(fabs(actual-expected)<=tol), or explicit '
        'classification checks before the numeric comparison. Only accept NaN or infinity when required for that input by the '
        'specification, and check infinity sign. Derive any allowed tolerance from the task. '
        'Before a failure return/assert, print one concise std::cerr record beginning FAIL case= '
        'with index, relevant inputs, expected and actual values where representable, then flush. '
        'Make numeric radix explicit: prefix hexadecimal values with 0x (std::showbase), '
        'or label a decimal record radix=10. '
        'Apart from the short phase markers, emit diagnostics only for failures; no whole-array dumps. Use ordinary statements/loops '
        'and named free functions: no lambdas, local classes, function-pointer aliases or custom '
        'preprocessor macros/conditionals. Do not disable assertions. No external files, processes, '
        'network or nonstandard libraries beyond those provided by the public interface and HLS '
        'toolchain. Keep the complete testbench compact: at most about 120 lines including helpers, loop over an array of at least 8 inputs instead of unrolling one block per case, no comments and no large literal tables. Output one complete cpp code block, no prose.'
    )
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': description}]
    hints = interface_guidance(description, testbench=True)
    if hints and (max_prompt_chars is None or
                  sum(len(row['content']) for row in messages)+len(hints)+2 <= max_prompt_chars):
        messages[0]['content'] += '\n\n'+hints
    if max_prompt_chars is not None and sum(len(row['content']) for row in messages) > max_prompt_chars:
        raise ValueError('Complete task plus reference instructions exceeds max_prompt_chars; task was not truncated')
    return messages


def validate_reference_testbench(source, top, header):
    """Raise ValueError for an unsupported or evidently dependent reference.

    This deliberately accepts only the prompt's simple form. Forbidding every
    DUT identifier outside main catches ordinary transitive helper calls and
    function-pointer aliases without pretending to implement a C++ call graph.
    Header internals and arbitrary data flow are not interpreted. The existing
    quality screen is applied to both the DUT and reference, so an unused
    reference call alone is insufficient; this is still a lexical heuristic.
    """
    # competition imports this module; import its shared screen only at call time.
    from agent.competition import (
        _closing_delimiter, _selftest_main_evidence, _selftest_views, validate_selftest,
    )
    from agent.selftest_quality import validate_selftest_quality

    reference = _reference_name(top)
    validate_selftest(source, top, header)
    _, code = _selftest_views(source)
    if re.search(r'^\s*#\s*(?:define|undef)\b', code, re.M):
        raise ValueError('Reference self-test must not use custom preprocessor macros')
    # Preserve offsets while removing directive bodies and quoted include names.
    code = re.sub(r'^[ \t]*#[^\n]*', lambda m: ' '*len(m.group()), code, flags=re.M)
    main = re.search(r'\bint\s+main\s*\([^(){};]*\)\s*\{', code)
    main_end = _closing_delimiter(code, main.end()-1, '{', '}')
    outside_main = code[:main.start()]+' '*(main_end+1-main.start())+code[main_end+1:]
    if re.search(r'\b'+re.escape(top)+r'\b', outside_main):
        raise ValueError('Reference helpers must be independent: DUT references are allowed only in main')

    definitions = []
    for match in re.finditer(r'\b'+re.escape(reference)+r'\s*\(', outside_main):
        end = _closing_delimiter(outside_main, match.end()-1, '(', ')')
        suffix = re.match(r'\s*(?:noexcept\s*)?\{', outside_main[end+1:])
        if suffix:
            opening = end+1+suffix.end()-1
            closing = _closing_delimiter(outside_main, opening, '{', '}')
            definitions.append((match.start(), opening, closing))
    if len(definitions) != 1:
        raise ValueError('Reference self-test must define one ordinary '+reference+' function with a body')
    start, opening, closing = definitions[0]
    # A nested/local definition is never an ordinary reference helper. Anonymous
    # namespaces are unnecessary for these single-file testbenches and unsupported.
    if outside_main[:start].count('{') != outside_main[:start].count('}'):
        raise ValueError('Reference function must be a free function at file scope')
    reference_body = _selftest_main_evidence(outside_main[opening+1:closing])
    if not reference_body.strip() or re.search(r'\b'+re.escape(reference)+r'\b', reference_body):
        raise ValueError('Reference must contain a nonrecursive computation')

    main_body = _selftest_main_evidence(code[main.end():main_end])
    if re.search(r'\b'+re.escape(reference)+r'\s*(?:=|\{)|\(\s*[*&]\s*'+re.escape(reference)+r'\b', main_body):
        raise ValueError('Self-test must not bind a substitute reference inside main')
    if re.search(r'\b(?:void|bool|char|short|int|long|float|double|auto)\s+'
                 +re.escape(reference)+r'\s*\(', main_body):
        raise ValueError('A local reference declaration is not call evidence')
    direct = [m for m in re.finditer(r'\b'+re.escape(reference)+r'\s*\(', main_body)
              if not re.search(r'(?:\.|->)\s*$', main_body[:m.start()])]
    if not direct:
        raise ValueError('Self-test must call '+reference+' directly inside main')
    validate_selftest_quality(main_body, reference)
    parameters_start = outside_main.index('(', start)
    parameters_end = _closing_delimiter(outside_main, parameters_start, '(', ')')
    try:
        validate_reference_integrity(main_body, top, reference,
            outside_main[parameters_start+1:parameters_end], reference_body)
    except ValueError as exc:
        # Keep the testbench only for the adjacency/overwrite heuristic: it fires
        # on legitimate check layouts, and Vitis still compiles and runs the
        # testbench, so that verdict is recorded rather than enforced. Reference
        # storage that is literally shared with the DUT stays a hard error: that
        # is a self-confirming oracle, not a false alarm.
        if 'passed to the DUT' in str(exc):
            raise
        INTEGRITY_DOWNGRADES.append(str(exc))
    from agent.floating_checks import reject_unguarded_nan_checks
    reject_unguarded_nan_checks(source)
