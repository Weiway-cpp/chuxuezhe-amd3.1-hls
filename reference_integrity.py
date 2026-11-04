"""Reject simple reference/output contamination in the requested testbench form.

This is a bounded lexical screen, not C++ alias, type, or control-flow analysis.
Only ordinary named objects and direct assignments/calls are tracked. Unknown
typedef roles and complex output expressions are left to the existing screen;
passing this check never establishes reference correctness or independence.
"""
import re

from agent.selftest_quality import _end, _names, _branches, _truth, is_failure_return

_ID = r'[A-Za-z_]\w*'
_ASSIGN = re.compile(r'\b('+_ID+r')(?:\s*\[[^;{}]*?\]|\s*\.\s*'+_ID+r')*\s*'
                     r'(?P<op>[+|&^*-]=|=(?!=))\s*([^;{}]+)')


def _arguments(text, *, signature=False):
    depth, start, result = 0, 0, []
    for i, ch in enumerate(text):
        if ch in '([{' or (signature and ch == '<'): depth += 1
        elif ch in ')]}' or (signature and ch == '>'): depth -= 1
        elif ch == ',' and depth == 0:
            result.append(text[start:i].strip()); start = i+1
    result.append(text[start:].strip())
    return result if result != [''] else []


def _object(expression):
    """One named storage root; complex expressions deliberately return None."""
    expression = expression.strip()
    while expression.startswith('(') and _end(expression, 0) == len(expression)-1:
        expression = expression[1:-1].strip()
    match = re.fullmatch(r'[&*]?\s*('+_ID+r')(?:\s*\[[^;{}()]*\]|\s*\.\s*'+_ID+r')*', expression)
    return match[1] if match else None


def _receiver(body, start, end):
    prefix = body[:start]
    # A complete ordinary RHS call, or a direct scalar initializer.
    assignment = re.search(r'\b('+_ID+r')\s*=\s*$', prefix)
    if assignment and re.match(r'\s*;', body[end+1:]):
        return assignment[1]
    direct = re.search(r'(?:^|[;{}])\s*(?:const\s+)?'+_ID+r'\s+('+_ID+r')\s*([({])\s*$', prefix)
    if direct and re.match(r'\s*'+re.escape(')' if direct[2]=='(' else '}')+r'\s*;', body[end+1:]):
        return direct[1]
    return None


def _writable_positions(signature, reference_body):
    """Recognize explicit writes, never infer output roles merely from a type."""
    result = []
    for index, param in enumerate(_arguments(signature, signature=True)):
        # No function-pointer declarators or parameter defaults are inferred.
        if '(' in param or '=' in param:
            continue
        match = re.search(r'\b('+_ID+r')\s*(?:\[[^\]]*\]\s*)*$', param)
        if not match: continue
        name, prefix = match[1], param[:match.start(1)]
        first_indirection = re.search(r'[*&]', prefix)
        is_const = 'const' in re.findall(_ID, prefix[:first_indirection.start()] if first_indirection else prefix)
        assign = r'\s*(?:[+|&^*-]=|=(?!=))'
        indexed_write = re.search(r'\b'+re.escape(name)+r'\s*(?:\[[^;{}]*?\])+'+assign, reference_body)
        dereference_write = re.search(r'\*\s*'+re.escape(name)+r'\b'+assign, reference_body)
        pointer_member_write = re.search(r'\b'+re.escape(name)+r'\s*->\s*'+_ID+assign, reference_body)
        reference_write = ('&' in prefix and re.search(r'\b'+re.escape(name)+
            r'(?:\s*\.\s*'+_ID+r')*'+assign, reference_body))
        if not is_const and (indexed_write or dereference_write or pointer_member_write or reference_write):
            result.append(index)
    return result


_CERR_HEAD = re.compile(r'\bstd\s*::\s*cerr\s*<<')


def _cerr_tokens_ok(text):
    """Linear check for the token grammar allowed inside a std::cerr chain."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace() or ch in '[]':
            i += 1
        elif text.startswith('::', i) or text.startswith('<<', i):
            i += 2
        elif ch == '.':
            i += 1
        elif ch.isdigit():
            while i < n and text[i].isdigit():
                i += 1
        elif ch.isalpha() or ch == '_':
            i += 1
            while i < n and (text[i].isalnum() or text[i] == '_'):
                i += 1
        else:
            return False
    return True


def _strip_cerr_statements(statement):
    """Remove ``std::cerr << ...;`` spans with a linear scan.

    The previous single-regex form could backtrack catastrophically on
    adversarial testbench text, hanging the whole solve past its deadline.
    """
    pieces, pos = [], 0
    while True:
        match = _CERR_HEAD.search(statement, pos)
        if not match:
            break
        depth, i, end = 0, match.end(), -1
        stop = min(len(statement), match.end() + 4096)
        while i < stop:
            ch = statement[i]
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                if depth == 0:
                    break
                depth -= 1
            elif ch == ';' and depth == 0:
                end = i + 1
                break
            i += 1
        if end < 0 or not _cerr_tokens_ok(statement[match.end():end-1]):
            pieces.append(statement[pos:match.end()])
            pos = match.end()
            continue
        pieces.append(statement[pos:match.start()])
        pos = end
    pieces.append(statement[pos:])
    return ''.join(pieces)


def _adjacent_failure_bridges(body, branches):
    """Recognize adjacent scalar equality checks, not arbitrary control flow.

    Literal/comment text is already blanked by the caller. A diagnostic stream
    statement may precede return 1, but calls, assignments, increments and other
    statements are deliberately unsupported. Ordinary braces match the actual
    v35 CLZ testbench. Bare nested if/loop/else bodies must not become bridges.
    """
    checks = []
    for start, end, condition, positive, position in branches:
        relation = re.fullmatch(r'\s*('+_ID+r')\s*!=\s*('+_ID+r')\s*', condition)
        if not positive or not relation:
            continue
        if not re.search(r'(?:^|[;{}])\s*$', body[:position]):
            continue
        statement = body[start:end].strip()
        if statement.startswith('{'):
            if _end(statement, 0, '{', '}') != len(statement)-1:
                continue
            statement = statement[1:-1].strip()
        # Only a narrowly recognized stream expression; never strip a call or
        # a mutation merely because its statement also prints a diagnostic.
        statement = _strip_cerr_statements(statement)
        if re.fullmatch(r'\s*return\s+1\s*;\s*', statement):
            checks.append((position, end, relation[1], relation[2]))
    result = {}
    for previous, current in zip(checks, checks[1:]):
        if previous[3] == current[3] and not body[previous[1]:current[0]].strip():
            result[current[0]] = previous[2]
    return result


def validate_reference_integrity(body, top, reference, signature, reference_body):
    """Screen known output roots and simple overwrites, preserving unknown forms.

    Each recognized reference output must retain a reference-derived check.
    Common read-only inputs are not outputs. Direct sharing of a known writable
    reference buffer with a DUT call is unsupported; independent copies are
    required by the prompt. The screen does not infer arbitrary aliases.
    """
    writable = _writable_positions(signature, reference_body)
    calls = []
    for name in (reference, top):
        for match in re.finditer(r'\b'+re.escape(name)+r'\s*\(', body):
            if re.search(r'(?:\.|->)\s*$', body[:match.start()]): continue
            end = _end(body, match.end()-1)
            args = _arguments(body[match.end():end])
            calls.append({'name':name, 'start':match.start(), 'end':end,
                          'args':args, 'roots':[_object(arg) for arg in args],
                          'receiver':_receiver(body,match.start(),end)})
    ref_calls = [call for call in calls if call['name']==reference]
    dut_calls = [call for call in calls if call['name']==top]
    reference_outputs = set()
    for call in ref_calls:
        call['outputs'] = {call['roots'][i] for i in writable if i < len(call['roots']) and call['roots'][i]}
        if call['receiver']: call['outputs'].add(call['receiver'])
        reference_outputs.update(call['outputs'])
        shared = {root for dut in dut_calls for root in dut['roots'] if root} & call['outputs']
        if shared:
            raise ValueError('Reference output storage is also passed to the DUT; use independent input/output copies')

    # No reliable output receiver/argument was recognized: do not guess roles
    # for typedef-by-value or complex expressions. Existing checks still apply.
    if not reference_outputs:
        return
    # Only ordinary scalar return receivers are known DUT outputs here. The
    # reference may use different argument order/roles even at equal arity;
    # without the DUT's type information never infer output roles from it.
    dut_outputs=set()
    for call in dut_calls:
        call['outputs'] = set()
        if call['receiver']: call['outputs'].add(call['receiver'])
        dut_outputs.update(call['outputs'])
    branches = _branches(body)
    bridges = _adjacent_failure_bridges(body, branches)
    def active(position):
        return not any(a <= position < b and _truth(cond) is (not positive)
                       for a,b,cond,positive,_ in branches)
    origins, validated = {}, set()
    events=[]
    for call in calls:
        events.append((call['end'],0,'call',call))
    for match in _ASSIGN.finditer(body):
        events.append((match.end(),1,'assign',match))
    for match in re.finditer(r'\bassert\s*\(',body):
        end=_end(body,match.end()-1)
        events.append((end,2,'check',(body[match.end():end],match.start())))
    for match in re.finditer(r'\breturn\s+([^;{}]+);',body):
        if is_failure_return(match[1]):
            for a,b,cond,_,position in branches:
                if a <= match.start() < b:
                    events.append((position,2,'check',(cond,position)))

    for position,_,kind,event in sorted(events,key=lambda value:(value[0],value[1])):
        if not active(position): continue
        if kind=='call':
            for output in event['outputs']: origins[output]={(event['name'],output)}
        elif kind=='assign':
            lhs,rhs=event[1],event[3]
            from_call = {(call['name'],root) for call in calls if event.start(3)<=call['start']<event.end()
                         for root in call['outputs']}
            derived=set().union(*(origins.get(name,set()) for name in _names(rhs)))
            if event['op']!='=': derived |= origins.get(lhs,set())
            # Preserve an explicit constant failure flag controlled by a
            # reference-related branch. A value copied from the DUT must not
            # acquire reference provenance merely from the enclosing condition.
            for a,b,condition,_,cond_pos in branches:
                if (lhs not in reference_outputs | dut_outputs and _truth(rhs) is not None
                        and a<=event.start()<b and _truth(condition) is None):
                    derived |= set().union(*(origins.get(name,set()) for name in _names(condition)))
            origins[lhs]=from_call|derived
        else:
            expression,check_pos=event
            if _truth(expression) is not None: continue
            checked=set().union(*(origins.get(name,set()) for name in _names(expression)))
            if check_pos in bridges:
                checked |= origins.get(bridges[check_pos], set())
            # Require a still-DUT-derived observation in the same comparison
            # when simple DUT outputs were recognized. This catches copying the
            # expectation over the actual result before comparing them.
            if not dut_outputs or any(origin==top for origin,_ in checked):
                validated |= {root for origin,root in checked if origin==reference}
    if reference_outputs - validated:
        raise ValueError('Reference outputs must reach a failure check without being overwritten by DUT results or constants')
