"""Screen obvious vacuous checks in a generated, simple C++ main body.

This is a bounded lexical heuristic, not C++ data-flow or coverage analysis.
The caller supplies code with comments/string contents and unevaluated operands
already masked, and rejects local classes/lambdas/preprocessor substitutes.
We require one nonconstant check related to a direct DUT call. Arguments count
as related because their input/output roles are not available here; ordinary
assignments and output aliases are followed conservatively. This cannot prove
that an oracle is independent, that checks execute, or that all outputs/cases
are covered. Self-comparisons are unsupported even for floating-point NaN
checks; use isfinite/isnan explicitly. Unsupported forms should use the
existing unverified fallback.
"""
import re


_IDENT = r'[A-Za-z_]\w*'
_NUMBER = re.compile(r"(?<![\w.])(?:\d|\.\d)(?:[eEpP][+-]|[\w.]|'[\w])*")
_IGNORED = {'true', 'false', 'nullptr', 'NULL', 'std', 'const', 'volatile',
            'int', 'long', 'short', 'unsigned', 'signed', 'float', 'double',
            'char', 'bool', 'auto', 'return', 'if', 'else', 'assert'}
_MAX_DEPTH = 64


def _end(code, start, opening='(', closing=')'):
    depth = 0
    for i in range(start, len(code)):
        if code[i] == opening:
            depth += 1
            if depth > _MAX_DEPTH:
                raise ValueError('Self-test expression nesting exceeds the supported screening depth')
        elif code[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('Self-test quality check encountered unbalanced delimiters')


def _unparen(expression):
    expression = expression.strip()
    while expression.startswith('(') and _end(expression, 0) == len(expression)-1:
        expression = expression[1:-1].strip()
    return expression


def _split(expression, operators):
    """Split one outer operator, ignoring function/array argument interiors."""
    depth = 0
    for i, char in enumerate(expression):
        if char in '([':
            depth += 1
        elif char in ')]':
            depth -= 1
        elif depth == 0:
            for op in operators:
                if expression.startswith(op, i):
                    return expression[:i], op, expression[i+len(op):]
    return None


def _truth(expression, depth=0):
    """Classify literals and unsupported identity checks, not arbitrary C++.

    Identical operand comparisons are deliberately treated as unsupported
    checks, not a claim about IEEE floating-point NaNs or volatile reads.
    """
    if depth >= _MAX_DEPTH:
        raise ValueError('Self-test expression is too complex for bounded screening')
    expression = _unparen(expression)
    if expression in ('true', 'false'):
        return expression == 'true'
    if re.fullmatch(r'(?:0[xX][\da-fA-F]+|0[bB][01]+|0[0-7]*|[1-9]\d*)[uUlL]*', expression):
        digits = re.sub(r'[uUlL]+$', '', expression)
        base = 16 if digits.lower().startswith('0x') else 2 if digits.lower().startswith('0b') else 8 if digits.startswith('0') else 10
        return int(digits, base) != 0
    for operators in (('||',), ('&&',)):
        split = _split(expression, operators)
        if split:
            lhs, op, rhs = split
            a, b = _truth(lhs, depth+1), _truth(rhs, depth+1)
            if op == '||':
                return True if a is True or b is True else False if a is False and b is False else None
            return False if a is False or b is False else True if a is True and b is True else None
    split = _split(expression, ('==', '!=', '<=', '>=', '<', '>'))
    if split:
        lhs, op, rhs = split
        lhs, rhs = _unparen(lhs), _unparen(rhs)
        # Calls/increments may have side effects, so identical spellings alone
        # do not make those expressions a constant comparison.
        if (re.sub(r'\s+', '', lhs) == re.sub(r'\s+', '', rhs)
                and lhs and not re.search(r'\(|\+\+|--|=', lhs)):
            return op in ('==', '<=', '>=')
    if expression.startswith('!') and not expression.startswith('!='):
        truth = _truth(expression[1:], depth+1)
        return None if truth is None else not truth
    return None


def _names(expression):
    return set(re.findall(_IDENT, _NUMBER.sub(' ', expression))) - _IGNORED


def _statement_end(code, start, depth=0):
    if depth >= _MAX_DEPTH:
        raise ValueError('Self-test statement nesting exceeds the supported screening depth')
    while start < len(code) and code[start].isspace():
        start += 1
    if start >= len(code):
        return start
    if code[start] == '{':
        return _end(code, start, '{', '}') + 1
    nested = re.match(r'if\s*\(', code[start:])
    if nested:
        cond_end = _end(code, start+nested.end()-1)
        finish = _statement_end(code, cond_end+1, depth+1)
        alternate = re.match(r'\s*else\b', code[finish:])
        return _statement_end(code, finish+alternate.end(), depth+1) if alternate else finish
    paren = bracket = 0
    for i in range(start, len(code)):
        char = code[i]
        paren += (char == '(') - (char == ')')
        bracket += (char == '[') - (char == ']')
        if char == ';' and paren == bracket == 0:
            return i+1
    return len(code)


def _branches(body):
    branches = []
    for match in re.finditer(r'\bif\s*\(', body):
        end = _end(body, match.end()-1)
        condition = body[match.end():end]
        finish = _statement_end(body, end+1)
        branches.append((end+1, finish, condition, True, match.start()))
        alternate = re.match(r'\s*else\b', body[finish:])
        if alternate:
            start = finish+alternate.end()
            branches.append((start, _statement_end(body, start), condition, False, match.start()))
    return branches


def is_failure_return(expression):
    expression = _unparen(expression)
    if expression == 'EXIT_FAILURE':
        return True
    literal = re.fullmatch(r'(0[xX][\da-fA-F]+|0[bB][01]+|0[0-7]*|[1-9]\d*)'
                           r'(?:[uU](?:ll|LL|[lL])?|(?:ll|LL|[lL])[uU]?)?', expression)
    if literal is None:
        return False
    expression = literal[1]
    base = 16 if expression.lower().startswith('0x') else 2 if expression.lower().startswith('0b') else 8 if expression.startswith('0') else 10
    return 1 <= int(expression, base) <= 255


def validate_selftest_quality(body, top):
    """Require one plausible DUT-related failure check; raise ValueError otherwise.

    Accepts assert(expr), or a portable failure return under an if/else whose
    condition depends on the DUT call or related values. Other checks may be
    present, but constant assertions and unconditional returns add no evidence.
    """
    branches = _branches(body)
    related = {}
    top_call = re.compile(r'\b'+re.escape(top)+r'\s*\(')

    def active(position):
        return not any(a <= position < b and _truth(cond) is (not positive)
                       for a, b, cond, positive, _ in branches)

    def direct_calls(expression, offset=0):
        return [m for m in top_call.finditer(expression)
                if not re.search(r'(?:\.|->)\s*$', expression[:m.start()])
                and active(offset+m.start())]

    # Use positions so a check before the first DUT call cannot validate it.
    for call in direct_calls(body):
        end = _end(body, call.end()-1)
        for name in _names(body[call.end():end]):
            related[name] = min(related.get(name, end), end)
        # A scalar return may initialize an object without '='. Only recognize
        # one complete initializer; do not confuse an array, function prototype
        # or a later unrelated statement with the receiver of this DUT call.
        receiver = re.search(
            r'(?:^|[;{}])\s*(?!(?:return|throw|new|delete|else)\b)'
            r'(?:(?:const|volatile|signed|unsigned|long|short)\s+)*'
            r'(?:'+_IDENT+r'\s*::\s*)*'+_IDENT+r'\s+'
            r'('+_IDENT+r')\s*([({])\s*$', body[:call.start()])
        if receiver:
            closing = ')' if receiver[2] == '(' else '}'
            suffix = re.match(r'\s*'+re.escape(closing)+r'\s*;', body[end+1:])
            if suffix:
                position = end+1+suffix.end()
                name = receiver[1]
                related[name] = min(related.get(name, position), position)

    assignments = list(re.finditer(
        r'\b('+_IDENT+r')(?:\s*\[[^;{}]*?\]|\s*\.\s*'+_IDENT+r')*\s*'
        r'(?:[+|&^*-]=|=(?!=))\s*([^;{}]+)', body))
    # Fixed point covers simple output aliases and helper-derived comparisons.
    # The bound keeps work proportional to this small, generated testbench.
    for _ in range(len(assignments)+1):
        changed = False
        for assign in assignments:
            if not active(assign.start()):
                continue
            lhs, rhs = assign[1], assign[2]
            available = [related[n] for n in _names(rhs)
                         if _truth(rhs) is None and related.get(n, len(body)+1) <= assign.start()]
            if direct_calls(rhs, assign.start(2)) and _truth(rhs) is None:
                available.append(assign.end())
            # A mismatch flag set in a DUT-related branch is also a check.
            available.extend(related[n] for a, b, cond, _, position in branches
                             if a <= assign.start() < b and _truth(cond) is None
                             for n in _names(cond) if related.get(n, len(body)+1) <= position)
            if available:
                position = max(assign.end(), min(available))
                if position < related.get(lhs, len(body)+1):
                    related[lhs] = position
                    changed = True
        if not changed:
            break

    def meaningful(expression, position):
        return (_truth(expression) is None and
                (bool(direct_calls(expression, position)) or
                 any(related.get(n, len(body)+1) <= position for n in _names(expression))))

    for check in re.finditer(r'\bassert\s*\(', body):
        end = _end(body, check.end()-1)
        if active(check.start()) and meaningful(body[check.end():end], check.start()):
            return
    for check in re.finditer(r'\breturn\s+([^;{}]+);', body):
        if not is_failure_return(check[1]) or not active(check.start()):
            continue
        if any(a <= check.start() < b and meaningful(cond, position)
               for a, b, cond, _, position in branches):
            return
    raise ValueError('Self-test needs a nonconstant DUT-related assertion or conditional failure check; '
                     'constant assertions, self-comparisons and unconditional returns are not evidence')
