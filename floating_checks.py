"""Reject a narrow, observed NaN hole in generated tolerance checks.

This is a lexical screen, not floating-point analysis or a tolerance oracle.
It recognizes direct fabs/fabsf/fabsl difference comparisons in main and
simple preceding class guards on those same operands. It neither rewrites a
test nor guesses whether a task permits exceptional values. Unsupported
compound computations can remain outside this check's coverage.
"""
import re


def reject_unguarded_nan_checks(source):
    """Raise for common negative-error tests that silently accept NaN.

    For finite comparisons prefer ``if (!(fabs(a-b) <= tolerance)) return 1``.
    Explicit ``isfinite``/``isnan`` failure guards on both operands also work.
    Integer ``abs`` and task-specific equality/bit tests are not interpreted.
    """
    # Shared lexer masks comments/literals and unevaluated operands. Keep this
    # import local because competition imports the reference-screen module.
    from agent.competition import _selftest_views, _selftest_main_evidence
    from agent.selftest_quality import _end, _unparen, _split, _branches, is_failure_return

    _, code = _selftest_views(source)
    code = re.sub(r'^[ \t]*#[^\n]*', '', code, flags=re.M)
    main = re.search(r'\bint\s+main\s*\([^(){};]*\)\s*\{', code)
    if not main:
        return  # The ordinary selftest screen diagnoses a missing main.
    main_end = _end(code, main.end()-1, '{', '}')
    body = _selftest_main_evidence(code[main.end():main_end])
    branches = _branches(body)

    def canonical(expression):
        return re.sub(r'\s+', '', _unparen(expression))

    def scope(position):
        stack = []
        for index, char in enumerate(body[:position]):
            if char == '{':
                stack.append(index)
            elif char == '}' and stack:
                stack.pop()
        return tuple(stack)

    def names_protected(expression, truth, depth=0):
        """Operands guaranteed non-NaN by this narrow condition's truth."""
        if depth >= 64:
            raise ValueError('Floating self-test class guard exceeds bounded screening depth')
        expression = _unparen(expression)
        # Both sides of a true conjunction/false disjunction hold. Other
        # combinations require proof on every alternative, hence intersection.
        for operators in (('||',), ('&&',)):
            split = _split(expression, operators)
            if split:
                left, operator, right = split
                a, b = names_protected(left, truth, depth+1), names_protected(right, truth, depth+1)
                both = (operator == '&&' and truth) or (operator == '||' and not truth)
                return a | b if both else a & b
        if expression.startswith('!') and not expression.startswith('!='):
            return names_protected(expression[1:], not truth, depth+1)
        match = re.fullmatch(r'(?:(?:::)?std\s*::\s*|::\s*)?(isfinite|isnan)\s*\(([\s\S]*)\)', expression)
        if match and ((match[1] == 'isfinite' and truth) or (match[1] == 'isnan' and not truth)):
            # The entire call must be one expression, not isfinite(a) && ... .
            opening = expression.find('(')
            if _end(expression, opening) == len(expression)-1:
                return {canonical(match[2])}
        return set()

    def unconditional_failure(start, finish):
        """Recognize ordinary failure returns, not nested conditional exits."""
        snippet = body[start:finish].strip()
        if snippet.startswith('{') and _end(snippet, 0, '{', '}') == len(snippet)-1:
            snippet = snippet[1:-1]
        if re.search(r'\b(?:if|for|while|switch|do|return)\b', snippet):
            returns = list(re.finditer(r'\breturn\s+([^;{}]+);', snippet))
            if len(returns) != 1 or re.search(r'\b(?:if|for|while|switch|do)\b', snippet):
                return False
            return is_failure_return(returns[0][1])
        return False

    def comparison_operands(condition):
        condition = _unparen(condition)
        split = _split(condition, ('>=', '>'))
        if not split:
            return None
        left, _, right = split
        # Keep this check narrow: no logical operators or side-effecting
        # thresholds. Positive acceptance !(fabs(...) <= tol) is not matched.
        if _split(right, ('||', '&&', ',', '=')):
            return None
        left = _unparen(left)
        call = re.fullmatch(r'(?:(?:::)?std\s*::\s*|::\s*)?fabs(?:f|l)?\s*\(([\s\S]*)\)', left)
        if not call or _end(left, left.find('(')) != len(left)-1:
            return None
        difference = _split(_unparen(call[1]), ('-',))
        if not difference:
            return None
        a, _, b = difference
        a, b = canonical(a), canonical(b)
        # Ordinary scalar/member/subscript expressions only. Calls, arithmetic
        # and casts would need real data-flow/type analysis, which this is not.
        operand = r'[A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*|\[[A-Za-z_0-9]+\])*'
        if not re.fullmatch(operand, a) or not re.fullmatch(operand, b):
            return None
        return {a, b}

    def unchanged(operand, start, finish):
        base = re.match(r'[A-Za-z_]\w*', operand)
        if not base:
            return False
        name = re.escape(base[0])
        tail = body[start:finish]
        return not re.search(r'\b'+name+r'\b[^;{}]*?(?:=(?!=)|\+\+|--)'
                             r'|(?:\+\+|--)\s*\b'+name+r'\b', tail)

    for start, finish, condition, positive, position in branches:
        if not positive or not unconditional_failure(start, finish):
            continue
        required = comparison_operands(condition)
        if not required:
            continue
        protected = set()
        # A containing positive/negative class branch can establish the facts
        # locally; an unrelated earlier conditional branch cannot.
        for a, b, cond, branch_truth, parent in branches:
            if parent < position and a <= position < b:
                protected |= {operand for operand in names_protected(cond, branch_truth)
                              if unchanged(operand, a, position)}
        for a, b, cond, branch_truth, guard in branches:
            if not branch_truth or b > position or scope(guard) != scope(position):
                continue
            if any(x <= guard < y and not x <= position < y for x, y, _, _, _ in branches):
                continue
            if unconditional_failure(a, b):
                guarded = names_protected(cond, False)
                # Assignments after the guard can invalidate its classification.
                for operand in guarded:
                    if unchanged(operand, b, position):
                        protected.add(operand)
        if not required <= protected:
            raise ValueError('Floating self-test tolerance failure can silently accept NaN: '
                             'use positive acceptance !(fabs(actual-expected) <= tolerance), '
                             'or explicit classification failure guards on the same operands')
