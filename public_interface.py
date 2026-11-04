"""Conservative discovery of ordinary functions in the supplied public headers.

This is not a C++ parser or a promise that every interface is covered. The probe
uses the compiler's actual declarations (``&name``), never reconstructed types.
Unknown conditional preprocessing, templates, overloads, members and namespaces
are deliberately left out of the extra checks and reported as limitations.
"""
from pathlib import Path
import re


_LEX = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|R"(?P<raw>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=raw)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|::|\S')
_IDENT = re.compile(r'[A-Za-z_]\w*\Z')
MAX_HEADER_BYTES = 1024 * 1024


def _surface(text):
    # Keep quoted literals intact; comment-looking bytes inside them are data.
    text = re.sub(r'\\\r?\n', '', text)
    text = _LEX.sub(lambda m: ('\n' * m[0].count('\n') or ' ')
                    if m[0].startswith(('//', '/*')) else m[0], text)
    lines = text.splitlines()
    meaningful = [line.strip() for line in lines if line.strip()]
    guard = None
    if len(meaningful) >= 2:
        first = re.fullmatch(r'#\s*ifndef\s+([A-Za-z_]\w*)', meaningful[0])
        if first and re.fullmatch(r'#\s*define\s+'+re.escape(first[1]), meaningful[1]):
            guard = first[1]
    conditions, output = [], []
    for line in lines:
        directive = re.match(r'^\s*#\s*(\w+)\b(.*)', line)
        if not directive:
            if all(conditions):
                output.append(line)
            continue
        kind, value = directive[1], directive[2].strip()
        if kind in ('if', 'ifdef', 'ifndef'):
            if kind == 'ifndef' and value == guard and not conditions:
                enabled = True
            elif kind in ('ifdef', 'ifndef') and value == '__cplusplus':
                enabled = kind == 'ifdef'
            elif kind == 'if' and value in ('0', '1'):
                enabled = value == '1'
            elif kind == 'if' and re.fullmatch(r'defined\s*(?:\(\s*__cplusplus\s*\)|__cplusplus)', value):
                enabled = True
            else:
                # No guesses about macros defined by Vitis/includes or options.
                return None, 'conditional preprocessing requires a real preprocessor'
            conditions.append(enabled)
        elif kind == 'else':
            if not conditions:
                return None, 'unbalanced preprocessing'
            conditions[-1] = not conditions[-1]
        elif kind == 'endif':
            if not conditions:
                return None, 'unbalanced preprocessing'
            conditions.pop()
        elif kind == 'elif':
            return None, 'conditional preprocessing requires a real preprocessor'
    if conditions:
        return None, 'unbalanced preprocessing'
    return '\n'.join(output), None


def _function(statement):
    """Return an identifiable function name and whether it is a plain prototype."""
    if '(' not in statement or 'typedef' in statement or 'using' in statement:
        return None, False
    opening = statement.index('(')
    if opening < 2 or not _IDENT.fullmatch(statement[opening - 1]):
        return None, False
    name, prefix = statement[opening - 1], statement[:opening - 1]
    if prefix[-1] == '::' or '=' in prefix:
        return None, False
    depth, closing = 0, None
    for i in range(opening, len(statement)):
        if statement[i] == '(':
            depth += 1
        elif statement[i] == ')':
            depth -= 1
            if depth == 0:
                closing = i
                break
    if closing is None:
        return name, False
    tail = statement[closing + 1:]
    unsupported = {'static', 'inline', 'constexpr', 'consteval', 'template',
                   'typedef', 'using', 'friend', 'operator', '__attribute__', '__declspec'}
    plain = not (set(prefix) & unsupported) and tail in ([], ['noexcept'])
    # A parameter list is allowed to contain arrays, references, function
    # pointers and templates: no type reconstruction is attempted here.
    return name, plain


def _prototypes(tokens, skipped):
    found, statement, index = [], [], 0
    while index < len(tokens):
        token = tokens[index]
        if token == '{':
            depth, end = 1, index + 1
            while end < len(tokens) and depth:
                depth += (tokens[end] == '{') - (tokens[end] == '}')
                end += 1
            if depth:
                skipped.append('unbalanced declaration braces')
                break
            if statement in (['extern', '"C"'], ['extern', '"C++"']):
                found.extend(_prototypes(tokens[index + 1:end - 1], skipped))
            else:
                name, _ = _function(statement)
                if name:
                    found.append((name, tuple(statement), False))
                if 'namespace' in statement:
                    skipped.append('namespace declarations are not discovered')
                # Definitions, classes, enums and aggregate initializers are
                # opaque: their members/local prototypes are not public globals.
            statement = []
            index = end
            continue
        if token == ';':
            name, plain = _function(statement)
            if name:
                found.append((name, tuple(statement), plain))
            statement = []
        else:
            statement.append(token)
        index += 1
    return found


def public_functions(supports):
    """Return discovered names plus explicit omissions, reading only public input."""
    declarations, skipped = {}, []
    for filename, path in supports.items():
        if path is None:
            # Legacy link_entry callers can supply names alone; production
            # verify_compile requires validated, existing paths before this call.
            skipped.append({'header': filename, 'reason': 'header text unavailable'})
            continue
        try:
            with Path(path).open('rb') as stream:
                raw = stream.read(MAX_HEADER_BYTES + 1)
            if len(raw) > MAX_HEADER_BYTES:
                skipped.append({'header': filename, 'reason': 'header exceeds discovery bound'})
                continue
            surface, reason = _surface(raw.decode('utf-8-sig'))
        except (OSError, UnicodeError) as exc:
            skipped.append({'header': filename, 'reason': 'header text unavailable: '+type(exc).__name__})
            continue
        if reason:
            skipped.append({'header': filename, 'reason': reason})
            continue
        limitations = []
        tokens = [match[0] for match in _LEX.finditer(surface)]
        for name, signature, plain in _prototypes(tokens, limitations):
            declarations.setdefault(name, []).append((signature, plain))
        skipped.extend({'header': filename, 'reason': reason} for reason in limitations)
    names = []
    for name, instances in declarations.items():
        signatures = {signature for signature, _ in instances}
        if len(signatures) == 1 and all(plain for _, plain in instances):
            names.append(name)
        else:
            skipped.append({'symbol': name, 'reason': 'overloaded, defined in header, or unsupported declaration'})
    return sorted(names), skipped
