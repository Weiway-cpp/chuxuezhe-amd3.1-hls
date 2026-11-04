"""Bind public function declarations while a generated selftest is compiled.

The inserted code only takes function addresses. It never calls a helper or
changes expected values. This closes the gap where a working top-only testbench
could hide a missing public auxiliary function; it is not a functional oracle
or a complete C++ interface/ABI proof. Public declaration discovery remains
conservative, with every known omission retained in the returned metadata.
"""
import hashlib
from pathlib import Path
import re

from agent.public_interface import MAX_HEADER_BYTES, public_functions


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _original_offset(source, logical_offset):
    """Map an offset after C++ line splicing back to unchanged model bytes."""
    offset = logical_offset
    for match in re.finditer(r'\\\r?\n', source):
        if match.start() > offset:
            break
        offset += match.end()-match.start()
    return offset


def instrument_selftest(source, top, support_files):
    """Return (checked_source, metadata), without mutating the original file.

    Call this after structural validation, immediately before each functional
    verifier dispatch, including a revised selftest. Review and reference
    screening must continue to use the raw source. Only one ordinary file-scope
    ``int main`` is supported. A separate nested scope prevents the injected
    identifiers from colliding with the testbench's local declarations/gotos.
    """
    # competition imports this module, so share its lexer only at call time.
    from agent.competition import _closing_delimiter, _selftest_views

    if not isinstance(source, str) or not isinstance(top, str) or not re.fullmatch(r'[A-Za-z_]\w*', top, re.ASCII):
        raise ValueError('Public-link instrumentation needs source and an ordinary top identifier')
    if not support_files:
        raise ValueError('Public-link instrumentation requires public headers')
    functions, omissions = public_functions(support_files)
    symbols = list(dict.fromkeys([top, *functions]))
    _, code = _selftest_views(source)
    if re.search(r'^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b', code, re.M):
        raise ValueError('Conditional selftest preprocessing cannot be instrumented safely')
    protected = '|'.join(re.escape(name) for name in ['main', *symbols])
    if re.search(r'^\s*#\s*(?:define|undef)\s+(?:'+protected+r')\b', code, re.M):
        raise ValueError('Selftest must not replace a public function or main with a macro')
    # Blank directives without shifting the offsets used to insert into source.
    code = re.sub(r'^[ \t]*#[^\n]*', lambda m: ' '*len(m[0]), code, flags=re.M)
    entries = list(re.finditer(r'\bint\s+main\s*\([^(){};]*\)\s*\{', code))
    if len(entries) != 1:
        raise ValueError('Public-link instrumentation requires one ordinary int main definition')
    main = entries[0]
    if code[:main.start()].count('{') != code[:main.start()].count('}'):
        raise ValueError('Public-link instrumentation requires main at file scope')
    _closing_delimiter(code, main.end()-1, '{', '}')
    # A helper supplied by the testbench would satisfy linkage without proving
    # that the candidate supplies the public API. Keep that failure explicit.
    for symbol in symbols:
        for call in re.finditer(r'\b'+re.escape(symbol)+r'\s*\(', code):
            end = _closing_delimiter(code, call.end()-1, '(', ')')
            suffix = code[end+1:]
            if re.match(r'\s*(?:noexcept(?:\s*\([^{};]*\))?\s*)?'
                        r'(?:\[\[[^;{}]*?\]\]\s*)*(?:->\s*[^;{}]+)?\{', suffix):
                raise ValueError('Selftest must not define public function '+symbol)

    identifiers = set(re.findall(r'[A-Za-z_]\w*', re.sub(r'\\\r?\n', '', source)))
    # Include public-header macro names in collision avoidance. Read no private
    # implementation, judge testbench, or include outside the supplied files.
    for path in support_files.values():
        if path is not None:
            try:
                with Path(path).open('rb') as stream:
                    raw = stream.read(MAX_HEADER_BYTES+1)
                if len(raw) <= MAX_HEADER_BYTES:
                    identifiers.update(re.findall(r'[A-Za-z_]\w*',
                        re.sub(r'\\\r?\n', '', raw.decode('utf-8-sig'))))
            except (OSError, UnicodeError):
                pass  # Discovery has already reported this omission.
    counter = 0
    injected = ['\n    { // Public interface link requirements; no helper is executed.\n']
    generated = []
    for symbol in symbols:
        while True:
            name = 'hls_agent_public_link_'+str(counter)
            counter += 1
            if name not in identifiers:
                break
        identifiers.add(name)
        generated.append(name)
        # Global qualification prevents a main parameter/local name from binding
        # a substitute. Volatile initialization retains the linker relocation.
        injected.append('        auto volatile '+name+' = &::'+symbol+';\n')
        injected.append('        (void)'+name+';\n')
    injected.append('    }\n')
    offset = _original_offset(source, main.end()-1)+1
    if source[offset-1:offset] != '{':
        raise ValueError('Cannot map selftest entry to original source safely')
    insertion = ''.join(injected)
    checked = source[:offset]+insertion+source[offset:]
    return checked, {
        'source_sha256': _sha(source), 'instrumented_sha256': _sha(checked),
        'symbols': symbols, 'discovery_omissions': omissions,
        'insertion_offset': offset, 'insertion_chars': len(insertion),
        'local_identifiers': generated, 'helpers_executed': False,
        'scope': 'compile_link_requirements_for_top_and_discovered_ordinary_public_functions',
        'linked': None,  # Instrumentation alone has no compiler verdict.
    }
