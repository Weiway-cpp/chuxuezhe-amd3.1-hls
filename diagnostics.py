"""Bounded repair feedback selected from full logs; never success evidence."""
import re


INFRA_ERROR = re.compile(
    r'license checkout failed|failed to check out (?:a )?license|'
    r'no valid license|cannot find (?:a )?license|'
    r'valid license was not found|could not obtain the necessary license|'
    r'unable to get licensing information|license checkout failure|'
    r'does not have an appropriate license|'
    r'NtCreateDirectoryObject\([^\r\n]*\):\s*0xC0000022|'
    r'permission denied|access is denied|no space left on device|'
    r'out of memory|cannot allocate memory|'
    r'failed to (?:launch|spawn) (?:process|command)|'
    r'couldn.t execute .+ no such file or directory', re.I)
SPECIFIC = re.compile(
    r'fatal error|(?-i:\berror:)|undefined reference|\bassertion\b|\bassert\s*\(|'
    r'\bmismatch\b|differs from reference|\bFAIL\s+(?:case|trial|index)\s*=|'
    r'segmentation (?:fault|violation)|access violation|bus error|floating point exception|'
    r'\[HLS \d+-\d+\].*(?:unsupported|cannot)', re.I)
GENERIC = re.compile(r'\berror\b|\bfail(?:ed|ure)?\b|\bfatal\b|@E', re.I)
VALUES = re.compile(r'\b(?:case|trial|input|expected|actual|index|SELFTEST_PHASE)\s*[:=]', re.I)
SCAFFOLD = re.compile(
    r'^\s*(?:while executing|invoked from within|uplevel|source |agent_stage|exit [01]|'
    r'\(.*(?:body|file).*line\s+\d+)', re.I)
NO_ERRORS = re.compile(r'^\s*(?:INFO:\s*)?0 errors?(?:\s*[,;]\s*\d+ warnings?)?\.?\s*$', re.I)

# Parse the diagnostic location, never a filename mentioned in its message.
# Unsupported compiler formats remain unclassified; excerpts are not evidence.
_COMPILER_FILE = r'''(?P<file>(?:[A-Za-z]:)?[^:\r\n|<>"']*?\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx))'''
_COMPILER_ERRORS = (
    re.compile(r'^\s*' + _COMPILER_FILE + r':(?P<line>\d+)(?::(?P<column>\d+))?:\s*'
               r'(?:fatal\s+)?error:\s*(?P<message>.*)$', re.I),
    re.compile(r'^\s*' + _COMPILER_FILE + r'\((?P<line>\d+)(?:,\s*(?P<column>\d+))?\)\s*:\s*'
               r'(?:fatal\s+)?error(?:\s+[A-Z]+\d+)?:\s*(?P<message>.*)$', re.I),
)
_LOCATED_DIAGNOSTIC = re.compile(
    r'^\s*' + _COMPILER_FILE + r':(?P<line>\d+)(?::\d+)?:\s*'
    r'(?P<severity>error|warning):\s*(?P<message>.*)$', re.I)
_PREPROCESSOR_QUOTE = re.compile(
    r'^\s*(?P<line>\d+)\s*\|\s*#\s*(?P<kind>error|warning)\s+(?P<message>.*?)\s*$')


def _compiler_error(line):
    return next((match for pattern in _COMPILER_ERRORS
                 if (match := pattern.fullmatch(line))), None)


def _source_quote(line):
    return bool(re.match(r'^\s*(?:\d+\s*)?\|', line))


def _infrastructure_line(line):
    # GCC/Clang quote source and user-authored diagnostic strings in tool logs.
    # Those strings are not evidence of license, memory or filesystem failure.
    if _source_quote(line):
        return False
    error = _compiler_error(line)
    if error and re.match(r'(?:static assertion failed\b|#\s*error\b)', error['message'], re.I):
        return False
    return bool(INFRA_ERROR.search(line))


def has_infrastructure_error(output):
    """Scan complete logs, excluding recognized source quotes/user diagnostics.

    This is a format-aware heuristic, not a general log provenance proof.
    Independent infrastructure messages remain evidence even in mixed logs.
    """
    rows = output.splitlines()
    for index, line in enumerate(rows):
        if not _infrastructure_line(line):
            continue
        diagnostic = _LOCATED_DIAGNOSTIC.fullmatch(line)
        if diagnostic:
            following = index+1
            # A Windows tool can emit CRCRLF; splitlines leaves a blank row.
            while following < len(rows) and not rows[following].strip():
                following += 1
            quote = _PREPROCESSOR_QUOTE.fullmatch(rows[following]) if following < len(rows) else None
            if quote and quote['line'] == diagnostic['line']:
                # Clang drops #error/#warning from the diagnostic; GCC retains
                # it. Match only this adjacent source line and message, never
                # ignore a whole compiler block or a standalone tool failure.
                message = re.sub(r'\s+\[-W[^\]]+\]$', '', diagnostic['message'])
                message = re.sub(r'^#\s*'+quote['kind']+r'\s+', '', message)
                if (' '.join(message.split()) == ' '.join(quote['message'].split()) and
                        (quote['kind']=='warning' or diagnostic['severity'].lower()=='error')):
                    continue
        return True
    return False


def compiler_error_locations(lines, *, max_records=16):
    """Count located errors across full logs, retaining only bounded examples.

    Locations identify where the compiler reported an error, not its ultimate
    cause. Counts are occurrences (tool and stage logs may repeat a message).
    This does not infer success, or attribute unlocated/linker errors.
    """
    counts = {'design': 0, 'selftest': 0, 'other': 0}
    records = []
    for value in lines:
        for line in str(value).splitlines():
            found = _compiler_error(line)
            if not found:
                continue
            path = found['file'].strip()
            basename = path.replace('\\', '/').rsplit('/', 1)[-1].lower()
            origin = {'candidate.cpp': 'design', 'testbench.cpp': 'selftest'}.get(basename, 'other')
            counts[origin] += 1
            if len(records) < max_records:
                records.append({'origin': origin, 'file': path[:1000],
                                'line': int(found['line']),
                                'column': int(found['column']) if found['column'] else None,
                                'message': found['message'][:1000]})
    total = sum(counts.values())
    return {'origins': [name for name, count in counts.items() if count],
            'counts': counts, 'total_occurrences': total, 'records': records,
            'records_truncated': total > len(records)}


def _match(line):
    if _source_quote(line):
        return 4, 0
    for priority, pattern in enumerate((INFRA_ERROR, SPECIFIC, GENERIC, VALUES)):
        found = pattern.search(line)
        if priority == 0 and not _infrastructure_line(line):
            continue
        if found:
            return priority, found.start()
    return 4, 0


def select_diagnostics(lines, *, max_lines=20, max_chars=6000, line_chars=1200):
    """Prioritize before truncation, retaining nearby compiler/counterexample context.

    Raw logs remain on disk. Ordering is a repair heuristic, not a parser or a
    claim that a test result is correct. All sizes count characters, not tokens.
    """
    rows = [line.rstrip() for value in lines for line in str(value).splitlines()]
    usable = [i for i, line in enumerate(rows)
              if line.strip() and not SCAFFOLD.search(line) and not NO_ERRORS.fullmatch(line)]
    # Infrastructure and specific errors first, then values before generic wrappers.
    anchors = sorted((i for i in usable if _match(rows[i])[0] < 4),
                     key=lambda i: ({0:0, 1:1, 3:2, 2:3}[_match(rows[i])[0]], i))
    # Keep the message with its immediate source/caret or input/value lines.
    order, queued = [], set()
    usable_set = set(usable)
    def enqueue(i):
        if i in usable_set and i not in queued:
            # Do not fill a useful excerpt with unrelated tool progress. Warnings
            # remain eligible: some carry real synthesis constraints.
            if anchors and _match(rows[i])[0] == 4 and re.match(r'^\s*INFO:', rows[i], re.I):
                return
            queued.add(i)
            order.append(i)
    for i in anchors:
        enqueue(i)
    # Reserve space for other errors and counterexample values before adding
    # long source excerpts. Otherwise one excerpt can evict every later error.
    for i in anchors:
        for neighbor in (i-1, i+1, i+2, i+3):
            if neighbor in usable_set and _match(rows[neighbor])[0] >= 3:
                enqueue(neighbor)
    for i in [] if anchors else usable[-max_lines:]:
        enqueue(i)
    result, seen, used = [], set(), 0
    for i in order:
        line = rows[i].strip()
        if line in seen:
            continue
        seen.add(line)
        # A very long compiler prefix must not hide the actual error marker.
        position = _match(line)[1]
        if len(line) > line_chars:
            start = max(0, position - min(160, max(0, (line_chars-60)//3)))
            line = ('[... truncated prefix ...] ' if start else '') + line[start:start+line_chars-60]
            line += ' [... truncated ...]'
        available = max_chars - used - bool(result)
        if available <= 0 or len(result) >= max_lines:
            break
        if len(line) > available:
            # Leave the original line in the evidence files; mark the excerpt.
            marker = ' [... truncated ...]'
            if available <= len(marker):
                break
            line = line[:available-len(marker)] + marker
        result.append(line)
        used += len(line) + (len(result) > 1)
    return result
