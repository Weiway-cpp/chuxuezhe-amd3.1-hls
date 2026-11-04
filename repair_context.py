"""Small excerpts of local feedback; never an oracle or a verdict parser."""
import re

from scripts.diagnostics import select_diagnostics


_CASE = re.compile(r'^\s*(?:FAIL\s+(?:case|trial|index)\s*[:=]|Test\s+(?:case\s+)?\d+\s+(?:failed|mismatch)|.*\bmismatch\b)', re.I)
_VALUE = re.compile(r'\b(?:[A-Za-z_]*input|expected|actual|output|index|a|b|x|y)\s*[:=]', re.I)
_ERROR = re.compile(r'\b(?:fatal\s+)?error\s*:|undefined reference|segmentation (?:fault|violation)|access violation|bus error|floating point exception', re.I)
_INTEGER = re.compile(r'(?P<label>\b(?:input|expected|actual|a|b|x|y)\s*=\s*)'
                      r'(?P<value>[0-9]{10,20})(?![\w.])', re.I)


def _integer_bits(line):
    """Convert only explicitly decimal records; digit-only hex is ambiguous."""
    if not re.search(r'\b(?:radix|base)\s*=\s*10\b', line, re.I):
        return line
    def annotate(match):
        value = int(match['value'])
        if not (2**32 <= value < 2**64):
            return match[0]
        annotation = ' [hex=0x' + format(value, '016x') + ']'
        return match[0] if line[match.end():].startswith(annotation) else match[0] + annotation
    return _INTEGER.sub(annotate, line)


def repair_diagnostics(lines, status, *, max_chars=6000, line_chars=1200):
    """Keep up to three distinct counterexamples and adjacent value lines.

    Only an explicit case/mismatch record permits dropping tool scaffolding.
    Compiler/crash diagnostics and unfamiliar assertion formats fall back to
    the existing bounded selector. Nothing here changes test outcomes.
    """
    rows = [line.strip() for value in lines
            for line in str(value).splitlines() if line.strip()]
    if status not in ('test_failed', 'csim_failed') or any(_ERROR.search(s) for s in rows):
        return select_diagnostics(rows, max_chars=max_chars, line_chars=line_chars)
    anchors = [i for i, row in enumerate(rows) if _CASE.search(row)]
    if not anchors:
        return select_diagnostics(rows, max_chars=max_chars, line_chars=line_chars)
    if any(b == a+1 and not _VALUE.search(rows[a]) for a, b in zip(anchors, anchors[1:])):
        # The generic selector may have placed all bare case headers first.
        # Their original adjacency cannot be reconstructed from this excerpt.
        # Do not invent associations by attaching the next case's value lines.
        note = '[Case/value ordering unavailable in this excerpt; do not associate unlabelled values with a case.]'
        rest = [row for row in rows if row != note]
        room = max_chars-len(note)-1
        if room <= 0:
            return [note[:max_chars]] if max_chars > 0 else []
        return [note, *select_diagnostics(rest, max_chars=room, line_chars=line_chars)]
    records = []
    for i in anchors:
        record = [rows[i]]
        for row in rows[i+1:i+4]:
            if _CASE.search(row) or not _VALUE.search(row):
                break
            record.append(row)
        # Repeated excerpts may already have exact integer annotations.
        text = '\n'.join(record)
        text = _integer_bits(text)
        if text not in records:
            records.append(text)
    selected = [text if len(text) <= 700 else text[:670] + ' [... truncated ...]'
                for text in records[:3]]
    if len(records) > 3 or any('Additional local counterexamples omitted' in row for row in rows):
        selected.append('[Additional local counterexamples omitted; full diagnostics remain in the run evidence.]')
    # Keep each record together when a prompt budget forces further reduction.
    # Generic line-priority sorting would separate FAIL headers from their values.
    bounded, used = [], 0
    for text in selected:
        room = min(2400, max_chars) - used - bool(bounded)
        if room <= 0:
            break
        if len(text) > room:
            marker = ' [... truncated ...]'
            if room > len(marker):
                bounded.append(text[:room-len(marker)] + marker)
            break
        bounded.append(text)
        used += len(text) + (len(bounded) > 1)
    return bounded
