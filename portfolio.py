"""Best-of-K candidate screening that runs *before* the graded attempt budget.

The graded loop spends its first design call on one sample. When that sample
picks a wrong approach, every later repair call inherits it: repairing a wrong
algorithm rarely turns it into a right one. This stage draws K independent
samples from the same baseline-aligned request with a temperature ladder,
screens each one with the host-side fast lane, and hands the strongest
candidate to ``run_policy`` as its first attempt.

The module is deliberately pure: it owns the sampling ladder and the ranking,
not the model call, the compiler, or the clock. A screening verdict is always
weaker than a graded verdict: the host lane runs the model's *own* testbench,
so "test_failed" is a hint, not proof. On a documented run the host lane
rejected a candidate that the graded Vitis verifier then passed, which is why a
self-test failure ranks *below* an unscreened sample here. Only a host compile
failure — a language-level fact that Vitis would also observe — sinks a
candidate to the bottom, and even then the candidate is kept, never deleted.
Everything here is opt-in (``portfolio_candidates`` defaults to 0) and any
caller-side failure falls back to the single-sample flow.
"""

TEMPERATURE_LADDER = (0.2, 0.6, 0.9, 1.1)
MAX_CANDIDATES = 4
# Ordering reflects how much each verdict can be trusted, not how bad it
# sounds. An unscreened sample stays above a self-test failure, because the
# testbench is model-written and has produced documented false negatives. But a
# candidate that ran and produced a diagnostic is stronger evidence than one
# that only compiled: v43i swept clz from a `test_failed` seed (rescued at round
# 6) while a `compile_passed` seed cost viterbi, a case the baseline solves. A
# csim verdict that never finished (`sim_timeout`) is weaker than an unscreened
# sample, because nothing about the design was observed.
STATUS_RANK = {'passed': 5, 'unchecked': 4, 'test_failed': 3.5,
               'compile_passed': 3, 'unsupported': 3, 'sim_timeout': 2.5,
               'compile_failed': 1, 'error': 0}


def temperature_ladder(count):
    """Return ``count`` sampling temperatures, low first, never empty when > 0."""
    if type(count) is not int or count < 0:
        raise ValueError('candidate count must be a non-negative integer')
    if count == 0:
        return ()
    if count <= len(TEMPERATURE_LADDER):
        return TEMPERATURE_LADDER[:count]
    # More candidates than ladder entries reuse the most diverse temperature.
    return TEMPERATURE_LADDER + (TEMPERATURE_LADDER[-1],) * (count - len(TEMPERATURE_LADDER))


def affordable_count(remaining_s, sample_s, reserve_s=150.0, limit=MAX_CANDIDATES):
    """How many independent samples fit while keeping the graded reserve.

    Slow inference hosts (the reference run measured ~27 tok/s, i.e. one design
    sample can cost 40-70 s) otherwise spend the whole budget on sampling and
    leave nothing for verification. Returns 0 when even one sample would eat the
    reserve, so the caller falls back to the single-sample flow.
    """
    try:
        remaining = float(remaining_s)
        per_sample = float(sample_s)
        reserved = float(reserve_s)
    except (TypeError, ValueError):
        return 0
    if per_sample <= 0 or not per_sample == per_sample or remaining != remaining:
        return 0
    room = remaining - reserved
    if room <= 0:
        return 0
    return max(0, min(int(limit), int(room // per_sample)))


def candidate_count(value, default=0):
    """Resolve the portfolio size from configuration or environment text."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('portfolio_candidates must be an integer')
    if not 0 <= value <= MAX_CANDIDATES:
        raise ValueError('portfolio_candidates must be between 0 and ' + str(MAX_CANDIDATES))
    return value


def rank(screening):
    """Order screenings by trust: pass > unscreened > self-test fail > compile fail.

    A self-test failure is a hint from a model-written testbench and therefore
    ranks below "never screened"; a compile failure is a language-level fact and
    ranks last. Ties prefer the candidate with fewer diagnostics (fewer errors is
    evidence of a closer design), then the earlier sample, so ranking is
    deterministic.
    """
    if not isinstance(screening, dict):
        raise ValueError('screening must be a mapping')
    diagnostics = screening.get('diagnostics') or []
    return (STATUS_RANK.get(screening.get('status'), 0), -len(diagnostics),
            -int(screening.get('index', 0) or 0))


def choice(screenings):
    """Return the strongest screening, or ``None`` when nothing was produced."""
    entries = [item for item in (screenings or ()) if isinstance(item, dict)]
    if not entries:
        return None
    return max(entries, key=rank)


def summary(screenings):
    """Compact, evidence-bearing record for the run result and the trace."""
    return [{'index': item.get('index'), 'temperature': item.get('temperature'),
             'status': item.get('status'), 'source': item.get('source'),
             'diagnostics': list(item.get('diagnostics') or [])}
            for item in (screenings or ()) if isinstance(item, dict)]
