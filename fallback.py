"""Align an unverified first request with the unchanged HLS baseline prompt.

This aligns model input and sampling, not outcomes, output extraction or the
remaining agent deadline. User-configured smaller output limits still apply.
"""
from baseline import MAX_TOKENS, SYS, TEMPERATURE, TOP_P
from agent.model import validate_config


def baseline_messages(description):
    """Use the official system prompt and complete public task without skills."""
    return [{'role':'system', 'content':SYS['hls']},
            {'role':'user', 'content':description}]


def fallback_model_config(config):
    """Clone a valid model configuration; retain stricter configured budgets."""
    value = validate_config(config)
    value.update(temperature=TEMPERATURE, top_p=TOP_P,
                 max_tokens=min(value['max_tokens'], MAX_TOKENS))
    value.pop('seed', None)  # The official request does not supply a seed.
    return value
