"""Shared utilities for LLM interactions."""

import re
import json
import builtins
from typing import Any, Dict, Tuple

def clean_str(response: str) -> str:
    """Clean response strings to make them JSON-safe."""
    title_pattern = r'"Optimized Title":\s*"([^\"]*?)"'
    title_match = re.search(title_pattern, response)
    title_value = title_match.group(1) if title_match else None

    justification_pattern = r'"Justification":\s*"(.*)"'
    justification_match = re.search(justification_pattern, response, re.DOTALL)
    justification_value = justification_match.group(1) if justification_match else None

    if title_value:
        updated_title_value = title_value.replace('"', "'").replace("\\", "")
        response = response.replace(f'"{title_value}"', f'"{updated_title_value}"')

    if justification_value:
        updated_justification_value = justification_value.replace('"', "'").replace("\\", "")
        response = response.replace(f'"{justification_value}"', f'"{updated_justification_value}"')

    return response


def filter_response(response: str) -> str:
    """Return only the last JSON object-looking substring."""
    substring = '{'
    start = response.rfind(substring)
    if start > -1:
        return response[start:]
    return response


def is_valid_response(original_dict: Any, required_keys: set) -> Tuple[Dict, bool]:
    """Validate response is a dict and contains required keys with non-empty values."""
    if not isinstance(original_dict, builtins.dict):
        return {}, False

    cleaned_dict = {key: original_dict[key] for key in required_keys if key in original_dict}

    ok = all(
        key in cleaned_dict
        and cleaned_dict[key] is not None
        and (cleaned_dict[key] or isinstance(cleaned_dict[key], (int, float)))
        for key in required_keys
    )

    return cleaned_dict, ok


def clean_prompt(text: str) -> str:
    """Clean whitespace and newlines from prompts."""
    return re.sub(r'\s+', ' ', text.strip())
    and removed backslashes.
    """
    # --- Extract Optimized Title value
    title_pattern = r'"Optimized Title":\s*"([^\"]*?)"'
    title_match = re.search(title_pattern, response)
    title_value = title_match.group(1) if title_match else None

    # --- Extract Justification value (greedy match to last closing quote)
    justification_pattern = r'"Justification":\s*"(.*)"'
    justification_match = re.search(justification_pattern, response, re.DOTALL)
    justification_value = justification_match.group(1) if justification_match else None

    if title_value:
        updated_title_value = title_value.replace('"', "'").replace("\\", "")
        response = response.replace(f'"{title_value}"', f'"{updated_title_value}"')

    if justification_value:
        updated_justification_value = justification_value.replace('"', "'").replace("\\", "")
        response = response.replace(f'"{justification_value}"', f'"{updated_justification_value}"')

    return response


def filter_response(response: str) -> str:
    """Return only the last JSON object-looking substring (heuristic from notebook)."""
    substring = '{'
    start = response.rfind(substring)
    if start > -1:
        return response[start:]
    return response


def is_valid_response(original_dict: Any, required_keys: set):
    """Validate response is a dict and contains required keys with non-empty values.

    Returns (cleaned_dict, is_valid_bool)
    """
    if not isinstance(original_dict, builtins.dict):
        return {}, False

    cleaned_dict = {key: original_dict[key] for key in required_keys if key in original_dict}

    ok = all(
        key in cleaned_dict
        and cleaned_dict[key] is not None
        and (cleaned_dict[key] or isinstance(cleaned_dict[key], (int, float)))
        for key in required_keys
    )

    return cleaned_dict, ok


def clean_prompt(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

