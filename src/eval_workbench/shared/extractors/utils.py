from __future__ import annotations

import ast
import json
from typing import Any


def fuzzy_dict_get(data: dict[str, Any], key: str) -> Any:
    """Case/underscore-insensitive dict lookup."""
    target = key.lower().replace('_', '')
    for candidate_key, value in data.items():
        if candidate_key.lower().replace('_', '') == target:
            return value
    return None


def safe_get(
    obj: Any,
    path: str,
    default: Any = None,
    *,
    fuzzy_dict_match: bool = False,
) -> Any:
    """Safely read nested values from dict/object/item trees via dot path."""
    if obj is None:
        return default

    current = obj
    for part in path.split('.'):
        try:
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                elif fuzzy_dict_match:
                    matched = fuzzy_dict_get(current, part)
                    if matched is None:
                        return default
                    current = matched
                else:
                    return default
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                current = current[part]
        except Exception:
            return default

        if current is None:
            return default

    return current


def select_step_generation(step: Any) -> Any:
    """Return the first GENERATION observation from a step."""
    if step is None:
        return None
    try:
        for obs in list(getattr(step, 'observations', [])):
            if getattr(obs, 'type', '').upper() == 'GENERATION':
                return obs
    except Exception:
        return None
    return safe_get(step, 'GENERATION', None) or safe_get(step, 'generation', None)


def select_step_span(step: Any) -> Any:
    """Return the first SPAN observation from a step."""
    if step is None:
        return None
    try:
        for obs in list(getattr(step, 'observations', [])):
            if getattr(obs, 'type', '').upper() == 'SPAN':
                return obs
    except Exception:
        return None
    return None


def to_plain_dict(value: Any) -> Any:
    """Convert nested model-ish objects into plain Python structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    if hasattr(value, 'to_dict'):
        try:
            return to_plain_dict(value.to_dict())
        except Exception:
            pass
    if hasattr(value, 'model_dump'):
        try:
            return to_plain_dict(value.model_dump())
        except Exception:
            pass
    if hasattr(value, '__dict__'):
        try:
            return to_plain_dict(vars(value))
        except Exception:
            pass
    return str(value)


def parse_json_like(value: Any, *, use_json_repair: bool = True) -> Any:
    """Best-effort parser for JSON-like text or model values."""
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return to_plain_dict(value)

    blob = value.strip()
    if not blob:
        return None

    try:
        return json.loads(blob)
    except Exception:
        pass

    if use_json_repair:
        try:
            from json_repair import repair_json

            repaired = repair_json(blob)
            return json.loads(repaired)
        except Exception:
            pass

    try:
        return ast.literal_eval(blob)
    except Exception:
        return None
