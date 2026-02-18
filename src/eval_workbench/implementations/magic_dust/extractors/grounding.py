from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from typing import Any

from axion.dataset import DatasetItem
from json_repair import repair_json

from eval_workbench.shared.langfuse.trace import Trace

logger = logging.getLogger(__name__)

_RE_BUSINESS_NAME = re.compile(r'Business Name:\s*(.+)', re.IGNORECASE)
_RE_ADDRESS = re.compile(r'Address:\s*(.+)', re.IGNORECASE)
_RE_REQUESTED_DATA = re.compile(r'REQUESTED DATA:\s*(\{.*\})\s*$', re.DOTALL)


def extract_grounding(trace: Trace) -> DatasetItem:
    """
    Extract a DatasetItem from a Magic Dust grounding trace.

    Expected step name is usually ``search-with-grounding``.
    """
    step_name, step = _resolve_grounding_step(trace)
    generation = _select_step_generation(step)
    span_for_meta = _select_step_span(step) or generation

    raw_input = _safe_get(generation, 'input', '')
    prompt_text = _normalize_prompt_text(raw_input)
    business_name = (
        _safe_get(step, 'variables.businessName', None)
        or _safe_get(step, 'variables.business_name', None)
        or _extract_business_name(prompt_text)
    )
    address = (
        _safe_get(step, 'variables.primaryLocation', None)
        or _safe_get(step, 'variables.address', None)
        or _extract_address(prompt_text)
    )

    requested_data_raw = (
        _safe_get(step, 'variables.requestedData', None)
        or _safe_get(step, 'variables.requested_data', None)
        or _safe_get(step, 'variables.schema', None)
        or _extract_requested_data_block(prompt_text)
    )
    requested_data = _parse_json_like(requested_data_raw)

    raw_output = _safe_get(generation, 'output', '')
    parsed_output = _parse_json_like(raw_output)

    trace_id = str(_safe_get(trace, 'id', ''))
    observation_id = _safe_get(generation, 'id', '') or _safe_get(span_for_meta, 'id', '')
    latency = _safe_get(span_for_meta, 'latency', None)

    trace_metadata = _to_plain_dict(_safe_get(trace, 'metadata', {}))
    if not isinstance(trace_metadata, dict):
        trace_metadata = {}

    dataset_id = _build_dataset_id(trace_id=trace_id, business_name=business_name, address=address)

    return DatasetItem(
        id=dataset_id,
        query=_build_query(business_name=business_name, address=address),
        expected_output=None,
        acceptance_criteria=None,
        additional_input={
            'business_name': business_name,
            'address': address,
            'requested_data': requested_data,
            'prompt_step': step_name,
        },
        dataset_metadata=json.dumps(trace_metadata),
        actual_output=_stringify_output(raw_output),
        latency=latency,
        trace_id=trace_id,
        observation_id=observation_id,
        additional_output={
            'parsed_output': parsed_output,
        },
    )

def _resolve_grounding_step(trace: Trace) -> tuple[str, Any]:
    for step_name in (
        'search-with-grounding',
        'search_with_grounding',
        'grounding',
    ):
        step = _safe_get(trace, step_name, None)
        if step is not None:
            return step_name, step
    raise ValueError(
        'Grounding step not found. Expected one of: '
        'search-with-grounding, search_with_grounding, grounding'
    )


def _select_step_generation(step: Any) -> Any:
    if step is None:
        return None
    try:
        for obs in list(getattr(step, 'observations', [])):
            if getattr(obs, 'type', '').upper() == 'GENERATION':
                return obs
    except Exception:
        pass
    return _safe_get(step, 'GENERATION', None) or _safe_get(step, 'generation', None)


def _select_step_span(step: Any) -> Any:
    if step is None:
        return None
    try:
        for obs in list(getattr(step, 'observations', [])):
            if getattr(obs, 'type', '').upper() == 'SPAN':
                return obs
    except Exception:
        return None
    return None


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    if obj is None:
        return default
    current = obj
    for part in path.split('.'):
        try:
            if isinstance(current, dict):
                current = current.get(part, default)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                current = current[part]
        except Exception:
            return default
        if current is None:
            return default
    return current


def _normalize_prompt_text(raw_text: Any) -> str:
    if isinstance(raw_text, str):
        return raw_text
    if isinstance(raw_text, dict):
        for key in ('content', 'prompt', 'text'):
            value = raw_text.get(key)
            if isinstance(value, str):
                return value
        return ''
    if isinstance(raw_text, list):
        parts: list[str] = []
        for item in raw_text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get('content')
                if isinstance(value, str):
                    parts.append(value)
        return '\n'.join(parts)
    return ''


def _extract_business_name(prompt_text: str) -> str | None:
    m = _RE_BUSINESS_NAME.search(prompt_text or '')
    return m.group(1).strip() if m else None


def _extract_address(prompt_text: str) -> str | None:
    m = _RE_ADDRESS.search(prompt_text or '')
    return m.group(1).strip() if m else None


def _extract_requested_data_block(prompt_text: str) -> str | None:
    m = _RE_REQUESTED_DATA.search(prompt_text or '')
    return m.group(1).strip() if m else None


def _parse_json_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return _to_plain_dict(value)

    blob = value.strip()
    if not blob:
        return None

    try:
        return json.loads(blob)
    except Exception:
        pass

    try:
        repaired = repair_json(blob)
        return json.loads(repaired)
    except Exception:
        pass

    try:
        return ast.literal_eval(blob)
    except Exception:
        logger.debug('Failed to parse json-like blob for grounding output.')
        return None


def _to_plain_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    if hasattr(value, 'to_dict'):
        try:
            return _to_plain_dict(value.to_dict())
        except Exception:
            pass
    if hasattr(value, 'model_dump'):
        try:
            return _to_plain_dict(value.model_dump())
        except Exception:
            pass
    if hasattr(value, '__dict__'):
        try:
            return _to_plain_dict(vars(value))
        except Exception:
            pass
    return str(value)


def _stringify_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    if raw_output is None:
        return ''
    try:
        return json.dumps(_to_plain_dict(raw_output))
    except Exception:
        return str(raw_output)


def _build_dataset_id(
    *, trace_id: str | None, business_name: str | None, address: str | None
) -> str:
    if trace_id:
        return trace_id

    seed = f'{business_name or ""}|{address or ""}'
    if not seed.strip('|'):
        return 'grounding-unknown'
    digest = hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]
    return f'grounding-{digest}'


def _build_query(*, business_name: str | None, address: str | None) -> str:
    return json.dumps(
        {
            'businessName': business_name,
            'primaryLocation': address,
        },
        separators=(',', ':'),
    )
