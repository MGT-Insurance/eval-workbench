import json
import logging
from typing import Any

from axion.dataset import DatasetItem

from eval_workbench.shared.langfuse.trace import Trace

logger = logging.getLogger(__name__)


def extract_location_extraction(trace: Trace) -> DatasetItem:
    """
    Extract a DatasetItem from an Athena "location-extraction" trace step.

    Adds `product_initiate` (from the rendered Quote JSON) into DatasetItem.dataset_metadata.
    """
    step = _safe_get(trace, 'location-extraction', None)
    if step is None:
        # Fallback to underscore variant (some callers may use it)
        step = _safe_get(trace, 'location_extraction', None)

    quote_raw = _safe_get(step, 'variables.quote', None) if step else None
    quote: dict[str, Any] = {}
    if isinstance(quote_raw, str) and quote_raw.strip():
        try:
            quote = json.loads(quote_raw)
        except json.JSONDecodeError:
            logger.warning('Failed to JSON-decode location-extraction variables.quote')
    elif isinstance(quote_raw, dict):
        quote = quote_raw

    quote_locator = (
        _safe_get(quote, 'locator', None)
        or _safe_get(step, 'span.input.quote_locator', None)
        or _safe_get(trace, 'id', 'unknown')
    )

    product_initiate = _safe_get(quote, 'element.data.product_initiate', None)

    # Trace metadata (existing) + augmentation
    trace_metadata: dict[str, Any] = {}
    meta = _safe_get(trace, 'metadata', None)
    if hasattr(meta, 'to_dict'):
        trace_metadata = meta.to_dict()
    elif isinstance(meta, dict):
        trace_metadata = meta

    if product_initiate is not None:
        trace_metadata['product_initiate'] = product_initiate

    selected_observation = _select_step_generation(step) if step else None
    span_for_meta = _select_step_span(step) if step else None

    latency = _safe_get(span_for_meta, 'latency', None)
    if latency is None:
        latency = _safe_get(selected_observation, 'latency', None)

    trace_id = str(_safe_get(trace, 'id', ''))
    observation_id = (
        _safe_get(selected_observation, 'id', '')
        or _safe_get(span_for_meta, 'id', '')
        or ''
    )

    # Best-effort actual output capture (keep raw for downstream parsing)
    actual_output = _safe_get(selected_observation, 'output', None)
    if isinstance(actual_output, (dict, list)):
        actual_output = json.dumps(actual_output)
    if actual_output is None:
        actual_output = ''

    return DatasetItem(
        id=str(quote_locator),
        query=f'Extract and classify all addresses for {quote_locator}',
        expected_output=None,
        acceptance_criteria=None,
        additional_input={
            # Keep this small; full quote is already recoverable via trace prompt variables.
            'quote_locator': quote_locator,
        },
        dataset_metadata=json.dumps(trace_metadata),
        actual_output=actual_output,
        latency=latency,
        trace_id=trace_id,
        observation_id=observation_id,
        additional_output={
            'product_initiate': product_initiate,
        },
    )


def _select_step_generation(step: Any) -> Any:
    try:
        for obs in list(getattr(step, 'observations', [])):
            if getattr(obs, 'type', '').upper() == 'GENERATION':
                return obs
    except Exception:
        return None
    return None


def _select_step_span(step: Any) -> Any:
    try:
        for obs in list(getattr(step, 'observations', [])):
            if getattr(obs, 'type', '').upper() == 'SPAN':
                return obs
    except Exception:
        return None
    return None


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safely get nested attribute/item using dot notation."""
    if obj is None:
        return default
    parts = path.split('.')
    current: Any = obj
    for part in parts:
        try:
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return default
            else:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    # allow dict-style access for SmartAccess wrappers
                    try:
                        current = current[part]
                    except Exception:
                        return default
        except Exception:
            return default
    return current if current is not None else default

