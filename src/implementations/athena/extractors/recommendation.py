import ast
import json
import logging
import re
from typing import Any

from axion.dataset import DatasetItem

from shared.langfuse.trace import Trace

logger = logging.getLogger(__name__)

# Regex to extract recommendation label
RE_LABEL = re.compile(
    r'Recommend\s+(Approve|Decline|Refer to Underwriter)',
    re.IGNORECASE,
)


def extract_recommendation_label(s: str) -> str:
    """Extract recommendation label from brief recommendation text."""
    m = RE_LABEL.search(s)
    if m:
        return m.group(1).upper()
    return 'UNKNOWN'


def extract_recommendation(trace: Trace) -> DatasetItem:
    """
    Extract a DatasetItem from an Athena recommendation trace.

    Maps trace data to axion's DatasetItem format for evaluation.

    Args:
        trace: Trace object from TraceCollection

    Returns:
        DatasetItem ready for evaluation_runner
    """
    # Input fields
    quote_locator = _safe_get(
        trace, 'recommendation.span.input.quote_locator', 'unknown'
    )

    # Parse underwriting flags (stored as string repr of list)
    underwriting_flags_raw = _safe_get(
        trace, 'recommendation.variables.underwriting_flags', '[]'
    )
    try:
        underwriting_flags = (
            ast.literal_eval(underwriting_flags_raw)
            if isinstance(underwriting_flags_raw, str)
            else underwriting_flags_raw
        )
    except (ValueError, SyntaxError):
        underwriting_flags = []

    # Parse context data (stored as JSON string)
    context_data_raw = _safe_get(trace, 'recommendation.variables.context_data', '{}')
    try:
        context_data = (
            json.loads(context_data_raw)
            if isinstance(context_data_raw, str)
            else context_data_raw
        )
    except json.JSONDecodeError:
        context_data = {}

    case_assessment = _safe_get(trace, 'recommendation.variables.case_assessment', '')

    # Performance
    latency = _safe_get(trace, 'recommendation.span.latency')

    # Output fields
    brief_recommendation = _safe_get(
        trace, 'recommendation.span.output.brief_recommendation', ''
    )
    detailed_recommendation = _safe_get(
        trace, 'recommendation.span.output.detailed_recommendation', ''
    )
    label = extract_recommendation_label(brief_recommendation)

    # Citations
    citations_raw = _safe_get(trace, 'recommendation.span.output.citations', [])
    citations = []
    if citations_raw:
        for c in citations_raw:
            if hasattr(c, 'to_dict'):
                citations.append(c.to_dict())
            elif isinstance(c, dict):
                citations.append(c)

    # Langfuse metadata
    trace_id = str(getattr(trace, 'id', ''))
    observation_id = _safe_get(trace, 'recommendation.span.id', '')

    # Trace metadata
    trace_metadata = {}
    if hasattr(trace, 'metadata'):
        meta = trace.metadata
        if hasattr(meta, 'to_dict'):
            trace_metadata = meta.to_dict()
        elif isinstance(meta, dict):
            trace_metadata = meta

    return DatasetItem(
        id=quote_locator,
        query=f'Provide a risk assessment for {quote_locator}',
        expected_output=None,  # Set from golden dataset if available
        acceptance_criteria=None,  # Set from golden dataset if available
        additional_input={
            'underwriting_flags': underwriting_flags,
            'context_data': context_data,
            'case_assessment': case_assessment,
        },
        dataset_metadata=json.dumps(trace_metadata),
        actual_output=label,
        latency=latency,
        trace_id=trace_id,
        actual_reference=citations,
        observation_id=observation_id,
        additional_output={
            'brief_recommendation': brief_recommendation,
            'detailed_recommendation': detailed_recommendation,
        },
    )


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safely get nested attribute using dot notation."""
    parts = path.split('.')
    current = obj

    for part in parts:
        try:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        except Exception:
            return default

    return current if current is not None else default
