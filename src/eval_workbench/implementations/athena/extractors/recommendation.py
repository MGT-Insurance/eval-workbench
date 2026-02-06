import ast
import json
import logging
import re
from typing import Any

from axion.dataset import DatasetItem

from eval_workbench.shared.langfuse.trace import Trace

logger = logging.getLogger(__name__)

# Regex to extract recommendation label
RE_LABEL = re.compile(
    r'Recommend\s+('
    r'Approve'
    r'|Decline'
    r'|Refer(?:ral)?(?:\s+to\s+(?:Underwriter|UW))?'
    r')',
    re.IGNORECASE,
)


def extract_recommendation_label(s: str) -> str:
    """Extract recommendation label from brief recommendation text."""
    m = RE_LABEL.search(s)
    if m:
        label = m.group(1).strip().upper()
        if label.startswith('REFER'):
            return 'REFER'
        return label
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
    swallow_debug_data = _safe_get(trace, 'recommendation.variables.swallow_debug_data', '')
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
            'swallow_debug_data': swallow_debug_data,
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


def extract_recommendation_from_row(row: dict[str, Any]) -> DatasetItem:
    """Extract a DatasetItem from an athena_cases database row."""
    quote_locator = row.get('quote_locator', 'unknown')

    # Parse recommendation_entries (JSONB returns list, may be str)
    recommendation_entries = row.get('recommendation_entries', [])
    if isinstance(recommendation_entries, str):
        try:
            recommendation_entries = json.loads(recommendation_entries)
        except json.JSONDecodeError:
            recommendation_entries = []

    # Get most recent entry
    latest_entry = {}
    if recommendation_entries:
        sorted_entries = sorted(
            recommendation_entries,
            key=lambda x: x.get('executedAt', ''),
            reverse=True,
        )
        latest_entry = sorted_entries[0] if sorted_entries else {}

    brief_recommendation = latest_entry.get('briefRecommendation', '')
    detailed_recommendation = latest_entry.get('detailedRecommendation', '')
    underwriting_flags = latest_entry.get('underwritingFlags', [])
    label = extract_recommendation_label(brief_recommendation)

    latency = latest_entry.get('executionTimeMs')
    if latency:
        latency = latency / 1000.0  # ms to seconds

    entry_metadata = {
        'model': latest_entry.get('model'),
        'claude_cost': latest_entry.get('claudeCost'),
        'claude_tokens': latest_entry.get('claudeTokens'),
        'prompt_version': latest_entry.get('promptVersion'),
        'executed_at': latest_entry.get('executedAt'),
    }

    trace_id = row.get('langfuse_trace_id', '')

    return DatasetItem(
        id=quote_locator,
        query=f'Provide a risk assessment for {quote_locator}',
        expected_output=None,
        acceptance_criteria=None,
        additional_input={
            'underwriting_flags': underwriting_flags,
            'all_entries': recommendation_entries,
        },
        dataset_metadata=json.dumps(entry_metadata),
        actual_output=label,
        latency=latency,
        trace_id=trace_id,
        additional_output={
            'brief_recommendation': brief_recommendation,
            'detailed_recommendation': detailed_recommendation,
        },
    )
