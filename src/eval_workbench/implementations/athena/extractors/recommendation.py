import ast
import json
import logging
import math
import re
from typing import Any

from axion.dataset import DatasetItem

from eval_workbench.shared.extractors import BaseExtractor, ExtractorMixin
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


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safely get nested values with fuzzy dict key matching enabled."""
    return _RECOMMENDATION_EXTRACTOR.safe_get(
        obj,
        path,
        default,
        fuzzy_dict_match=True,
    )


def _extract_location_extraction_first_location(trace: Trace) -> dict[str, Any] | None:
    """
    Best-effort extraction of:
      trace['location-extraction'].GENERATION.output.locations[0].to_dict()

    Stored into DatasetItem.dataset_metadata as `location_extraction`.
    """
    try:
        step = trace['location-extraction']
    except Exception:
        try:
            step = trace['location_extraction']
        except Exception:
            return None

    # Prefer the explicit GENERATION observation, but allow the alias too.
    gen = _safe_get(step, 'GENERATION', None) or _safe_get(step, 'generation', None)
    if gen is None:
        return None

    locations = _safe_get(gen, 'output.locations', None)
    if not isinstance(locations, list) or not locations:
        return None

    first = locations[0]
    out = _RECOMMENDATION_EXTRACTOR.to_plain_dict(first)
    return out if isinstance(out, dict) and out else None


def _is_invalid_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {'undefined', 'null', 'nan', ''}
    return False


def _select_recommendation_observation(trace: Trace) -> Any:
    observations = _get_recommendation_observations(trace)
    for obs in observations:
        brief, detailed, _ = _get_recommendation_output(obs)
        if not (_is_invalid_value(brief) and _is_invalid_value(detailed)):
            return obs

    return observations[0] if observations else None


def _get_recommendation_observations(trace: Trace) -> list[Any]:
    try:
        step = trace.recommendation
        return list(getattr(step, 'observations', []))
    except Exception:
        return []


def _get_first_recommendation_span(trace: Trace) -> Any:
    for obs in _get_recommendation_observations(trace):
        if getattr(obs, 'type', '').upper() == 'SPAN':
            return obs
    return None


def _get_recommendation_output(observation: Any) -> tuple[str, str, list[Any]]:
    if observation is None:
        return '', '', []
    brief = _safe_get(observation, 'output.brief_recommendation', '')
    detailed = _safe_get(observation, 'output.detailed_recommendation', '')
    citations_raw = _safe_get(observation, 'output.citations', [])
    return brief, detailed, citations_raw


class RecommendationExtractor(ExtractorMixin[Trace]):
    """OOP extractor for Athena recommendation traces."""

    def extract(self, source: Trace) -> DatasetItem:
        trace = source
        selected_observation = _select_recommendation_observation(trace)
        span_for_meta = _get_first_recommendation_span(trace) or selected_observation

        # Input fields
        quote_locator = self.safe_get(
            span_for_meta, 'input.quote_locator', 'unknown', fuzzy_dict_match=True
        )
        if quote_locator == 'unknown':
            quote_locator = self.safe_get(
                trace,
                'recommendation.span.input.quote_locator',
                'unknown',
                fuzzy_dict_match=True,
            )

        # Parse underwriting flags (stored as string repr of list)
        underwriting_flags_raw = self.safe_get(
            trace,
            'recommendation.variables.underwriting_flags',
            '[]',
            fuzzy_dict_match=True,
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
        context_data_raw = self.safe_get(
            trace, 'recommendation.variables.context_data', '{}', fuzzy_dict_match=True
        )
        try:
            context_data = (
                json.loads(context_data_raw)
                if isinstance(context_data_raw, str)
                else context_data_raw
            )
        except json.JSONDecodeError:
            context_data = {}

        case_assessment = self.safe_get(
            trace, 'recommendation.variables.case_assessment', '', fuzzy_dict_match=True
        )
        swallow_debug_data = self.safe_get(
            trace,
            'recommendation.variables.swallow_debug_data',
            '',
            fuzzy_dict_match=True,
        )

        latency = self.safe_get(
            span_for_meta, 'latency', fuzzy_dict_match=True
        )
        if latency is None:
            latency = self.safe_get(
                trace, 'recommendation.span.latency', fuzzy_dict_match=True
            )

        # Output fields
        brief_recommendation, detailed_recommendation, citations_raw = (
            _get_recommendation_output(selected_observation)
        )
        label = extract_recommendation_label(brief_recommendation)

        citations = []
        if citations_raw:
            for c in citations_raw:
                if hasattr(c, 'to_dict'):
                    citations.append(c.to_dict())
                elif isinstance(c, dict):
                    citations.append(c)

        trace_id = str(getattr(trace, 'id', ''))
        observation_id = self.safe_get(
            selected_observation, 'id', '', fuzzy_dict_match=True
        )
        if not observation_id:
            observation_id = self.safe_get(
                span_for_meta, 'id', '', fuzzy_dict_match=True
            )
        if not observation_id:
            observation_id = self.safe_get(
                trace, 'recommendation.span.id', '', fuzzy_dict_match=True
            )

        trace_metadata = {}
        if hasattr(trace, 'metadata'):
            meta = trace.metadata
            if hasattr(meta, 'to_dict'):
                trace_metadata = meta.to_dict()
            elif isinstance(meta, dict):
                trace_metadata = meta

        location_extraction = _extract_location_extraction_first_location(trace)
        if location_extraction is not None:
            trace_metadata['location_extraction'] = location_extraction

        return DatasetItem(
            id=quote_locator,
            query=f'Provide a risk assessment for {quote_locator}',
            expected_output=None,
            acceptance_criteria=None,
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


class RecommendationRowExtractor(BaseExtractor[dict[str, Any]]):
    """OOP extractor for Athena recommendation rows from Neon."""

    def extract(self, source: dict[str, Any]) -> DatasetItem:
        row = source
        quote_locator = row.get('quote_locator', 'unknown')

        recommendation_entries = row.get('recommendation_entries', [])
        if isinstance(recommendation_entries, str):
            try:
                recommendation_entries = json.loads(recommendation_entries)
            except json.JSONDecodeError:
                recommendation_entries = []

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
            latency = latency / 1000.0

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


_RECOMMENDATION_EXTRACTOR = RecommendationExtractor()
_RECOMMENDATION_ROW_EXTRACTOR = RecommendationRowExtractor()


def extract_recommendation(trace: Trace) -> DatasetItem:
    """Backward-compatible function wrapper for RecommendationExtractor."""
    return _RECOMMENDATION_EXTRACTOR.extract(trace)


def extract_recommendation_from_row(row: dict[str, Any]) -> DatasetItem:
    """Backward-compatible function wrapper for RecommendationRowExtractor."""
    return _RECOMMENDATION_ROW_EXTRACTOR.extract(row)
