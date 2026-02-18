from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from axion.dataset import DatasetItem

from eval_workbench.shared.extractors import ExtractorHelpers
from eval_workbench.shared.langfuse.trace import Trace

_RE_BUSINESS_NAME = re.compile(r'Business Name:\s*(.+)', re.IGNORECASE)
_RE_ADDRESS = re.compile(r'Address:\s*(.+)', re.IGNORECASE)
_RE_REQUESTED_DATA = re.compile(r'REQUESTED DATA:\s*(\{.*\})\s*$', re.DOTALL)


class GroundingExtractor(ExtractorHelpers[Trace]):
    """OOP extractor for Magic Dust grounding traces."""

    def extract(self, source: Trace) -> DatasetItem:
        """
        Extract a DatasetItem from a Magic Dust grounding trace.

        Expected step name is usually ``search-with-grounding``.
        """
        trace = source
        step_name, step = self._resolve_grounding_step(trace)
        generation = self.select_step_generation(step)
        span_for_meta = self.select_step_span(step) or generation

        raw_input = self.safe_get(generation, 'input', '')
        prompt_text = _normalize_prompt_text(raw_input)
        business_name = (
            self.safe_get(step, 'variables.businessName', None)
            or self.safe_get(step, 'variables.business_name', None)
            or _extract_business_name(prompt_text)
        )
        address = (
            self.safe_get(step, 'variables.primaryLocation', None)
            or self.safe_get(step, 'variables.address', None)
            or _extract_address(prompt_text)
        )

        requested_data_raw = (
            self.safe_get(step, 'variables.requestedData', None)
            or self.safe_get(step, 'variables.requested_data', None)
            or self.safe_get(step, 'variables.schema', None)
            or _extract_requested_data_block(prompt_text)
        )
        requested_data = self.parse_json_like(requested_data_raw)

        raw_output = self.safe_get(generation, 'output', '')
        parsed_output = self.parse_json_like(raw_output)

        trace_id = str(self.safe_get(trace, 'id', ''))
        observation_id = self.safe_get(generation, 'id', '') or self.safe_get(
            span_for_meta, 'id', ''
        )
        latency = self.safe_get(span_for_meta, 'latency', None)

        trace_metadata = self.to_plain_dict(self.safe_get(trace, 'metadata', {}))
        if not isinstance(trace_metadata, dict):
            trace_metadata = {}

        dataset_id = _build_dataset_id(
            trace_id=trace_id,
            business_name=business_name,
            address=address,
        )

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

    def _resolve_grounding_step(self, trace: Trace) -> tuple[str, Any]:
        for step_name in (
            'search-with-grounding',
            'search_with_grounding',
            'grounding',
        ):
            step = self.safe_get(trace, step_name, None)
            if step is not None:
                return step_name, step
        raise ValueError(
            'Grounding step not found. Expected one of: '
            'search-with-grounding, search_with_grounding, grounding'
        )


_GROUNDING_EXTRACTOR = GroundingExtractor()


def extract_grounding(trace: Trace) -> DatasetItem:
    """Backward-compatible function wrapper for GroundingExtractor."""
    return _GROUNDING_EXTRACTOR.extract(trace)

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


def _stringify_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    if raw_output is None:
        return ''
    try:
        return json.dumps(_GROUNDING_EXTRACTOR.to_plain_dict(raw_output))
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
