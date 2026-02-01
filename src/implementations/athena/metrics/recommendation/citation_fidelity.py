import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


class CitationVerdict(RichBaseModel):
    citation_text: str = Field(..., description='The citation found in the text.')
    path_referenced: str = Field(..., description='The JSON key/path.')
    json_value: str = Field(..., description='The value found in the JSON.')
    is_valid_path: bool = Field(..., description='True if path exists.')
    is_supported: bool = Field(
        ..., description='True if value matches text (or check skipped).'
    )
    reason: str = Field(..., description='Verdict explanation.')


class CitationFidelityResult(RichBaseModel):
    score: float = Field(...)
    total_citations: int = Field(...)
    valid_citations: int = Field(...)
    verdicts: List[CitationVerdict] = Field(...)


@metric(
    name='Citation Fidelity',
    key='citation_fidelity',
    description='Verifies that bracketed citations point to real JSON keys. Optionally checks if the value appears in the text.',
    required_fields=['actual_output', 'expected_output'],
    default_threshold=1.0,
    tags=['athena', 'citation'],
)
class CitationFidelity(BaseMetric):
    # Regex for citations: Matches [quote.x] or [items[0].y]
    CITATION_PATTERN = re.compile(r'\[([\w\d\.\_\-\[\]\'\"]+)\]')
    _MISSING = object()

    def __init__(
        self,
        check_values: bool = True,
        window_chars: int = 150,
        min_shared_tokens: int = 2,
        fuzzy_threshold: float = 0.88,
        numeric_tolerance: float = 0.02,
        **kwargs,
    ):
        """
        Args:
            check_values (bool): If True, verifies the JSON value appears in the preceding text.
                                 If False, only checks that the JSON path exists.
            window_chars (int): How many characters back to look for the value.
            min_shared_tokens (int): Min number of shared tokens for overlap match.
            fuzzy_threshold (float): Similarity threshold for fuzzy matching.
            numeric_tolerance (float): Relative tolerance for numeric comparisons.
        """
        super().__init__(**kwargs)
        self.check_values = check_values
        self.window_chars = window_chars
        self.min_shared_tokens = min_shared_tokens
        self.fuzzy_threshold = fuzzy_threshold
        self.numeric_tolerance = numeric_tolerance

    def _parse_path(self, path: str) -> List[Union[str, int]]:
        tokens: List[Union[str, int]] = []
        i = 0
        while i < len(path):
            if path[i] == '.':
                i += 1
                continue
            if path[i] == '[':
                end = path.find(']', i)
                if end == -1:
                    return []
                inner = path[i + 1 : end].strip()
                if len(inner) >= 2 and inner[0] in ("'", '"') and inner[-1] == inner[0]:
                    tokens.append(inner[1:-1])
                elif inner.isdigit():
                    tokens.append(int(inner))
                else:
                    return []
                i = end + 1
                continue
            j = i
            while j < len(path) and path[j] not in '.[':
                j += 1
            tokens.append(path[i:j])
            i = j
        return tokens

    def _resolve_json_path(self, data: Union[Dict, List], path: str) -> Any:
        keys = self._parse_path(path)
        if not keys:
            return self._MISSING
        current: Any = data
        try:
            for key in keys:
                if isinstance(key, int):
                    current = current[key]
                elif isinstance(current, dict):
                    current = current[key]
                else:
                    return self._MISSING
            return current
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return self._MISSING

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).lower().strip()

    def _get_text_window(self, full_text: str, citation_start: int) -> str:
        start_idx = max(0, citation_start - self.window_chars)
        window_text = full_text[start_idx:citation_start]
        last_boundary = max(
            window_text.rfind('.'), window_text.rfind('!'), window_text.rfind('?')
        )
        if last_boundary != -1 and last_boundary + 1 < len(window_text):
            return window_text[last_boundary + 1 :].strip()
        return window_text

    def _coerce_number(self, text: str) -> Union[float, None]:
        cleaned = text.lower().strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace(',', '')
        match = re.match(r'^\$?\s*([+-]?\d*\.?\d+)\s*([kmb])?$', cleaned)
        if not match:
            return None
        value = float(match.group(1))
        suffix = match.group(2)
        if suffix == 'k':
            value *= 1_000
        elif suffix == 'm':
            value *= 1_000_000
        elif suffix == 'b':
            value *= 1_000_000_000
        return value

    def _extract_numbers(self, text: str) -> List[float]:
        numbers = []
        for match in re.finditer(r'\$?\s*\d[\d,]*\.?\d*\s*[kmb]?', text.lower()):
            candidate = match.group(0).strip()
            num = self._coerce_number(candidate)
            if num is not None:
                numbers.append(num)
        return numbers

    def _value_is_supported(
        self, full_text: str, citation_start: int, json_val: Any
    ) -> Tuple[bool, str]:
        """Checks if json_val is supported by the text preceding the citation."""
        if json_val is None:
            return False, 'JSON value is null.'

        str_val = self._normalize_text(str(json_val))
        if not str_val:
            return True, 'Empty value.'

        # Get window of text before citation
        window_text = self._normalize_text(
            self._get_text_window(full_text, citation_start)
        )

        # 1. Exact Substring Match (Case-insensitive)
        if str_val in window_text:
            return True, 'Exact match.'

        # 2. Number Match (ignoring symbols)
        # "2,500,000" (JSON) vs "$2.5M" (Text)
        num_val = self._coerce_number(str_val)
        if num_val is not None:
            for num in self._extract_numbers(window_text):
                if num == 0:
                    if abs(num_val) <= self.numeric_tolerance:
                        return True, 'Numeric match.'
                else:
                    rel_err = abs(num - num_val) / abs(num)
                    if rel_err <= self.numeric_tolerance:
                        return True, 'Numeric match.'

        # 3. Token Overlap (For strings like "Dental Office" vs "DENTISTO")
        # Do they share significant words?
        val_tokens = set(re.findall(r'\w{4,}', str_val))  # Only words 4+ chars
        text_tokens = set(re.findall(r'\w{4,}', window_text))

        shared = val_tokens.intersection(text_tokens)
        if val_tokens and len(shared) >= self.min_shared_tokens:
            return True, 'Token overlap.'

        # 4. Fuzzy Matching against sentence window
        if window_text and len(str_val) >= 4:
            ratio = SequenceMatcher(None, str_val, window_text).ratio()
            if ratio >= self.fuzzy_threshold:
                return True, 'Fuzzy match.'

        return False, 'No match in window.'

    @trace(name='CitationFidelity', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)

        try:
            json_data = (
                json.loads(item.expected_output)
                if isinstance(item.expected_output, str)
                else item.expected_output
            )
        except Exception:
            return MetricEvaluationResult(
                score=np.nan, explanation='Invalid JSON input.'
            )

        matches = list(self.CITATION_PATTERN.finditer(item.actual_output))
        if not matches:
            return MetricEvaluationResult(score=1.0, explanation='No citations found.')

        verdicts = []
        valid_count = 0

        for match in matches:
            raw_path = match.group(1)
            # Clean prefixes often hallucinated by LLMs
            clean_path = re.sub(r'^(Source:|cite:)', '', raw_path).strip()

            # 1. Resolve Path
            json_val = self._resolve_json_path(json_data, clean_path)
            path_exists = json_val is not self._MISSING

            is_supported = False
            reason = ''

            if not path_exists:
                reason = 'Path not found in JSON.'
            elif not self.check_values:
                # If we don't check values, path existence is enough
                is_supported = True
                reason = 'Path valid (Value check skipped).'
            else:
                # Check value against text
                is_supported, reason = self._value_is_supported(
                    item.actual_output, match.start(), json_val
                )

            if path_exists and is_supported:
                valid_count += 1

            verdicts.append(
                CitationVerdict(
                    citation_text=match.group(0),
                    path_referenced=clean_path,
                    json_value=str(json_val),
                    is_valid_path=path_exists,
                    is_supported=is_supported,
                    reason=reason,
                )
            )

        score = valid_count / len(matches)

        return MetricEvaluationResult(
            score=score,
            explanation=f'Fidelity: {score:.0%} ({valid_count}/{len(matches)})',
            signals=CitationFidelityResult(
                score=score,
                total_citations=len(matches),
                valid_citations=valid_count,
                verdicts=verdicts,
            ),
        )

    def get_signals(self, result: CitationFidelityResult) -> List[SignalDescriptor]:
        signals = []
        signals.append(
            SignalDescriptor(
                name='fidelity_score',
                description='Score',
                extractor=lambda r: r.score,
                headline_display=True,
            )
        )

        for i, v in enumerate(result.verdicts):
            icon = '✅' if v.is_supported else '❌'
            signals.extend(
                [
                    SignalDescriptor(
                        name='citation',
                        group=f'{icon} {v.citation_text}',
                        description='Citation',
                        extractor=lambda r, idx=i: r.verdicts[idx].citation_text,
                    ),
                    SignalDescriptor(
                        name='json_value',
                        group=f'{icon} {v.citation_text}',
                        description='JSON Value',
                        extractor=lambda r, idx=i: r.verdicts[idx].json_value,
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=f'{icon} {v.citation_text}',
                        description='Reason',
                        extractor=lambda r, idx=i: r.verdicts[idx].reason,
                    ),
                ]
            )
        return signals
