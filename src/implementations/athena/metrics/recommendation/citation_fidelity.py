import re
import json
from typing import List, Any, Dict, Union
from pydantic import Field
import numpy as np

from axion._core.logging import get_logger
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace

logger = get_logger(__name__)


class CitationVerdict(RichBaseModel):
    citation_text: str = Field(..., description="The citation found in the text.")
    path_referenced: str = Field(..., description="The JSON key/path.")
    json_value: str = Field(..., description="The value found in the JSON.")
    is_valid_path: bool = Field(..., description="True if path exists.")
    is_supported: bool = Field(
        ..., description="True if value matches text (or check skipped)."
    )
    reason: str = Field(..., description="Verdict explanation.")


class CitationFidelityResult(RichBaseModel):
    score: float = Field(...)
    total_citations: int = Field(...)
    valid_citations: int = Field(...)
    verdicts: List[CitationVerdict] = Field(...)


@metric(
    name="Citation Fidelity",
    key="citation_fidelity",
    description="Verifies that bracketed citations point to real JSON keys. Optionally checks if the value appears in the text.",
    required_fields=["actual_output", "expected_output"],
    default_threshold=1.0,
    tags=["compliance", "programmatic"],
)
class CitationFidelity(BaseMetric):
    # Regex for citations: Matches [quote.x] or [items[0].y]
    CITATION_PATTERN = re.compile(r"\[([\w\d\.\_\[\]\'\"]+)\]")

    def __init__(self, check_values: bool = True, window_chars: int = 150, **kwargs):
        """
        Args:
            check_values (bool): If True, verifies the JSON value appears in the preceding text.
                                 If False, only checks that the JSON path exists.
            window_chars (int): How many characters back to look for the value.
        """
        super().__init__(**kwargs)
        self.check_values = check_values
        self.window_chars = window_chars

    def _resolve_json_path(self, data: Union[Dict, List], path: str) -> Any:
        keys = path.split(".")
        current = data
        try:
            for key in keys:
                if "[" in key and key.endswith("]"):
                    base_key, index_part = key[:-1].split("[")
                    if base_key:
                        current = current[base_key]
                    if index_part.isdigit():
                        current = current[int(index_part)]
                    else:
                        return None
                else:
                    current = current[key]
            return current
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None

    def _value_is_supported(
        self, full_text: str, citation_start: int, json_val: Any
    ) -> bool:
        """Checks if json_val is supported by the text preceding the citation."""
        if json_val is None:
            return False

        str_val = str(json_val).lower().strip()
        if not str_val:
            return True  # Empty value is passable

        # Get window of text before citation
        start_idx = max(0, citation_start - self.window_chars)
        window_text = full_text[start_idx:citation_start].lower()

        # 1. Exact Substring Match (Case-insensitive)
        if str_val in window_text:
            return True

        # 2. Number Match (ignoring symbols)
        # "2,500,000" (JSON) vs "$2.5M" (Text)
        val_digits = re.sub(r"[^\d\.]", "", str_val)
        text_digits = re.sub(r"[^\d\.]", "", window_text)

        if val_digits and len(val_digits) > 1 and val_digits in text_digits:
            return True

        # 3. K/M Abbreviation Logic for Numbers
        try:
            f_val = float(val_digits)
            # Check Millions
            if f_val >= 1_000_000:
                short_m = f"{f_val / 1_000_000:.1f}".replace(".0", "")  # 2500000 -> 2.5
                if f"{short_m}m" in window_text or f"{short_m} m" in window_text:
                    return True
            # Check Thousands
            if f_val >= 1_000:
                short_k = f"{f_val / 1_000:.0f}"  # 2500 -> 2.5
                if f"{short_k}k" in window_text or f"{short_k} k" in window_text:
                    return True
        except ValueError:
            pass

        # 4. Token Overlap (For strings like "Dental Office" vs "DENTISTO")
        # This is a 'Hail Mary' check: do they share significant words?
        val_tokens = set(re.findall(r"\w{4,}", str_val))  # Only words 4+ chars
        text_tokens = set(re.findall(r"\w{4,}", window_text))

        if val_tokens and not val_tokens.isdisjoint(text_tokens):
            # If they share any significant word (e.g. "Dental"), pass it.
            return True

        return False

    @trace(name="CitationFidelity", capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        self._validate_required_metric_fields(item)

        try:
            json_data = (
                json.loads(item.expected_output)
                if isinstance(item.expected_output, str)
                else item.expected_output
            )
        except:
            return MetricEvaluationResult(
                score=np.nan, explanation="Invalid JSON input."
            )

        matches = list(self.CITATION_PATTERN.finditer(item.actual_output))
        if not matches:
            return MetricEvaluationResult(score=1.0, explanation="No citations found.")

        verdicts = []
        valid_count = 0

        for match in matches:
            raw_path = match.group(1)
            # Clean prefixes often hallucinated by LLMs
            clean_path = re.sub(r"^(Source:|cite:)", "", raw_path).strip()

            # 1. Resolve Path
            json_val = self._resolve_json_path(json_data, clean_path)
            path_exists = json_val is not None

            is_supported = False
            reason = ""

            if not path_exists:
                reason = "Path not found in JSON."
            elif not self.check_values:
                # If we don't check values, path existence is enough
                is_supported = True
                reason = "Path valid (Value check skipped)."
            else:
                # Check value against text
                is_supported = self._value_is_supported(
                    item.actual_output, match.start(), json_val
                )
                reason = (
                    "Valid."
                    if is_supported
                    else f"Value '{json_val}' not found in preceding text."
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
            explanation=f"Fidelity: {score:.0%} ({valid_count}/{len(matches)})",
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
                name="fidelity_score",
                description="Score",
                extractor=lambda r: r.score,
                headline_display=True,
            )
        )

        for i, v in enumerate(result.verdicts):
            icon = "✅" if v.is_supported else "❌"
            signals.extend(
                [
                    SignalDescriptor(
                        name="citation",
                        group=f"{icon} {v.citation_text}",
                        description="Citation",
                        extractor=lambda r, idx=i: r.verdicts[idx].citation_text,
                    ),
                    SignalDescriptor(
                        name="json_value",
                        group=f"{icon} {v.citation_text}",
                        description="JSON Value",
                        extractor=lambda r, idx=i: r.verdicts[idx].json_value,
                    ),
                    SignalDescriptor(
                        name="reason",
                        group=f"{icon} {v.citation_text}",
                        description="Reason",
                        extractor=lambda r, idx=i: r.verdicts[idx].reason,
                    ),
                ]
            )
        return signals
