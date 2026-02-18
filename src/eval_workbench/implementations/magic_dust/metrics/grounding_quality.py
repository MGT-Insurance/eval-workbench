from __future__ import annotations

import math
import numpy as np
from typing import Any

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field


class GroundingFieldCheck(RichBaseModel):
    field_name: str
    required: bool
    has_answer: bool
    confidence_pct: int | None = None
    confidence_pct_needed: int = 0
    meets_confidence_threshold: bool = False
    default_applied: bool = False
    type_valid: bool = True
    options_valid: bool = True
    has_summary: bool = False


class GroundingQualitySignals(RichBaseModel):
    required_field_coverage: float = Field(default=0.0)
    confidence_threshold_pass_rate: float = Field(default=0.0)
    schema_adherence_rate: float = Field(default=0.0)
    default_applied_count: int = Field(default=0)
    total_fields: int = Field(default=0)
    required_fields: int = Field(default=0)
    checked_fields: list[GroundingFieldCheck] = Field(default_factory=list)


@metric(
    name='Magic Dust Grounding Quality',
    key='magic_dust_grounding_quality',
    description='Heuristic quality checks for grounding extraction output.',
    required_fields=[],
    optional_fields=['additional_input', 'additional_output'],
    default_threshold=0.75,
    score_range=(0, 1),
    tags=['magic_dust', 'heuristic'],
)
class MagicDustGroundingQuality(BaseMetric):
    """
    Simple non-LLM validation for grounding extraction quality.

    Uses:
    - `additional_input.requested_data` as schema/requirements
    - `additional_output.parsed_output` as model-produced field answers
    """

    def __init__(
        self,
        required_coverage_weight: float = 0.50,
        confidence_weight: float = 0.30,
        schema_weight: float = 0.20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        total = required_coverage_weight + confidence_weight + schema_weight
        if total <= 0:
            raise ValueError('Metric weights must sum to a positive value.')
        # Normalize so callers can pass any proportional values.
        self.required_coverage_weight = required_coverage_weight / total
        self.confidence_weight = confidence_weight / total
        self.schema_weight = schema_weight / total

    @trace(name='MagicDustGroundingQuality', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        requested_data = (item.additional_input or {}).get('requested_data') or {}
        parsed_output = (item.additional_output or {}).get('parsed_output') or {}

        if not isinstance(requested_data, dict) or not requested_data:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='Missing additional_input.requested_data schema.',
                signals=GroundingQualitySignals(),
            )
        if not isinstance(parsed_output, dict) or not parsed_output:
            return MetricEvaluationResult(
                score=np.nan,
                explanation='Missing additional_output.parsed_output payload.',
                signals=GroundingQualitySignals(total_fields=len(requested_data)),
            )

        checks: list[GroundingFieldCheck] = []
        required_total = 0
        required_answered = 0
        confidence_passed = 0
        schema_passed = 0
        default_applied_count = 0

        for field_name, schema in requested_data.items():
            schema_dict = schema if isinstance(schema, dict) else {}
            output_field = parsed_output.get(field_name)
            output_dict = output_field if isinstance(output_field, dict) else {}

            required = bool(schema_dict.get('required', False))
            if required:
                required_total += 1

            answer = output_dict.get('answer')
            has_answer = self._has_answer(answer)
            if required and has_answer:
                required_answered += 1

            confidence_pct = self._coerce_int(output_dict.get('confidence_pct'))
            confidence_needed = self._coerce_int(
                schema_dict.get('confidence_pct_needed'), default=0
            )
            if confidence_needed is None:
                confidence_needed = 0
            meets_confidence = (
                confidence_pct is not None and confidence_pct >= confidence_needed
            )
            if meets_confidence:
                confidence_passed += 1

            default_applied = self._same_value(answer, schema_dict.get('default'))
            if default_applied:
                default_applied_count += 1

            type_valid = self._type_valid(answer, schema_dict.get('type'))
            options_valid = self._options_valid(answer, schema_dict.get('options'))
            has_summary = self._has_answer(output_dict.get('summary'))
            if type_valid and options_valid and has_summary:
                schema_passed += 1

            checks.append(
                GroundingFieldCheck(
                    field_name=field_name,
                    required=required,
                    has_answer=has_answer,
                    confidence_pct=confidence_pct,
                    confidence_pct_needed=confidence_needed,
                    meets_confidence_threshold=meets_confidence,
                    default_applied=default_applied,
                    type_valid=type_valid,
                    options_valid=options_valid,
                    has_summary=has_summary,
                )
            )

        total_fields = len(requested_data)
        required_coverage = (
            (required_answered / required_total) if required_total > 0 else 1.0
        )
        confidence_rate = confidence_passed / total_fields if total_fields > 0 else 0.0
        schema_rate = schema_passed / total_fields if total_fields > 0 else 0.0

        # Weighted score emphasizing required-field completeness first.
        score = self._weighted_score(required_coverage, confidence_rate, schema_rate)

        signals = GroundingQualitySignals(
            required_field_coverage=required_coverage,
            confidence_threshold_pass_rate=confidence_rate,
            schema_adherence_rate=schema_rate,
            default_applied_count=default_applied_count,
            total_fields=total_fields,
            required_fields=required_total,
            checked_fields=checks,
        )

        explanation = (
            f"Required coverage={required_coverage:.2f}, "
            f"confidence pass rate={confidence_rate:.2f}, "
            f"schema adherence={schema_rate:.2f}, "
            f"defaults used={default_applied_count}/{total_fields}"
        )

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=signals,
        )

    def get_signals(self, result: GroundingQualitySignals) -> list[SignalDescriptor]:
        """Return human-readable signals explaining score components."""
        signals: list[SignalDescriptor] = [
            SignalDescriptor(
                name='Overall Quality',
                description='Weighted grounding quality score',
                group='Overview',
                extractor=lambda r, s=self: f"{s._weighted_score(r.required_field_coverage, r.confidence_threshold_pass_rate, r.schema_adherence_rate) * 100:.0f}%",
                headline_display=True,
            ),
            SignalDescriptor(
                name='Required Field Coverage',
                description='Required fields with non-empty answers',
                group='Coverage',
                extractor=lambda r: f'{r.required_field_coverage * 100:.0f}%',
                headline_display=True,
            ),
            SignalDescriptor(
                name='Confidence Threshold Pass Rate',
                description='Fields meeting confidence_pct_needed',
                group='Confidence',
                extractor=lambda r: f'{r.confidence_threshold_pass_rate * 100:.0f}%',
                headline_display=True,
            ),
            SignalDescriptor(
                name='Schema Adherence',
                description='Fields with valid type/options and non-empty summary',
                group='Schema',
                extractor=lambda r: f'{r.schema_adherence_rate * 100:.0f}%',
                headline_display=True,
            ),
            SignalDescriptor(
                name='Defaults Applied',
                description='Number of fields that used default answer',
                group='Defaults',
                extractor=lambda r: f'{r.default_applied_count}/{r.total_fields}',
                headline_display=True,
            ),
            SignalDescriptor(
                name='Missing Required Fields',
                description='Required fields without answers',
                group='Coverage',
                extractor=lambda r: ', '.join(
                    c.field_name
                    for c in r.checked_fields
                    if c.required and not c.has_answer
                )
                or 'None',
            ),
            SignalDescriptor(
                name='Fields Below Confidence Threshold',
                description='Fields where confidence is below required threshold',
                group='Confidence',
                extractor=lambda r: ', '.join(
                    c.field_name
                    for c in r.checked_fields
                    if not c.meets_confidence_threshold
                )
                or 'None',
            ),
            SignalDescriptor(
                name='Schema Issues',
                description='Fields failing type/options/summary checks',
                group='Schema',
                extractor=lambda r: '; '.join(
                    f"{c.field_name}("
                    + ', '.join(
                        issue
                        for issue, failed in (
                            ('type', not c.type_valid),
                            ('options', not c.options_valid),
                            ('summary', not c.has_summary),
                        )
                        if failed
                    )
                    + ')'
                    for c in r.checked_fields
                    if (not c.type_valid) or (not c.options_valid) or (not c.has_summary)
                )
                or 'None',
            ),
        ]
        return signals

    def _weighted_score(
        self,
        required_coverage: float,
        confidence_rate: float,
        schema_rate: float,
    ) -> float:
        return (
            self.required_coverage_weight * required_coverage
            + self.confidence_weight * confidence_rate
            + self.schema_weight * schema_rate
        )

    @staticmethod
    def _coerce_int(value: Any, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return default
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except Exception:
                return default
        return default

    @staticmethod
    def _has_answer(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @staticmethod
    def _same_value(left: Any, right: Any) -> bool:
        if left is None and right is None:
            return True
        return left == right

    @staticmethod
    def _type_valid(answer: Any, expected_type: Any) -> bool:
        if answer is None:
            # Null can still be valid for optional or defaulted fields.
            return True
        expected = (str(expected_type).strip().lower() if expected_type else '')
        if not expected:
            return True
        if expected == 'boolean':
            return isinstance(answer, bool)
        if expected == 'int':
            return isinstance(answer, int) and not isinstance(answer, bool)
        if expected == 'string':
            return isinstance(answer, str)
        return True

    @staticmethod
    def _options_valid(answer: Any, options: Any) -> bool:
        if not isinstance(options, list) or not options:
            return True
        if answer is None:
            return True
        return answer in options
