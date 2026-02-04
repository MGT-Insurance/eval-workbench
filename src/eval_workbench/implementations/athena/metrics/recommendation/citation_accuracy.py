import re
from typing import List, Literal, Optional, Set

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from eval_workbench.shared.metrics.flattening import flatten_paths

logger = get_logger(__name__)


class CitationAccuracyVerdict(RichBaseModel):
    citation_text: str = Field(..., description='The citation found in the text.')
    citation_number: int = Field(..., description='The numeric citation index.')
    source: Optional[str] = Field(None, description='Reference source string.')
    is_scorable: bool = Field(..., description='True if included in score.')
    is_valid: bool = Field(..., description='True if citation is validated.')
    reason: str = Field(..., description='Verdict explanation.')
    missing_fields: Optional[List[str]] = Field(
        None, description='Fields missing from input data.'
    )


class CitationAccuracyResult(RichBaseModel):
    score: float = Field(...)
    total_citations: int = Field(...)
    scorable_citations: int = Field(...)
    valid_citations: int = Field(...)
    verdicts: List[CitationAccuracyVerdict] = Field(...)


def _extract_source_fields(source: Optional[str]) -> List[str]:
    if not source:
        return []
    if ' - ' in source:
        _, remainder = source.split(' - ', 1)
    else:
        remainder = source
    parts = [p.strip() for p in remainder.split(',')]
    fields = []
    for part in parts:
        if not part:
            continue
        if ':' in part:
            part = part.split(':', 1)[0].strip()
        part = part.strip('`\'"')
        if not part:
            continue
        if re.search(r'\s', part):
            # Try to salvage a field-like token from free text
            tokens = re.findall(r'[A-Za-z_][\w\.]*', part)
            for token in tokens:
                if ('_' in token or '.' in token) or (
                    re.search(r'[A-Z]', token) and len(token) >= 6
                ):
                    fields.append(token.lower())
            continue
        if not re.match(r'^[A-Za-z_][\w\.]*$', part):
            continue
        if ('_' in part or '.' in part) or (
            re.search(r'[A-Z]', part) and len(part) >= 6
        ):
            fields.append(part.lower())
    if fields:
        return fields

    # Fallback: scan full source for field-like tokens
    colon_tokens = re.findall(r':\s*([A-Za-z_][\w\.]*)', source)
    for token in colon_tokens:
        if ('_' in token or '.' in token) or (
            re.search(r'[A-Z]', token) and len(token) >= 6
        ):
            fields.append(token.lower())
    if fields:
        return fields

    tokens = re.findall(r'[A-Za-z_][\w\.]*', source)
    for token in tokens:
        if ('_' in token or '.' in token) or (
            re.search(r'[A-Z]', token) and len(token) >= 6
        ):
            fields.append(token.lower())
    return fields


@metric(
    name='Citation Accuracy',
    key='citation_accuracy',
    description='Validates numeric citations against actual_reference and input data.',
    required_fields=[],
    optional_fields=[
        'actual_output',
        'additional_output',
        'additional_input',
        'actual_reference',
    ],
    default_threshold=1.0,
    tags=['athena', 'heuristic'],
)
class CitationAccuracy(BaseMetric):
    CITATION_PATTERN = re.compile(r'\[(\d+)\]')

    def __init__(
        self,
        validation_mode: Literal['ref_only', 'ref_plus_input'] = 'ref_only',
        output_key: str = 'brief_recommendation',
        **kwargs,
    ):
        """
        Args:
            validation_mode: "ref_only" (citation exists in actual_reference) or
                             "ref_plus_input" (also verify referenced fields exist in input).
            output_key: Key in item.additional_output to analyze (falls back to actual_output).
        """
        super().__init__(**kwargs)
        self.validation_mode = validation_mode
        self.output_key = output_key

    def _field_exists(
        self,
        field: str,
        paths: Set[str],
        keys: Set[str],
        input_blob: str,
    ) -> bool:
        if field in keys:
            return True
        if field in paths:
            return True
        suffix = f'.{field}'
        if any(p.endswith(suffix) for p in paths):
            return True
        return field in input_blob

    @trace(name='CitationAccuracy', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        # Select the text source
        text = ''
        if self.output_key and item.additional_output:
            val = item.additional_output.get(self.output_key)
            if val is not None:
                text = str(val)
        if not text:
            text = item.actual_output or ''

        matches = list(self.CITATION_PATTERN.finditer(text))
        if not matches:
            return MetricEvaluationResult(score=1.0, explanation='No citations found.')

        reference_list = item.actual_reference or []
        reference_by_number = {
            int(str(ref.get('number'))): ref
            for ref in reference_list
            if isinstance(ref, dict) and str(ref.get('number', '')).isdigit()
        }

        paths: Set[str] = set()
        keys: Set[str] = set()
        input_blob = ''
        if self.validation_mode == 'ref_plus_input' and item.additional_input:
            paths, keys = flatten_paths(item.additional_input)
            input_blob = str(item.additional_input)
            try:
                import json as _json

                input_blob = _json.dumps(item.additional_input)
            except Exception:
                pass
            input_blob = input_blob.lower()

        verdicts: List[CitationAccuracyVerdict] = []
        scorable_count = 0
        valid_count = 0

        for match in matches:
            citation_num = int(match.group(1))
            ref = reference_by_number.get(citation_num)
            source = ref.get('source') if ref else None

            is_scorable = True
            is_valid = False
            reason = ''
            missing_fields: Optional[List[str]] = None

            if not ref:
                reason = 'Citation number not found in actual_reference.'
            elif self.validation_mode == 'ref_only':
                is_valid = True
                reason = 'Reference exists.'
            else:
                fields = _extract_source_fields(source)
                if not fields:
                    is_scorable = False
                    reason = 'Unverifiable source (no fields); excluded from score.'
                else:
                    missing = [
                        f
                        for f in fields
                        if not self._field_exists(f, paths, keys, input_blob)
                    ]
                    if missing:
                        missing_fields = missing
                        reason = f'Missing fields in input: {", ".join(missing)}.'
                    else:
                        is_valid = True
                        reason = 'All referenced fields found in input.'

            if is_scorable:
                scorable_count += 1
                if is_valid:
                    valid_count += 1

            verdicts.append(
                CitationAccuracyVerdict(
                    citation_text=match.group(0),
                    citation_number=citation_num,
                    source=source,
                    is_scorable=is_scorable,
                    is_valid=is_valid,
                    reason=reason,
                    missing_fields=missing_fields,
                )
            )

        if scorable_count == 0:
            return MetricEvaluationResult(
                score=1.0,
                explanation='No scorable citations.',
                signals=CitationAccuracyResult(
                    score=1.0,
                    total_citations=len(matches),
                    scorable_citations=scorable_count,
                    valid_citations=valid_count,
                    verdicts=verdicts,
                ),
            )

        score = valid_count / scorable_count
        return MetricEvaluationResult(
            score=score,
            explanation=f'Accuracy: {score:.0%} ({valid_count}/{scorable_count})',
            signals=CitationAccuracyResult(
                score=score,
                total_citations=len(matches),
                scorable_citations=scorable_count,
                valid_citations=valid_count,
                verdicts=verdicts,
            ),
        )

    def get_signals(self, result: CitationAccuracyResult) -> List[SignalDescriptor]:
        signals = [
            SignalDescriptor(
                name='accuracy_score',
                description='Score',
                extractor=lambda r: r.score,
                headline_display=True,
            )
        ]

        for i, v in enumerate(result.verdicts):
            icon = '✅' if v.is_valid else '❌'
            group = f'{icon} {v.citation_text}'
            signals.extend(
                [
                    SignalDescriptor(
                        name='citation_number',
                        group=group,
                        description='Citation number',
                        extractor=lambda r, idx=i: r.verdicts[idx].citation_number,
                    ),
                    SignalDescriptor(
                        name='source',
                        group=group,
                        description='Source',
                        extractor=lambda r, idx=i: r.verdicts[idx].source,
                    ),
                    SignalDescriptor(
                        name='scorable',
                        group=group,
                        description='Scorable',
                        extractor=lambda r, idx=i: r.verdicts[idx].is_scorable,
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=group,
                        description='Reason',
                        extractor=lambda r, idx=i: r.verdicts[idx].reason,
                    ),
                ]
            )

        return signals
