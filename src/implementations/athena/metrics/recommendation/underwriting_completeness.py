import asyncio
from typing import Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


class CriteriaInput(RichBaseModel):
    text_to_evaluate: str = Field(..., description='The recommendation note.')
    context: Optional[str] = Field(None, description='Detailed report if needed.')


class CriteriaResult(RichBaseModel):
    score: float = Field(..., description='1.0 if passed, 0.0 if failed.')
    reasoning: str = Field(..., description='Explanation of the verdict.')
    evidence_found: Optional[str] = Field(
        None, description='Quote or snippet supporting the verdict.'
    )


class CriteriaJudge(BaseMetric[CriteriaInput, CriteriaResult]):
    input_model = CriteriaInput
    output_model = CriteriaResult


class CompletenessCriterion(RichBaseModel):
    name: str = Field(..., description='Criteria name.')
    score: float = Field(..., description='1.0 if passed, 0.0 if failed.')
    reasoning: str = Field(..., description='Explanation of the verdict.')
    evidence_found: Optional[str] = Field(
        None, description='Quote or snippet supporting the verdict.'
    )


class UnderwritingCompletenessResult(RichBaseModel):
    overall_score: float = Field(..., description='Weighted completeness score.')
    criteria: List[CompletenessCriterion] = Field(
        ..., description='Per-criteria results in evaluation order.'
    )


@metric(
    name='UnderwritingCompleteness',
    key='uw_completeness',
    description='Evaluates completeness using dedicated LLM calls for each dimension.',
    required_fields=['actual_output'],
    score_range=(0, 1),
    tags=['athena', 'agent'],
)
class UnderwritingCompleteness(BaseMetric):
    def __init__(self, weights: Optional[Dict[str, float]] = None, **kwargs):
        """
        Args:
            weights: Dictionary mapping criteria names to their score weight.
                     Default: Decision (0.4), Rationale (0.2), Evidence (0.2), NextStep (0.2).
        """
        super().__init__(**kwargs)

        # Default Weights
        self.weights = weights or {
            'Decision': 0.4,
            'Rationale': 0.2,
            'Evidence': 0.2,
            'NextStep': 0.2,
        }

        # Decision Judge
        self.decision_judge = CriteriaJudge(**kwargs)
        self.decision_judge.instruction = """
        Analyze the text for a clear Underwriting Decision.
        Return score 1.0 if: The text explicitly states "Approve", "Decline", "Refer", "Quote", or "Bind".
        Return score 0.0 if: The text is vague like "Review needed" without a clear status.
        """
        self.decision_judge.examples = [
            (
                CriteriaInput(text_to_evaluate='Recommend Decline due to roof age.'),
                CriteriaResult(
                    score=1.0,
                    reasoning="Explicitly states 'Decline'.",
                    evidence_found='Recommend Decline',
                ),
            ),
            (
                CriteriaInput(text_to_evaluate='This risk has significant issues.'),
                CriteriaResult(
                    score=0.0,
                    reasoning='No final decision status.',
                    evidence_found=None,
                ),
            ),
        ]

        # Rationale Judge
        self.rationale_judge = CriteriaJudge(**kwargs)
        self.rationale_judge.instruction = """
        Analyze the text for a Specific Risk Rationale.
        Return score 1.0 if: It cites a specific driver (e.g., "Roof Age > 20 years").
        Return score 0.0 if: It uses generic fluff like "Does not meet guidelines".
        """
        self.rationale_judge.examples = [
            (
                CriteriaInput(text_to_evaluate='Refer due to Roof Age of 25 years.'),
                CriteriaResult(
                    score=1.0,
                    reasoning="Cites specific factor 'Roof Age'.",
                    evidence_found='Roof Age of 25 years',
                ),
            ),
            (
                CriteriaInput(text_to_evaluate='Refer due to building condition.'),
                CriteriaResult(
                    score=0.0, reasoning='Too generic.', evidence_found=None
                ),
            ),
        ]

        # Evidence Judge
        self.evidence_judge = CriteriaJudge(**kwargs)
        self.evidence_judge.instruction = """
        Analyze the text for Data Evidence.
        Return score 1.0 if: It includes specific numbers, dates, or boolean flags.
        Return score 0.0 if: It makes claims without specific data points.
        """
        self.evidence_judge.examples = [
            (
                CriteriaInput(text_to_evaluate='Revenue is $1.2M.'),
                CriteriaResult(
                    score=1.0,
                    reasoning='Specific financial data point.',
                    evidence_found='$1.2M',
                ),
            ),
            (
                CriteriaInput(text_to_evaluate='Revenue is strong.'),
                CriteriaResult(
                    score=0.0, reasoning='Qualitative only.', evidence_found=None
                ),
            ),
        ]

        # Next Step Judge
        self.next_step_judge = CriteriaJudge(**kwargs)
        self.next_step_judge.instruction = """
        Analyze the text for a Clear Next Step.
        Return score 1.0 if: It instructs what to do (e.g., "Order inspection").
        Return score 0.0 if: It ends without a call to action.
        """
        self.next_step_judge.examples = [
            (
                CriteriaInput(text_to_evaluate='Approve subject to inspection.'),
                CriteriaResult(
                    score=1.0,
                    reasoning='Clear instruction.',
                    evidence_found='inspection',
                ),
            ),
            (
                CriteriaInput(text_to_evaluate='Risk is acceptable.'),
                CriteriaResult(
                    score=0.0, reasoning='Statement of fact only.', evidence_found=None
                ),
            ),
        ]

    @trace(name='UnderwritingCompleteness', capture_args=True, capture_response=True)
    async def execute(
        self, dataset_item: DatasetItem, **kwargs
    ) -> MetricEvaluationResult:
        actual_output = self.get_field(dataset_item, 'actual_output') or ''
        additional_output = dataset_item.additional_output or {}
        detailed = additional_output.get('detailed_recommendation') or ''
        brief = additional_output.get('brief_recommendation') or ''
        text = detailed or brief or actual_output

        if not text:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No recommendation text available.',
                signals=UnderwritingCompletenessResult(
                    overall_score=0.0,
                    criteria=[],
                ),
            )

        input_payload = CriteriaInput(text_to_evaluate=text, context=detailed)

        # Run Parallel Judges
        results = await asyncio.gather(
            self.decision_judge.execute(input_payload),
            self.rationale_judge.execute(input_payload),
            self.evidence_judge.execute(input_payload),
            self.next_step_judge.execute(input_payload),
        )

        criteria_order = ['Decision', 'Rationale', 'Evidence', 'NextStep']
        res_map = dict(zip(criteria_order, results))

        # --- Weighted Scoring Calculation ---
        total_score = 0.0

        # Check Decision first (Hard Gate)
        # If Decision is 0.0, the total score is 0.0 regardless of other factors.
        if res_map['Decision'].score == 0.0:
            total_score = 0.0
        else:
            # Sum weighted scores
            for key, res in res_map.items():
                weight = self.weights.get(key, 0.0)
                total_score += res.score * weight

        criteria_results = [
            CompletenessCriterion(
                name=key,
                score=res.score,
                reasoning=res.reasoning,
                evidence_found=res.evidence_found,
            )
            for key, res in res_map.items()
        ]

        return MetricEvaluationResult(
            score=total_score,
            signals=UnderwritingCompletenessResult(
                overall_score=total_score,
                criteria=criteria_results,
            ),
        )

    def get_signals(
        self, result: UnderwritingCompletenessResult
    ) -> List[SignalDescriptor]:
        signals = []

        # 1. High Level Score
        signals.append(
            SignalDescriptor(
                name='Completeness Score',
                description='Weighted sum of passing criteria',
                group='Overview',
                extractor=lambda r: f'{int(r.overall_score * 100)}%',
                headline_display=True,
            )
        )

        # 2. Detailed Breakdown
        for i, criterion in enumerate(result.criteria):
            key = criterion.name
            is_pass = criterion.score == 1.0
            signals.append(
                SignalDescriptor(
                    name=f'{key} Status',
                    group=f'{key} Analysis',
                    extractor=lambda r, idx=i: (
                        '✅ Pass' if r.criteria[idx].score == 1.0 else '❌ Fail'
                    ),
                    headline_display=True,
                )
            )

            if not is_pass:
                signals.append(
                    SignalDescriptor(
                        name=f'{key} Issue',
                        group=f'{key} Analysis',
                        extractor=lambda r, idx=i: r.criteria[idx].reasoning,
                        description='Why this criteria failed',
                    )
                )
            elif criterion.evidence_found:
                signals.append(
                    SignalDescriptor(
                        name=f'{key} Evidence',
                        group=f'{key} Analysis',
                        extractor=lambda r,
                        idx=i: f'"{r.criteria[idx].evidence_found}"',
                        headline_display=False,
                    )
                )

        return signals
