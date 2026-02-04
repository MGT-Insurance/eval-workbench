from typing import Dict, List, Literal

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

logger = get_logger(__name__)


class ReasoningGap(RichBaseModel):
    """A specific concept mentioned by the human but missed by the AI."""

    concept: str = Field(..., description='The specific risk factor or reason missed.')
    impact: str = Field(
        ..., description="Why this omission matters (e.g., 'Critical', 'Nuance')."
    )
    model_config = {'extra': 'forbid'}


class ReasoningMatch(RichBaseModel):
    """A concept where both Human and AI aligned."""

    concept: str = Field(
        ..., description='The risk factor or reason identified by both.'
    )
    model_config = {'extra': 'forbid'}


class DecisionQualityResult(RichBaseModel):
    """The structured output for the Decision Quality metric."""

    # Outcome Component
    overall_score: float = Field(
        ..., description='Combined score of outcome and reasoning.'
    )
    outcome_match: bool = Field(
        ..., description='True if the final decisions (Approve/Decline) match.'
    )
    outcome_score: float = Field(..., description='1.0 for match, 0.0 for mismatch.')
    human_decision_detected: str = Field(
        ..., description='Normalized decision extracted from Human Notes.'
    )
    ai_decision_detected: str = Field(
        ..., description='Normalized decision extracted from AI Output.'
    )

    # Reasoning Component
    reasoning_score: float | None = Field(
        ...,
        description='Score based on coverage of human concepts. None when no reasoning is applicable.',
    )
    missing_concepts: List[ReasoningGap] = Field(
        default_factory=list, description='List of reasons the AI missed.'
    )
    matched_concepts: List[ReasoningMatch] = Field(
        default_factory=list, description='List of reasons both agreed on.'
    )
    model_config = {'extra': 'forbid'}


ImpactLevel = Literal['Critical', 'High', 'Medium', 'Low']


IMPACT_WEIGHTS: Dict[str, float] = {
    'Critical': 1.0,
    'High': 0.8,
    'Medium': 0.5,
    'Low': 0.2,
}


class HumanRiskFactor(RichBaseModel):
    """A single concept with an impact level."""

    concept: str = Field(..., description='A specific risk factor or instruction.')
    impact: ImpactLevel = Field(
        ...,
        description='Severity of the factor (Critical, High, Medium, Low).',
    )
    model_config = {'extra': 'forbid'}


class HumanRiskFactors(RichBaseModel):
    risk_factors: List[HumanRiskFactor] = Field(
        ...,
        description='Distinct risk concerns, instructions, or positive factors.',
        examples=[
            [
                {'concept': 'High crime area', 'impact': 'High'},
                {'concept': 'Sprinklers required', 'impact': 'Medium'},
                {'concept': 'Financials approved', 'impact': 'Low'},
            ]
        ],
    )
    model_config = {'extra': 'forbid'}


class RiskExtractorInput(RichBaseModel):
    human_notes: str = Field(
        ..., description="The underwriter's notes serving as the ground truth."
    )
    model_config = {'extra': 'forbid'}


class RiskExtractor(BaseMetric[RiskExtractorInput, HumanRiskFactors]):
    instruction = """
    You are a Senior Underwriting Auditor.
    Your Goal: Convert the informal "Human Underwriter Notes" into a strict checklist of
    risk factors or instructions with an impact level.

    RULES:
    1. Extract specific concerns (e.g., "Roof age", "Revenue discrepancy").
    2. Extract specific instructions (e.g., "Order inspection", "Set for pre-renewal review").
    3. Extract positive notes if they justify the decision (e.g., "No 24-hour ops confirmed").
    4. Ignore signatures, names, and administrative noise.

    Output strictly a list of concepts with an impact level:
    - Critical: material underwriting blocker
    - High: strong risk signal
    - Medium: actionable but moderate concern
    - Low: minor or resolved note
    """
    input_model = RiskExtractorInput
    output_model = HumanRiskFactors
    examples = [
        (
            RiskExtractorInput(
                human_notes='Re-approving this one. Had originally referred for 24-hour operations, agent confirmed they close at 9pm.'
            ),
            HumanRiskFactors(
                risk_factors=[
                    HumanRiskFactor(
                        concept='Originally referred for 24-hour operations; confirmed closes at 9pm',
                        impact='Low',
                    )
                ]
            ),
        ),
        (
            RiskExtractorInput(
                human_notes='Significant differences between MD and agent entered revenue and payroll, but there is no quick way to validate online for a new entity. This should be set for pre-renewal review to audit or increase.'
            ),
            HumanRiskFactors(
                risk_factors=[
                    HumanRiskFactor(
                        concept='Revenue discrepancy between third-party and agent input',
                        impact='High',
                    ),
                    HumanRiskFactor(
                        concept='Payroll discrepancy between third-party and agent input',
                        impact='High',
                    ),
                    HumanRiskFactor(
                        concept='Set for pre-renewal review or audit due to unverifiable data',
                        impact='Medium',
                    ),
                ]
            ),
        ),
    ]


class CoverageCheckInput(RichBaseModel):
    required_factors: List[HumanRiskFactor] = Field(
        ..., description='Checklist extracted from human notes.'
    )
    ai_output: str = Field(..., description="The AI agent's full recommendation.")
    model_config = {'extra': 'forbid'}


class CoverageJudgeOutput(RichBaseModel):
    missing_concepts: List[str]
    matched_concepts: List[str]
    model_config = {'extra': 'forbid'}


class CoverageJudge(BaseMetric[CoverageCheckInput, CoverageJudgeOutput]):
    instruction = """
    You are a Compliance Checker.

    Task: Check if the "AI Recommendation" addresses the "Required Factors".

    For each concept in "Required Factors":
    1. Search the "AI Recommendation" for any mention of this concept.
    2. Mark it as "Matched" if the AI discusses it (even if it disagrees or uses different words).
    3. Mark it as "Missing" if the AI is completely silent on the topic.

    Do not re-interpret the human notes. The list provided is the absolute truth.
    Use the exact concept text from "Required Factors" in your outputs.
    """
    input_model = CoverageCheckInput
    output_model = CoverageJudgeOutput
    examples = [
        (
            CoverageCheckInput(
                required_factors=[
                    HumanRiskFactor(
                        concept='Revenue discrepancy between third-party and agent input',
                        impact='High',
                    ),
                    HumanRiskFactor(
                        concept='Payroll discrepancy between third-party and agent input',
                        impact='High',
                    ),
                    HumanRiskFactor(
                        concept='Set for pre-renewal review or audit due to unverifiable data',
                        impact='Medium',
                    ),
                ],
                ai_output=(
                    'We noted a revenue discrepancy versus third-party data and the payroll figures do not align. '
                    'Recommend placing this account into a pre-renewal audit bucket.'
                ),
            ),
            CoverageJudgeOutput(
                matched_concepts=[
                    'Revenue discrepancy between third-party and agent input',
                    'Payroll discrepancy between third-party and agent input',
                    'Set for pre-renewal review or audit due to unverifiable data',
                ],
                missing_concepts=[],
            ),
        ),
        (
            CoverageCheckInput(
                required_factors=[
                    HumanRiskFactor(
                        concept='New organization with limited operating history',
                        impact='Medium',
                    ),
                    HumanRiskFactor(
                        concept='Trigger inspection if issued',
                        impact='Medium',
                    ),
                    HumanRiskFactor(
                        concept='Liability not offered in this state',
                        impact='High',
                    ),
                ],
                ai_output=(
                    'This is a new business; recommend issuing contents-only coverage and '
                    'triggering an inspection upon bind.'
                ),
            ),
            CoverageJudgeOutput(
                matched_concepts=[
                    'New organization with limited operating history',
                    'Trigger inspection if issued',
                ],
                missing_concepts=['Liability not offered in this state'],
            ),
        ),
    ]


@metric(
    name='Decision Quality',
    key='decision_quality',
    description='Evaluates if the AI made the right decision for the right reasons.',
    required_fields=['actual_output', 'expected_output'],
    score_range=(0, 1),
    tags=['athena'],
)
class DecisionQuality(BaseMetric):
    def __init__(
        self,
        outcome_weight: float = 1.0,
        reasoning_weight: float = 0.0,
        hard_fail_on_outcome_mismatch: bool = True,
        recommendation_column_name: str = 'brief_recommendation',
        **kwargs,
    ):
        """
        Args:
            outcome_weight: Weight given to the binary outcome match (default: 0.5).
            reasoning_weight: Weight given to the reasoning completeness score (default: 0.5).
            hard_fail_on_outcome_mismatch: If True, forces the total score to 0.0 if the
                                           outcome does not match, regardless of reasoning (default: True).
            **kwargs: Arguments passed to the underlying extractor/judge.
        """
        super().__init__(**kwargs)
        self.outcome_weight = outcome_weight
        self.reasoning_weight = reasoning_weight
        self.hard_fail_on_outcome_mismatch = hard_fail_on_outcome_mismatch
        self.recommendation_column_name = recommendation_column_name

        self.risk_extractor = RiskExtractor(**kwargs)
        self.coverage_judge = CoverageJudge(**kwargs)

    @staticmethod
    def _impact_weight(impact: str) -> float:
        return IMPACT_WEIGHTS.get(impact, IMPACT_WEIGHTS['Medium'])

    async def execute(
        self, dataset_item: DatasetItem, **kwargs
    ) -> MetricEvaluationResult:
        # Parse Decisions
        human_decision = dataset_item.expected_output or 'UNKNOWN'
        ai_decision = dataset_item.actual_output or 'UNKNOWN'

        # Simple string equality
        outcome_match = (
            human_decision.strip().lower() == ai_decision.strip().lower()
        ) and (human_decision != 'UNKNOWN')
        outcome_score = 1.0 if outcome_match else 0.0

        # Reasoning Check (LLM Logic)
        human_text = (
            dataset_item.acceptance_criteria[0]
            if dataset_item.acceptance_criteria
            else human_decision
        )
        ai_text = dataset_item.additional_output.get(
            self.recommendation_column_name, dataset_item.actual_output
        )

        extraction_input = RiskExtractorInput(human_notes=human_text)
        ground_truth = await self.risk_extractor.execute(extraction_input)
        required_factors = ground_truth.risk_factors or []

        if not required_factors:
            reasoning_score = None
            matched_concepts: List[ReasoningMatch] = []
            missing_concepts: List[ReasoningGap] = []
        else:
            check_input = CoverageCheckInput(
                required_factors=required_factors, ai_output=ai_text
            )
            coverage_result = await self.coverage_judge.execute(check_input)

            matched_set = set(coverage_result.matched_concepts or [])
            matched_factors = [
                factor for factor in required_factors if factor.concept in matched_set
            ]
            missing_factors = [
                factor
                for factor in required_factors
                if factor.concept not in matched_set
            ]

            matched_concepts = [
                ReasoningMatch(concept=factor.concept) for factor in matched_factors
            ]
            missing_concepts = [
                ReasoningGap(concept=factor.concept, impact=factor.impact)
                for factor in missing_factors
            ]

            total_weight = sum(
                self._impact_weight(factor.impact) for factor in required_factors
            )
            matched_weight = sum(
                self._impact_weight(factor.impact) for factor in matched_factors
            )
            reasoning_score = matched_weight / total_weight if total_weight > 0 else 1.0

        # Weighted Score
        reasoning_component = (
            reasoning_score * self.reasoning_weight
            if reasoning_score is not None
            else 0.0
        )
        final_score = (outcome_score * self.outcome_weight) + reasoning_component

        # Apply Hard Fail Logic
        if self.hard_fail_on_outcome_mismatch and not outcome_match:
            final_score = 0.0

        result_data = DecisionQualityResult(
            overall_score=final_score,
            outcome_match=outcome_match,
            outcome_score=outcome_score,
            human_decision_detected=human_decision,
            ai_decision_detected=ai_decision,
            reasoning_score=reasoning_score,
            missing_concepts=missing_concepts,
            matched_concepts=matched_concepts,
        )

        self.compute_cost_estimate([self.risk_extractor, self.coverage_judge])

        return MetricEvaluationResult(score=final_score, signals=result_data)

    def get_signals(self, result: DecisionQualityResult) -> List[SignalDescriptor]:
        """
        Define how the results are displayed in the UI.
        Uses extractors to pull data from the result object dynamically.
        """
        signals = []

        # Group 1: High Level Outcome
        signals.append(
            SignalDescriptor(
                name='Outcome Match',
                description='Did the AI and Human make the same final decision?',
                group='Decision Logic',
                extractor=lambda r: r.outcome_match,
                score_mapping={True: 1.0, False: 0.0},
                headline_display=True,
            )
        )

        signals.append(
            SignalDescriptor(
                name='Decisions',
                description='Comparison of detected decisions.',
                group='Decision Logic',
                extractor=lambda r: f'Human: {r.human_decision_detected} | AI: {r.ai_decision_detected}',
            )
        )

        # Group 2: Missing Concepts (The Critical Feedback)
        for i, gap in enumerate(result.missing_concepts):
            group_name = f'Missing: {gap.concept}'
            signals.extend(
                [
                    SignalDescriptor(
                        name=f'missed_concept_{i}',
                        group=group_name,
                        extractor=lambda r, idx=i: r.missing_concepts[idx].concept,
                        description='The concept missed by the AI.',
                    ),
                    SignalDescriptor(
                        name=f'impact_{i}',
                        group=group_name,
                        extractor=lambda r, idx=i: r.missing_concepts[idx].impact,
                        description='Why this omission matters.',
                        headline_display=False,
                    ),
                ]
            )

        # Group 3: Matched Concepts (The "Good" Stuff)
        for i, match in enumerate(result.matched_concepts):
            group_name = f'Matched: {match.concept}'
            signals.extend(
                [
                    SignalDescriptor(
                        name=f'matched_concept_{i}',
                        group=group_name,
                        extractor=lambda r, idx=i: r.matched_concepts[idx].concept,
                        description='AI correctly identified this factor.',
                        headline_display=False,
                    )
                ]
            )

        return signals
