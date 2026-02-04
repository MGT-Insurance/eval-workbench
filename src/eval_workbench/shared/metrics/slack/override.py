from typing import List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from eval_workbench.shared.metrics.slack.utils import (
    build_transcript,
    extract_recommendation_type,
    find_recommendation_turn,
)

OverrideType = Literal[
    'no_override',
    'full_override',
    'partial_override',
    'pending_override',
]

OverrideReasonCategory = Literal[
    'none',
    'additional_info',
    'risk_assessment',
    'policy_exception',
    'class_code_issue',
    'rate_issue',
    'experience_judgment',
    'other',
]


class OverrideInput(RichBaseModel):
    """Input model for override detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    recommendation_type: str = Field(
        description='Type of recommendation made (approve, decline, review, hold)',
    )
    recommendation_turn: int = Field(
        description='Turn index where recommendation was made',
    )


class OverrideOutput(RichBaseModel):
    """Output model for override detection."""

    is_overridden: bool = Field(
        description='Whether the recommendation was overridden',
    )
    override_type: OverrideType = Field(
        description='Type of override detected',
    )
    original_recommendation: str = Field(
        default='',
        description='What AI recommended',
    )
    final_decision: Optional[str] = Field(
        default=None,
        description='What was actually decided',
    )
    override_reason: str = Field(
        default='',
        description='Extracted reason for override',
    )
    override_reason_category: OverrideReasonCategory = Field(
        default='none',
        description='Category of override reason',
    )
    reasoning: str = Field(
        default='',
        description='Explanation of the override assessment',
    )


class OverrideResult(RichBaseModel):
    """Result model for override analysis."""

    has_recommendation: bool = Field(
        default=False,
        description='Whether a recommendation was found to analyze',
    )
    is_overridden: bool = Field(
        default=False,
        description='Whether recommendation was overridden',
    )
    override_type: OverrideType = Field(
        default='no_override',
        description='Type of override',
    )
    original_recommendation: str = Field(
        default='',
        description='What AI recommended',
    )
    final_decision: Optional[str] = Field(
        default=None,
        description='What was actually decided',
    )
    override_reason: str = Field(
        default='',
        description='Reason for override',
    )
    override_reason_category: OverrideReasonCategory = Field(
        default='none',
        description='Category of override reason',
    )
    recommendation_turn: Optional[int] = Field(
        default=None,
        description='Turn where recommendation was made',
    )
    reasoning: str = Field(
        default='',
        description='Assessment explanation',
    )


class OverrideAnalyzer(BaseMetric[OverrideInput, OverrideOutput]):
    """Internal LLM-based analyzer for override detection."""

    instruction = """You are an expert at analyzing Slack conversations to detect when humans override AI recommendations.

**TASK**: Analyze the conversation to determine if the AI's recommendation was overridden.

**OVERRIDE TYPES**:
1. **no_override**: AI recommendation was followed
   - User agreed with recommendation
   - No changes were made to the decision

2. **full_override**: Completely different decision made
   - AI recommended approve, human declined (or vice versa)
   - Clear contradiction of the recommendation
   - "I don't think decline is the right answer"

3. **partial_override**: Modified recommendation
   - Core decision similar but with changes
   - "Let's approve but with conditions"
   - Rate adjustments, class code modifications

4. **pending_override**: Discussion suggests override but not confirmed
   - Questions raised about the recommendation
   - Alternative approaches discussed
   - No final decision reached yet

**OVERRIDE REASON CATEGORIES**:
- **additional_info**: Override based on info AI didn't have
- **risk_assessment**: Different risk evaluation
- **policy_exception**: Policy exception being applied
- **class_code_issue**: Class code concerns
- **rate_issue**: Rating or pricing concerns
- **experience_judgment**: Human judgment/experience differs
- **other**: Other reasons
- **none**: No override occurred

**OVERRIDE DETECTION PATTERNS**:
- "I don't think [recommendation] is the right answer"
- Discussions about alternative approaches after AI recommendation
- Manual rate adjustments mentioned
- Class code modifications discussed
- Team disagreement with AI assessment
- User stating they'll do something different

**OUTPUT**:
- is_overridden: True if recommendation was overridden (full or partial)
- override_type: One of the types above
- original_recommendation: What AI recommended (e.g., "Approve", "Decline")
- final_decision: What was actually decided (or null if pending/no override)
- override_reason: The stated reason for override (extracted from conversation)
- override_reason_category: Category from the list above
- reasoning: Brief explanation (1-2 sentences) of your assessment"""

    input_model = OverrideInput
    output_model = OverrideOutput


@metric(
    name='Override Detector',
    key='override_detector',
    description='Detects when humans override AI recommendations in Slack conversations.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.CLASSIFICATION,
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['slack', 'multi_turn', 'classification'],
)
class OverrideDetector(BaseMetric):
    """
    LLM-based metric that detects when humans override AI recommendations.

    Used for computing:
    - override_rate: Recommendations overridden / Total recommendations

    Classification categories:
    - no_override: AI recommendation followed
    - full_override: Completely different decision
    - partial_override: Modified recommendation
    - pending_override: Discussion suggests override but not confirmed

    Note: Only runs if a recommendation is found in the conversation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.override_analyzer = OverrideAnalyzer(**kwargs)

    @trace(name='OverrideDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect override of AI recommendation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=OverrideResult(),
            )

        # Find recommendation
        rec_turn = find_recommendation_turn(item.conversation)
        if rec_turn is None:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No recommendation found to analyze.',
                signals=OverrideResult(has_recommendation=False),
            )

        # Get recommendation type
        rec_message = item.conversation.messages[rec_turn]
        rec_type = extract_recommendation_type(rec_message.content or '') or 'unknown'

        # Build transcript
        transcript = build_transcript(item.conversation)

        # Prepare input for LLM
        analysis_input = OverrideInput(
            conversation_transcript=transcript,
            recommendation_type=rec_type,
            recommendation_turn=rec_turn,
        )

        # Run LLM analysis
        try:
            llm_result = await self.override_analyzer.execute(analysis_input)

            result = OverrideResult(
                has_recommendation=True,
                is_overridden=llm_result.is_overridden,
                override_type=llm_result.override_type,
                original_recommendation=llm_result.original_recommendation or rec_type,
                final_decision=llm_result.final_decision,
                override_reason=llm_result.override_reason,
                override_reason_category=llm_result.override_reason_category,
                recommendation_turn=rec_turn,
                reasoning=llm_result.reasoning,
            )

            # Score: 1.0 if overridden, 0.0 if not
            score = 1.0 if result.is_overridden else 0.0

            explanation = (
                f"Recommendation '{rec_type}': {result.override_type}. "
                f'{result.reasoning}'
            )

        except Exception as e:
            result = OverrideResult(
                has_recommendation=True,
                is_overridden=False,
                override_type='no_override',
                original_recommendation=rec_type,
                recommendation_turn=rec_turn,
                reasoning=f'LLM analysis failed: {e}',
            )
            score = 0.0
            explanation = f'Analysis failed, marked as no override: {e}'

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=result,
        )

    def get_signals(
        self, result: OverrideResult
    ) -> List[SignalDescriptor[OverrideResult]]:
        """Generate signal descriptors for override detection."""

        override_scores = {
            'no_override': 0.0,
            'full_override': 1.0,
            'partial_override': 0.5,
            'pending_override': 0.25,
        }

        reason_scores = {
            'none': 0.0,
            'additional_info': 0.8,
            'risk_assessment': 0.7,
            'policy_exception': 0.6,
            'class_code_issue': 0.7,
            'rate_issue': 0.6,
            'experience_judgment': 0.8,
            'other': 0.5,
        }

        return [
            # Headline signals
            SignalDescriptor(
                name='has_recommendation',
                extractor=lambda r: r.has_recommendation,
                description='Whether recommendation was found',
            ),
            SignalDescriptor(
                name='is_overridden',
                extractor=lambda r: r.is_overridden,
                headline_display=True,
                description='Whether recommendation was overridden',
            ),
            SignalDescriptor(
                name='override_type',
                extractor=lambda r: r.override_type,
                headline_display=True,
                score_mapping=override_scores,
                description='Type of override',
            ),
            # Decision details
            SignalDescriptor(
                name='original_recommendation',
                extractor=lambda r: r.original_recommendation,
                description='What AI recommended',
            ),
            SignalDescriptor(
                name='final_decision',
                extractor=lambda r: r.final_decision,
                description='What was actually decided',
            ),
            # Override reason
            SignalDescriptor(
                name='override_reason',
                extractor=lambda r: r.override_reason,
                description='Stated reason for override',
            ),
            SignalDescriptor(
                name='override_reason_category',
                extractor=lambda r: r.override_reason_category,
                score_mapping=reason_scores,
                description='Category of override reason',
            ),
            # Context
            SignalDescriptor(
                name='recommendation_turn',
                extractor=lambda r: r.recommendation_turn,
                description='Turn where recommendation was made',
            ),
            SignalDescriptor(
                name='reasoning',
                extractor=lambda r: r.reasoning,
                description='Assessment explanation',
            ),
        ]


class SatisfactionInput(RichBaseModel):
    """Input model for override satisfaction analysis."""

    conversation_transcript: str = Field(description='Full conversation transcript')
    override_reason: str = Field(
        description='The stated reason for the override',
    )
    override_reason_category: str = Field(
        description='Category of the override reason',
    )


class SatisfactionOutput(RichBaseModel):
    """Output model for override satisfaction analysis."""

    satisfaction_score: float = Field(
        ge=0.0,
        le=1.0,
        description='Quality score for override explanation (0-1)',
    )
    has_clear_reason: bool = Field(
        description='Whether the override has a clear, stated reason',
    )
    has_supporting_evidence: bool = Field(
        description='Whether evidence supports the override decision',
    )
    is_actionable: bool = Field(
        description='Whether the override provides actionable feedback',
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description='Suggestions for improving AI recommendations',
    )
    reasoning: str = Field(
        default='',
        description='Explanation of the satisfaction assessment',
    )


class SatisfactionResult(RichBaseModel):
    """Result model for override satisfaction analysis."""

    has_override: bool = Field(
        default=False,
        description='Whether an override was found to analyze',
    )
    satisfaction_score: float = Field(
        default=0.0,
        description='Quality score for override explanation (0-1)',
    )
    is_satisfactory: bool = Field(
        default=False,
        description='Whether override explanation meets threshold',
    )
    has_clear_reason: bool = Field(
        default=False,
        description='Has clear, stated reason',
    )
    has_supporting_evidence: bool = Field(
        default=False,
        description='Has evidence supporting decision',
    )
    is_actionable: bool = Field(
        default=False,
        description='Provides actionable feedback',
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description='Suggestions for AI improvement',
    )
    override_reason: str = Field(
        default='',
        description='The override reason analyzed',
    )
    reasoning: str = Field(
        default='',
        description='Assessment explanation',
    )


class SatisfactionAnalyzer(BaseMetric[SatisfactionInput, SatisfactionOutput]):
    """Internal LLM-based analyzer for override satisfaction."""

    instruction = """You are an expert at evaluating the quality of override explanations in underwriting workflows.

**TASK**: Analyze the quality of the override reason/explanation provided when a human overrode an AI recommendation.

**QUALITY CRITERIA**:

1. **Clear Reason** (has_clear_reason):
   - Is there a specific, understandable reason stated?
   - Not just "I disagree" but actual justification
   - Explains WHY the AI recommendation was wrong

2. **Supporting Evidence** (has_supporting_evidence):
   - Does the override cite specific information?
   - References to policies, precedents, or data
   - Concrete examples or facts that support the decision

3. **Actionable Feedback** (is_actionable):
   - Could this override help improve the AI?
   - Identifies what the AI missed or got wrong
   - Provides learnable pattern for future cases

**SCORING GUIDELINES** (satisfaction_score 0-1):
- 0.0-0.3: Vague or no reason given ("just because", "I know better")
- 0.3-0.5: Basic reason but lacks detail or evidence
- 0.5-0.7: Good reason with some supporting context
- 0.7-0.9: Clear reason with evidence and actionable insight
- 0.9-1.0: Excellent explanation with comprehensive justification

**OUTPUT**:
- satisfaction_score: 0.0 to 1.0 based on explanation quality
- has_clear_reason: True if specific reason is provided
- has_supporting_evidence: True if evidence/data supports override
- is_actionable: True if feedback could improve AI
- improvement_suggestions: List of specific suggestions for AI improvement (1-3 items)
- reasoning: Brief explanation (1-2 sentences) of your assessment"""

    input_model = SatisfactionInput
    output_model = SatisfactionOutput


@metric(
    name='Override Satisfaction Analyzer',
    key='override_satisfaction_analyzer',
    description='Scores the quality of override explanations in Slack conversations.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.SCORE,
    default_threshold=0.7,
    score_range=(0, 1),
    tags=['slack', 'llm', 'multi_turn', 'score'],
)
class OverrideSatisfactionAnalyzer(BaseMetric):
    """
    LLM-based metric that scores the quality of override explanations.

    Used for computing:
    - override_satisfaction: Positive override responses / Total overrides

    Note: Depends on OverrideDetector - only runs if an override was detected.
    """

    def __init__(self, satisfaction_threshold: float = 0.7, **kwargs):
        """
        Initialize the override satisfaction analyzer.

        Args:
            satisfaction_threshold: Score threshold for "satisfactory" classification (default: 0.7)
        """
        super().__init__(**kwargs)
        self.satisfaction_threshold = satisfaction_threshold
        self.override_detector = OverrideDetector(**kwargs)
        self.satisfaction_analyzer = SatisfactionAnalyzer(**kwargs)

    @trace(
        name='OverrideSatisfactionAnalyzer', capture_args=True, capture_response=True
    )
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Analyze satisfaction of override explanation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=SatisfactionResult(),
            )

        # First, detect if there's an override
        override_result = await self.override_detector.execute(item, **kwargs)
        override_signals = override_result.signals

        if not override_signals or not override_signals.is_overridden:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No override detected to analyze.',
                signals=SatisfactionResult(has_override=False),
            )

        # Build transcript
        transcript = build_transcript(item.conversation)

        # Prepare input for satisfaction analysis
        analysis_input = SatisfactionInput(
            conversation_transcript=transcript,
            override_reason=override_signals.override_reason or '',
            override_reason_category=override_signals.override_reason_category
            or 'none',
        )

        # Run LLM analysis
        try:
            llm_result = await self.satisfaction_analyzer.execute(analysis_input)

            is_satisfactory = (
                llm_result.satisfaction_score >= self.satisfaction_threshold
            )

            result = SatisfactionResult(
                has_override=True,
                satisfaction_score=llm_result.satisfaction_score,
                is_satisfactory=is_satisfactory,
                has_clear_reason=llm_result.has_clear_reason,
                has_supporting_evidence=llm_result.has_supporting_evidence,
                is_actionable=llm_result.is_actionable,
                improvement_suggestions=llm_result.improvement_suggestions,
                override_reason=override_signals.override_reason or '',
                reasoning=llm_result.reasoning,
            )

            explanation = (
                f'Override satisfaction: {result.satisfaction_score:.2f} '
                f'({"satisfactory" if result.is_satisfactory else "unsatisfactory"}). '
                f'{result.reasoning}'
            )

        except Exception as e:
            result = SatisfactionResult(
                has_override=True,
                satisfaction_score=0.0,
                is_satisfactory=False,
                override_reason=override_signals.override_reason or '',
                reasoning=f'LLM analysis failed: {e}',
            )
            explanation = f'Analysis failed: {e}'

        return MetricEvaluationResult(
            score=result.satisfaction_score,
            explanation=explanation,
            signals=result,
        )

    def get_signals(
        self, result: SatisfactionResult
    ) -> List[SignalDescriptor[SatisfactionResult]]:
        """Generate signal descriptors for satisfaction analysis."""

        return [
            # Headline signals
            SignalDescriptor(
                name='has_override',
                extractor=lambda r: r.has_override,
                description='Whether override was found to analyze',
            ),
            SignalDescriptor(
                name='satisfaction_score',
                extractor=lambda r: r.satisfaction_score,
                headline_display=True,
                description='Quality score for override explanation (0-1)',
            ),
            SignalDescriptor(
                name='is_satisfactory',
                extractor=lambda r: r.is_satisfactory,
                headline_display=True,
                description=f'Score >= {self.satisfaction_threshold}',
            ),
            # Quality components
            SignalDescriptor(
                name='has_clear_reason',
                extractor=lambda r: r.has_clear_reason,
                description='Has clear, stated reason',
            ),
            SignalDescriptor(
                name='has_supporting_evidence',
                extractor=lambda r: r.has_supporting_evidence,
                description='Has supporting evidence',
            ),
            SignalDescriptor(
                name='is_actionable',
                extractor=lambda r: r.is_actionable,
                description='Provides actionable feedback',
            ),
            # Details
            SignalDescriptor(
                name='improvement_suggestions',
                extractor=lambda r: '; '.join(r.improvement_suggestions)
                if r.improvement_suggestions
                else None,
                description='Suggestions for AI improvement',
            ),
            SignalDescriptor(
                name='override_reason',
                extractor=lambda r: r.override_reason,
                description='The override reason analyzed',
            ),
            SignalDescriptor(
                name='reasoning',
                extractor=lambda r: r.reasoning,
                description='Assessment explanation',
            ),
        ]
