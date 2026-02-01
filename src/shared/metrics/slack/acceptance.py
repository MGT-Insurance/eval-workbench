from typing import List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import (
    build_transcript,
    extract_recommendation_type,
    find_recommendation_turn,
)

AcceptanceStatus = Literal[
    'accepted',
    'accepted_with_discussion',
    'pending',
    'rejected',
    'modified',
]


class AcceptanceInput(RichBaseModel):
    """Input model for acceptance detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    recommendation_type: str = Field(
        description='Type of recommendation made (approve, decline, review, hold)',
    )
    recommendation_turn: int = Field(
        description='Turn index where recommendation was made',
    )


class AcceptanceOutput(RichBaseModel):
    """Output model for acceptance detection."""

    acceptance_status: AcceptanceStatus = Field(
        description='The acceptance status of the recommendation',
    )
    acceptance_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where acceptance/rejection occurred (0-based)',
    )
    decision_maker: Optional[str] = Field(
        default=None,
        description='Who made the final decision (if identifiable)',
    )
    turns_to_decision: Optional[int] = Field(
        default=None,
        description='Number of turns between recommendation and decision',
    )
    reasoning: str = Field(
        default='',
        description='Explanation of the acceptance assessment',
    )


class AcceptanceResult(RichBaseModel):
    """Result model for acceptance analysis."""

    has_recommendation: bool = Field(
        default=False,
        description='Whether a recommendation was found to analyze',
    )
    acceptance_status: AcceptanceStatus = Field(
        default='pending',
        description='Acceptance status of the recommendation',
    )
    is_accepted: bool = Field(
        default=False,
        description='Whether recommendation was accepted (accepted or accepted_with_discussion)',
    )
    acceptance_turn_index: Optional[int] = Field(
        default=None,
        description='Turn where acceptance occurred',
    )
    decision_maker: Optional[str] = Field(
        default=None,
        description='Who made the final decision',
    )
    turns_to_decision: Optional[int] = Field(
        default=None,
        description='Turns between recommendation and decision',
    )
    recommendation_type: Optional[str] = Field(
        default=None,
        description='Type of recommendation analyzed',
    )
    recommendation_turn: Optional[int] = Field(
        default=None,
        description='Turn where recommendation was made',
    )
    reasoning: str = Field(
        default='',
        description='Assessment explanation',
    )


class AcceptanceAnalyzer(BaseMetric[AcceptanceInput, AcceptanceOutput]):
    """Internal LLM-based analyzer for acceptance detection."""

    instruction = """You are an expert at analyzing Slack conversations to determine if AI recommendations were accepted.

**TASK**: Analyze the conversation to determine if the AI's recommendation was accepted.

**ACCEPTANCE STATUSES**:
1. **accepted**: Recommendation was accepted without significant change
   - User explicitly agrees: "Sounds good", "Let's do that", "Approve it"
   - Action taken matching recommendation
   - No objections raised

2. **accepted_with_discussion**: Recommendation accepted after team discussion
   - Initial questions or concerns were raised
   - Discussion occurred but ultimately agreed with recommendation
   - Minor clarifications but core recommendation followed

3. **pending**: No clear resolution in the thread
   - Discussion is ongoing without conclusion
   - User said they'll review/think about it
   - Thread ends without clear decision

4. **rejected**: Recommendation was explicitly rejected
   - Clear disagreement: "No", "I don't think so", "That's not right"
   - User states they won't follow the recommendation
   - Action taken contrary to recommendation

5. **modified**: Recommendation accepted with modifications
   - User agrees in principle but changes details
   - "Yes, but..." responses
   - Partial acceptance with adjustments

**ACCEPTANCE INDICATORS**:
- Explicit agreement phrases
- Questions being asked (suggests not immediately accepted)
- Counter-proposals (suggests modification or rejection)
- Action statements matching or contradicting recommendation
- Discussion of alternatives

**OUTPUT**:
- acceptance_status: One of the statuses above
- acceptance_turn_index: Turn number (0-indexed) where acceptance/rejection occurred (or null if pending)
- decision_maker: Name/identifier of who made the decision (if identifiable)
- turns_to_decision: Number of turns from recommendation to decision (or null if pending)
- reasoning: Brief explanation (1-2 sentences) of your assessment"""

    input_model = AcceptanceInput
    output_model = AcceptanceOutput


@metric(
    name='Acceptance Detector',
    key='acceptance_detector',
    description='Determines if AI recommendations in Slack conversations were accepted.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.CLASSIFICATION,
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['slack', 'multi_turn', 'classification'],
)
class AcceptanceDetector(BaseMetric):
    """
    LLM-based metric that determines if AI recommendations were accepted.

    Used for computing:
    - acceptance_rate: Recommendations accepted / Total recommendations

    Classification categories:
    - accepted: Recommendation accepted without change
    - accepted_with_discussion: Accepted after team discussion
    - pending: No clear resolution in thread
    - rejected: Explicitly rejected
    - modified: Accepted with modifications

    Note: Only runs if a recommendation is found in the conversation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.acceptance_analyzer = AcceptanceAnalyzer(**kwargs)

    @trace(name='AcceptanceDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect acceptance status of AI recommendation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=AcceptanceResult(),
            )

        # Find recommendation
        rec_turn = find_recommendation_turn(item.conversation)
        if rec_turn is None:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No recommendation found to analyze.',
                signals=AcceptanceResult(has_recommendation=False),
            )

        # Get recommendation type
        rec_message = item.conversation.messages[rec_turn]
        rec_type = extract_recommendation_type(rec_message.content or '') or 'unknown'

        # Build transcript
        transcript = build_transcript(item.conversation)

        # Prepare input for LLM
        analysis_input = AcceptanceInput(
            conversation_transcript=transcript,
            recommendation_type=rec_type,
            recommendation_turn=rec_turn,
        )

        # Run LLM analysis
        try:
            llm_result = await self.acceptance_analyzer.execute(analysis_input)

            # Determine if accepted
            is_accepted = llm_result.acceptance_status in [
                'accepted',
                'accepted_with_discussion',
            ]

            result = AcceptanceResult(
                has_recommendation=True,
                acceptance_status=llm_result.acceptance_status,
                is_accepted=is_accepted,
                acceptance_turn_index=llm_result.acceptance_turn_index,
                decision_maker=llm_result.decision_maker,
                turns_to_decision=llm_result.turns_to_decision,
                recommendation_type=rec_type,
                recommendation_turn=rec_turn,
                reasoning=llm_result.reasoning,
            )

            # Score mapping
            status_scores = {
                'accepted': 1.0,
                'accepted_with_discussion': 0.8,
                'modified': 0.5,
                'pending': 0.0,  # Neither accepted nor rejected
                'rejected': 0.0,
            }
            score = status_scores.get(result.acceptance_status, 0.0)

            explanation = (
                f"Recommendation '{rec_type}' at turn {rec_turn}: {result.acceptance_status}. "
                f'{result.reasoning}'
            )

        except Exception as e:
            # Fallback to pending if LLM fails
            result = AcceptanceResult(
                has_recommendation=True,
                acceptance_status='pending',
                is_accepted=False,
                recommendation_type=rec_type,
                recommendation_turn=rec_turn,
                reasoning=f'LLM analysis failed: {e}',
            )
            score = 0.0
            explanation = f'Analysis failed, marked as pending: {e}'

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=result,
        )

    def get_signals(
        self, result: AcceptanceResult
    ) -> List[SignalDescriptor[AcceptanceResult]]:
        """Generate signal descriptors for acceptance detection."""

        status_scores = {
            'accepted': 1.0,
            'accepted_with_discussion': 0.8,
            'modified': 0.5,
            'pending': 0.0,
            'rejected': 0.0,
        }

        return [
            # Headline signals
            SignalDescriptor(
                name='has_recommendation',
                extractor=lambda r: r.has_recommendation,
                description='Whether recommendation was found',
            ),
            SignalDescriptor(
                name='acceptance_status',
                extractor=lambda r: r.acceptance_status,
                headline_display=True,
                score_mapping=status_scores,
                description='Acceptance status of recommendation',
            ),
            SignalDescriptor(
                name='is_accepted',
                extractor=lambda r: r.is_accepted,
                headline_display=True,
                description='Whether recommendation was accepted',
            ),
            # Recommendation context
            SignalDescriptor(
                name='recommendation_type',
                extractor=lambda r: r.recommendation_type,
                description='Type of recommendation made',
            ),
            SignalDescriptor(
                name='recommendation_turn',
                extractor=lambda r: r.recommendation_turn,
                description='Turn where recommendation was made',
            ),
            # Decision details
            SignalDescriptor(
                name='acceptance_turn_index',
                extractor=lambda r: r.acceptance_turn_index,
                description='Turn where acceptance occurred',
            ),
            SignalDescriptor(
                name='decision_maker',
                extractor=lambda r: r.decision_maker,
                description='Who made the final decision',
            ),
            SignalDescriptor(
                name='turns_to_decision',
                extractor=lambda r: r.turns_to_decision,
                description='Turns between recommendation and decision',
            ),
            SignalDescriptor(
                name='reasoning',
                extractor=lambda r: r.reasoning,
                description='Assessment explanation',
            ),
        ]
