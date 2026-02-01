from typing import List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import (
    extract_case_id,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
)

RecommendationType = Literal['approve', 'decline', 'review', 'hold', 'none']


class RecommendationResult(RichBaseModel):
    """Result model for recommendation analysis."""

    # Core recommendation signals
    has_recommendation: bool = Field(
        default=False,
        description='Whether AI made a recommendation',
    )
    recommendation_type: RecommendationType = Field(
        default='none',
        description='Type of recommendation: approve, decline, review, hold, or none',
    )
    recommendation_text: Optional[str] = Field(
        default=None,
        description='Full text of the recommendation message',
    )
    recommendation_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where recommendation was made (0-based)',
    )

    # Confidence and metadata
    recommendation_confidence: Optional[float] = Field(
        default=None,
        description='Extracted confidence level if present (0-1)',
    )

    # Case information
    case_id: Optional[str] = Field(
        default=None,
        description='Extracted case identifier (e.g., MGT-BOP-123456)',
    )
    case_priority: Optional[int] = Field(
        default=None,
        description='Extracted priority/base score (0-100)',
    )


@metric(
    name='Recommendation Analyzer',
    key='recommendation_analyzer',
    description='Extracts AI recommendations from Slack conversations for KPI aggregation.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'heuristic', 'multi_turn', 'analysis'],
)
class RecommendationAnalyzer(BaseMetric):
    """
    Heuristic metric that extracts AI recommendations from Slack conversations.

    Used for computing:
    - acceptance_rate: Recommendations accepted / Total recommendations
    - override_rate: Recommendations overridden / Total recommendations

    This is an ANALYSIS metric - it produces structured signals without a numeric score.
    """

    @trace(name='RecommendationAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Extract recommendation signals from Slack conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=RecommendationResult(),
            )

        # Find recommendation turn
        rec_turn = find_recommendation_turn(item.conversation)

        if rec_turn is None:
            # No recommendation found
            result = RecommendationResult(
                has_recommendation=False,
                recommendation_type='none',
            )
            return MetricEvaluationResult(
                score=None,
                explanation='No AI recommendation found in conversation.',
                signals=result,
            )

        # Get the recommendation message
        rec_message = item.conversation.messages[rec_turn]
        rec_text = rec_message.content if rec_message.content else ''

        # Extract recommendation details
        rec_type = extract_recommendation_type(rec_text) or 'none'
        case_id = self._extract_case_id_from_conversation(item.conversation)
        priority = self._extract_priority_from_conversation(item.conversation)
        confidence = self._extract_confidence(rec_text)

        result = RecommendationResult(
            has_recommendation=True,
            recommendation_type=rec_type,
            recommendation_text=rec_text[:500]
            if rec_text
            else None,  # Truncate for storage
            recommendation_turn_index=rec_turn,
            recommendation_confidence=confidence,
            case_id=case_id,
            case_priority=priority,
        )

        explanation = (
            f"Found recommendation: '{rec_type}' at turn {rec_turn}. "
            f'{f"Case: {case_id}. " if case_id else ""}'
            f'{f"Priority: {priority}/100. " if priority else ""}'
        )

        return MetricEvaluationResult(
            score=None,
            explanation=explanation,
            signals=result,
        )

    def _extract_case_id_from_conversation(self, conversation) -> Optional[str]:
        """Extract case ID from any message in the conversation."""
        for msg in conversation.messages:
            if msg.content:
                case_id = extract_case_id(msg.content)
                if case_id:
                    return case_id
        return None

    def _extract_priority_from_conversation(self, conversation) -> Optional[int]:
        """Extract priority score from AI messages."""
        ai_messages = get_ai_messages(conversation)
        for msg in ai_messages:
            if msg.content:
                priority = extract_priority_score(msg.content)
                if priority is not None:
                    return priority
        return None

    def _extract_confidence(self, text: str) -> Optional[float]:
        """
        Extract confidence level from recommendation text.

        Looks for patterns like:
        - "confidence: 0.85" or "confidence: 85%"
        - "high confidence" / "medium confidence" / "low confidence"
        """
        import re

        if not text:
            return None

        # Numeric confidence
        match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%?', text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Normalize to 0-1 if percentage
            if value > 1:
                value = value / 100
            return min(max(value, 0), 1)

        # Categorical confidence
        text_lower = text.lower()
        if 'high confidence' in text_lower:
            return 0.9
        if 'medium confidence' in text_lower:
            return 0.6
        if 'low confidence' in text_lower:
            return 0.3

        return None

    def get_signals(
        self, result: RecommendationResult
    ) -> List[SignalDescriptor[RecommendationResult]]:
        """Generate signal descriptors for the recommendation analysis."""

        return [
            # Headline signals
            SignalDescriptor(
                name='has_recommendation',
                extractor=lambda r: r.has_recommendation,
                headline_display=True,
                description='Whether AI made a recommendation',
            ),
            SignalDescriptor(
                name='recommendation_type',
                extractor=lambda r: r.recommendation_type,
                headline_display=True,
                description='Type of recommendation',
                score_mapping={
                    'approve': 1.0,
                    'decline': 0.0,
                    'review': 0.5,
                    'hold': 0.5,
                },
            ),
            # Position and content
            SignalDescriptor(
                name='recommendation_turn_index',
                extractor=lambda r: r.recommendation_turn_index,
                description='Turn where recommendation was made',
            ),
            SignalDescriptor(
                name='recommendation_confidence',
                extractor=lambda r: r.recommendation_confidence,
                description='Extracted confidence level',
            ),
            # Case metadata
            SignalDescriptor(
                name='case_id',
                extractor=lambda r: r.case_id,
                description='Case identifier',
            ),
            SignalDescriptor(
                name='case_priority',
                extractor=lambda r: r.case_priority,
                description='Priority/base score (0-100)',
            ),
            # Full text (for debugging)
            SignalDescriptor(
                name='recommendation_text',
                extractor=lambda r: r.recommendation_text,
                description='Full recommendation text (truncated)',
            ),
        ]
