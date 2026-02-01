from typing import List, Optional, Set

from axion._core.schema import HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import (
    count_questions,
    extract_mentions,
    get_ai_messages,
    get_human_messages,
)


class ThreadEngagementResult(RichBaseModel):
    """Result model for thread engagement analysis."""

    # Engagement depth metrics
    interaction_depth: int = Field(
        default=0,
        description='Number of back-and-forth exchanges (human-AI pairs)',
    )
    has_multiple_interactions: bool = Field(
        default=False,
        description='Whether thread has more than one human message',
    )

    # Message quality metrics
    avg_human_response_length: float = Field(
        default=0.0,
        description='Average length of human messages in characters',
    )
    avg_ai_response_length: float = Field(
        default=0.0,
        description='Average length of AI messages in characters',
    )

    # Interaction signals
    question_count: int = Field(
        default=0,
        description='Total number of questions asked in thread',
    )
    mention_count: int = Field(
        default=0,
        description='Total number of @mentions in thread',
    )
    unique_participants: int = Field(
        default=0,
        description='Count of unique human participants',
    )

    # Engagement indicators
    total_human_messages: int = Field(
        default=0,
        description='Total human messages for engagement calculation',
    )
    total_ai_messages: int = Field(
        default=0,
        description='Total AI messages for engagement calculation',
    )


@metric(
    name='Thread Engagement Analyzer',
    key='thread_engagement_analyzer',
    description='Measures engagement depth within Slack threads for KPI aggregation.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'heuristic', 'multi_turn', 'analysis'],
)
class ThreadEngagementAnalyzer(BaseMetric):
    """
    Heuristic metric that measures engagement depth in Slack conversations.

    Used for computing:
    - engagement_rate: Avg interactions per case or % with multiple interactions

    This is an ANALYSIS metric - it produces structured signals without a numeric score.
    """

    @trace(name='ThreadEngagementAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Analyze engagement depth in Slack conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=ThreadEngagementResult(),
            )

        # Get messages by type
        ai_messages = get_ai_messages(item.conversation)
        human_messages = get_human_messages(item.conversation)

        # Calculate interaction depth (number of exchanges)
        # An exchange is a human message followed by an AI response or vice versa
        interaction_depth = self._calculate_interaction_depth(
            item.conversation.messages
        )

        # Check for multiple interactions
        has_multiple_interactions = len(human_messages) > 1

        # Calculate average response lengths
        avg_human_length = 0.0
        if human_messages:
            total_human_len = sum(len(msg.content or '') for msg in human_messages)
            avg_human_length = total_human_len / len(human_messages)

        avg_ai_length = 0.0
        if ai_messages:
            total_ai_len = sum(len(msg.content or '') for msg in ai_messages)
            avg_ai_length = total_ai_len / len(ai_messages)

        # Count questions across all messages
        total_questions = 0
        for msg in item.conversation.messages:
            if msg.content:
                total_questions += count_questions(msg.content)

        # Count mentions across all messages
        total_mentions = 0
        for msg in item.conversation.messages:
            if msg.content:
                total_mentions += len(extract_mentions(msg.content))

        # Count unique participants
        # Use sender metadata if available, otherwise try to extract from messages
        unique_participants = self._count_unique_participants(
            item.conversation.messages, item.additional_input
        )

        result = ThreadEngagementResult(
            interaction_depth=interaction_depth,
            has_multiple_interactions=has_multiple_interactions,
            avg_human_response_length=round(avg_human_length, 1),
            avg_ai_response_length=round(avg_ai_length, 1),
            question_count=total_questions,
            mention_count=total_mentions,
            unique_participants=unique_participants,
            total_human_messages=len(human_messages),
            total_ai_messages=len(ai_messages),
        )

        explanation = (
            f'Thread engagement: {interaction_depth} exchanges, '
            f'{len(human_messages)} human messages, '
            f'{total_questions} questions, '
            f'{unique_participants} unique participants.'
        )

        return MetricEvaluationResult(
            score=None,
            explanation=explanation,
            signals=result,
        )

    def _calculate_interaction_depth(self, messages: list) -> int:
        """
        Calculate the number of back-and-forth exchanges.

        An exchange is counted when there's a transition between
        human and AI messages.
        """
        if not messages:
            return 0

        exchanges = 0
        last_role = None

        for msg in messages:
            current_role = 'human' if isinstance(msg, HumanMessage) else 'ai'

            if last_role is not None and current_role != last_role:
                exchanges += 1

            last_role = current_role

        # Convert transitions to exchanges (pairs)
        return (exchanges + 1) // 2 if exchanges > 0 else 0

    def _count_unique_participants(
        self, messages: list, additional_input: Optional[dict]
    ) -> int:
        """
        Count unique human participants in the thread.

        Tries to extract participant info from message metadata
        or additional_input.
        """
        participants: Set[str] = set()

        # Try to get participant list from additional_input
        if additional_input:
            if 'participants' in additional_input:
                participants.update(additional_input['participants'])
            if 'sender' in additional_input:
                participants.add(additional_input['sender'])

        # If no metadata, count human messages as a proxy
        # (assumes each distinct human message could be from different person)
        # This is a heuristic when we don't have actual user IDs
        if not participants:
            # Check for mentions in human messages as proxy for participants
            for msg in messages:
                if isinstance(msg, HumanMessage) and msg.content:
                    mentions = extract_mentions(msg.content)
                    participants.update(mentions)

            # At minimum, we have 1 participant if there are human messages
            human_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
            if human_count > 0 and not participants:
                return 1  # At least one human participant

        return len(participants) if participants else 0

    def get_signals(
        self, result: ThreadEngagementResult
    ) -> List[SignalDescriptor[ThreadEngagementResult]]:
        """Generate signal descriptors for the engagement analysis."""

        return [
            # Headline signals
            SignalDescriptor(
                name='interaction_depth',
                extractor=lambda r: r.interaction_depth,
                headline_display=True,
                description='Number of back-and-forth exchanges',
            ),
            SignalDescriptor(
                name='has_multiple_interactions',
                extractor=lambda r: r.has_multiple_interactions,
                headline_display=True,
                description='Whether thread has >1 human message',
            ),
            # Message counts
            SignalDescriptor(
                name='total_human_messages',
                extractor=lambda r: r.total_human_messages,
                description='Total human messages',
            ),
            SignalDescriptor(
                name='total_ai_messages',
                extractor=lambda r: r.total_ai_messages,
                description='Total AI messages',
            ),
            # Quality metrics
            SignalDescriptor(
                name='avg_human_response_length',
                extractor=lambda r: r.avg_human_response_length,
                description='Average human message length (chars)',
            ),
            SignalDescriptor(
                name='avg_ai_response_length',
                extractor=lambda r: r.avg_ai_response_length,
                description='Average AI message length (chars)',
            ),
            # Interaction signals
            SignalDescriptor(
                name='question_count',
                extractor=lambda r: r.question_count,
                description='Total questions in thread',
            ),
            SignalDescriptor(
                name='mention_count',
                extractor=lambda r: r.mention_count,
                description='Total @mentions in thread',
            ),
            SignalDescriptor(
                name='unique_participants',
                extractor=lambda r: r.unique_participants,
                description='Unique human participants',
            ),
        ]
