from typing import List, Optional

from axion._core.schema import AIMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import (
    get_ai_messages,
    get_human_messages,
    parse_slack_metadata,
)


class SlackInteractionResult(RichBaseModel):
    """Result model for Slack interaction analysis."""

    # Message counts
    ai_message_count: int = Field(
        default=0, description='Number of AI messages in the thread'
    )
    human_message_count: int = Field(
        default=0, description='Number of human messages in the thread'
    )
    total_turn_count: int = Field(
        default=0, description='Total number of messages in the thread'
    )

    # Thread metadata
    reply_count: Optional[int] = Field(
        default=None, description='Number of replies from Slack metadata'
    )
    thread_id: Optional[str] = Field(
        default=None, description='Thread timestamp identifier'
    )
    channel_id: Optional[str] = Field(
        default=None, description='Slack channel identifier'
    )
    sender: Optional[str] = Field(default=None, description='Original message sender')

    # Interaction signals
    is_ai_initiated: bool = Field(
        default=False, description='Whether thread started with AI message'
    )
    has_human_response: bool = Field(
        default=False, description='Whether humans responded to AI'
    )
    has_ai_response: bool = Field(
        default=False, description='Whether AI responded in thread'
    )

    # Derived signals for aggregation
    is_interactive: bool = Field(
        default=False,
        description='Whether this qualifies as an AI interaction (has both AI and human messages)',
    )


@metric(
    name='Slack Interaction Analyzer',
    key='slack_interaction_analyzer',
    description='Extracts interaction signals from Slack threads for KPI aggregation.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'heuristic', 'multi_turn', 'analysis'],
)
class SlackInteractionAnalyzer(BaseMetric):
    """
    Heuristic metric that extracts interaction signals from Slack conversations.

    Used for computing:
    - interaction_rate: AI interactions / Eligible cases
    - MAU: Unique users (via sender field)

    This is an ANALYSIS metric - it produces structured signals without a numeric score.
    """

    @trace(name='SlackInteractionAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Extract interaction signals from Slack conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=SlackInteractionResult(),
            )

        # Parse Slack metadata from additional_input
        metadata = parse_slack_metadata(item.additional_input)

        # Count messages by type
        ai_messages = get_ai_messages(item.conversation)
        human_messages = get_human_messages(item.conversation)

        ai_count = len(ai_messages)
        human_count = len(human_messages)
        total_count = len(item.conversation.messages)

        # Determine if AI initiated (first message is AI)
        is_ai_initiated = False
        if item.conversation.messages:
            first_msg = item.conversation.messages[0]
            is_ai_initiated = isinstance(first_msg, AIMessage)

        # Check for responses
        has_human_response = human_count > 0 and ai_count > 0
        has_ai_response = ai_count > 0

        # A thread is "interactive" if it has both AI and human participation
        is_interactive = ai_count > 0 and human_count > 0

        result = SlackInteractionResult(
            ai_message_count=ai_count,
            human_message_count=human_count,
            total_turn_count=total_count,
            reply_count=metadata.reply_count,
            thread_id=metadata.thread_ts,
            channel_id=metadata.channel_id,
            sender=metadata.sender,
            is_ai_initiated=is_ai_initiated,
            has_human_response=has_human_response,
            has_ai_response=has_ai_response,
            is_interactive=is_interactive,
        )

        explanation = (
            f'Thread with {total_count} messages ({ai_count} AI, {human_count} human). '
            f'{"AI-initiated. " if is_ai_initiated else ""}'
            f'{"Interactive thread." if is_interactive else "Not interactive."}'
        )

        return MetricEvaluationResult(
            score=None,
            explanation=explanation,
            signals=result,
        )

    def get_signals(
        self, result: SlackInteractionResult
    ) -> List[SignalDescriptor[SlackInteractionResult]]:
        """Generate signal descriptors for the interaction analysis."""

        return [
            # Headline signals
            SignalDescriptor(
                name='is_interactive',
                extractor=lambda r: r.is_interactive,
                headline_display=True,
                description='Whether thread has both AI and human participation',
            ),
            SignalDescriptor(
                name='total_turn_count',
                extractor=lambda r: r.total_turn_count,
                headline_display=True,
                description='Total messages in thread',
            ),
            # Message counts
            SignalDescriptor(
                name='ai_message_count',
                extractor=lambda r: r.ai_message_count,
                description='Number of AI messages',
            ),
            SignalDescriptor(
                name='human_message_count',
                extractor=lambda r: r.human_message_count,
                description='Number of human messages',
            ),
            # Thread properties
            SignalDescriptor(
                name='is_ai_initiated',
                extractor=lambda r: r.is_ai_initiated,
                description='Thread started by AI',
            ),
            SignalDescriptor(
                name='has_human_response',
                extractor=lambda r: r.has_human_response,
                description='Humans responded to AI',
            ),
            SignalDescriptor(
                name='has_ai_response',
                extractor=lambda r: r.has_ai_response,
                description='AI responded in thread',
            ),
            # Metadata
            SignalDescriptor(
                name='thread_id',
                extractor=lambda r: r.thread_id,
                description='Slack thread timestamp',
            ),
            SignalDescriptor(
                name='channel_id',
                extractor=lambda r: r.channel_id,
                description='Slack channel ID',
            ),
            SignalDescriptor(
                name='sender',
                extractor=lambda r: r.sender,
                description='Original sender (for MAU)',
            ),
            SignalDescriptor(
                name='reply_count',
                extractor=lambda r: r.reply_count,
                description='Reply count from Slack metadata',
            ),
        ]
