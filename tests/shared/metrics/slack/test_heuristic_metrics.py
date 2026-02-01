"""Tests for Slack heuristic metrics."""

import pytest
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation

from shared.metrics.slack.engagement import (
    ThreadEngagementAnalyzer,
)
from shared.metrics.slack.interaction import (
    SlackInteractionAnalyzer,
)
from shared.metrics.slack.recommendation import (
    RecommendationAnalyzer,
)


@pytest.fixture
def simple_conversation():
    """Create a simple conversation fixture."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Hello, I need help with a case.'),
            AIMessage(
                content='Hi! I can help you with that. What case are you looking at?'
            ),
            HumanMessage(content='Case MGT-BOP-1234567'),
            AIMessage(content='Recommend Approve. Base Score: 85/100'),
        ]
    )


@pytest.fixture
def ai_initiated_conversation():
    """Create an AI-initiated conversation."""
    return MultiTurnConversation(
        messages=[
            AIMessage(content='New case alert: MGT-BOP-9999999. Recommend Review.'),
            HumanMessage(content='Thanks, I will look at it.'),
            AIMessage(content='Let me know if you need more details.'),
        ]
    )


@pytest.fixture
def conversation_with_mentions():
    """Create a conversation with @mentions."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Hello, I need help.'),
            AIMessage(content='I can assist you.'),
            HumanMessage(content='@john_doe can you also take a look? <@U12345>'),
            AIMessage(content='I have notified the team.'),
        ]
    )


class TestSlackInteractionAnalyzer:
    """Tests for SlackInteractionAnalyzer metric."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SlackInteractionAnalyzer()

    @pytest.mark.asyncio
    async def test_analyzes_simple_conversation(self, analyzer, simple_conversation):
        """Should extract interaction signals from conversation."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        assert result.score is None  # ANALYSIS metric
        assert result.signals is not None
        signals = result.signals
        assert signals.ai_message_count == 2
        assert signals.human_message_count == 2
        assert signals.total_turn_count == 4
        assert signals.is_interactive is True

    @pytest.mark.asyncio
    async def test_detects_ai_initiated(self, analyzer, ai_initiated_conversation):
        """Should detect AI-initiated conversations."""
        item = DatasetItem(conversation=ai_initiated_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.is_ai_initiated is True
        assert signals.has_human_response is True

    @pytest.mark.asyncio
    async def test_extracts_metadata(self, analyzer, simple_conversation):
        """Should extract Slack metadata from additional_input."""
        item = DatasetItem(
            conversation=simple_conversation,
            additional_input={
                'thread_ts': '123.456',
                'channel_id': 'C123',
                'reply_count': 3,
                'sender': 'user123',
            },
        )

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.thread_id == '123.456'
        assert signals.channel_id == 'C123'
        assert signals.reply_count == 3
        assert signals.sender == 'user123'

    @pytest.mark.asyncio
    async def test_handles_no_conversation(self, analyzer):
        """Should handle missing conversation gracefully."""
        item = DatasetItem(query='No conversation')

        result = await analyzer.execute(item)

        assert result.signals.is_interactive is False
        assert result.signals.total_turn_count == 0

    @pytest.mark.asyncio
    async def test_signals_generation(self, analyzer, simple_conversation):
        """Should generate proper signal descriptors."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)
        signals = analyzer.get_signals(result.signals)

        signal_names = [s.name for s in signals]
        assert 'is_interactive' in signal_names
        assert 'total_turn_count' in signal_names
        assert 'ai_message_count' in signal_names
        assert 'human_message_count' in signal_names


class TestThreadEngagementAnalyzer:
    """Tests for ThreadEngagementAnalyzer metric."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ThreadEngagementAnalyzer()

    @pytest.mark.asyncio
    async def test_calculates_interaction_depth(self, analyzer, simple_conversation):
        """Should calculate interaction depth correctly."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        # Human -> AI -> Human -> AI = 2 exchanges
        assert signals.interaction_depth >= 1
        assert signals.has_multiple_interactions is True  # 2 human messages

    @pytest.mark.asyncio
    async def test_calculates_response_lengths(self, analyzer, simple_conversation):
        """Should calculate average response lengths."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.avg_human_response_length > 0
        assert signals.avg_ai_response_length > 0

    @pytest.mark.asyncio
    async def test_counts_questions(self, analyzer):
        """Should count questions in conversation."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='What is this? How do I proceed?'),
                AIMessage(content='Let me explain.'),
                HumanMessage(content='Can you clarify that?'),
            ]
        )
        item = DatasetItem(conversation=conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.question_count >= 3

    @pytest.mark.asyncio
    async def test_counts_mentions(self, analyzer, conversation_with_mentions):
        """Should count @mentions in thread."""
        item = DatasetItem(conversation=conversation_with_mentions)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.mention_count >= 2  # john_doe and U12345

    @pytest.mark.asyncio
    async def test_handles_single_message(self, analyzer):
        """Should handle single-message conversations."""
        conversation = MultiTurnConversation(messages=[HumanMessage(content='Hello')])
        item = DatasetItem(conversation=conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.has_multiple_interactions is False
        assert signals.interaction_depth == 0


class TestRecommendationAnalyzer:
    """Tests for RecommendationAnalyzer metric."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return RecommendationAnalyzer()

    @pytest.mark.asyncio
    async def test_detects_approve_recommendation(self, analyzer, simple_conversation):
        """Should detect approve recommendation."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.has_recommendation is True
        assert signals.recommendation_type == 'approve'

    @pytest.mark.asyncio
    async def test_extracts_case_id(self, analyzer, simple_conversation):
        """Should extract case ID from conversation."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.case_id == 'MGT-BOP-1234567'

    @pytest.mark.asyncio
    async def test_extracts_priority_score(self, analyzer, simple_conversation):
        """Should extract priority score from conversation."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.case_priority == 85

    @pytest.mark.asyncio
    async def test_detects_decline_recommendation(self, analyzer):
        """Should detect decline recommendation."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Review this case.'),
                AIMessage(content='Recommend Decline. Risk factors present.'),
            ]
        )
        item = DatasetItem(conversation=conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.has_recommendation is True
        assert signals.recommendation_type == 'decline'

    @pytest.mark.asyncio
    async def test_handles_no_recommendation(self, analyzer):
        """Should handle conversations without recommendations."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi, how can I help?'),
            ]
        )
        item = DatasetItem(conversation=conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.has_recommendation is False
        assert signals.recommendation_type == 'none'

    @pytest.mark.asyncio
    async def test_records_recommendation_turn(self, analyzer, simple_conversation):
        """Should record which turn contains the recommendation."""
        item = DatasetItem(conversation=simple_conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.recommendation_turn_index == 3  # Last AI message

    @pytest.mark.asyncio
    async def test_extracts_confidence(self, analyzer):
        """Should extract confidence level if present."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Review this.'),
                AIMessage(content='Recommend Approve. Confidence: 0.85'),
            ]
        )
        item = DatasetItem(conversation=conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.recommendation_confidence == 0.85
