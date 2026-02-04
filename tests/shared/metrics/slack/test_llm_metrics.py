"""Tests for Slack LLM-based metrics."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation
from eval_workbench.shared.metrics.slack.acceptance import (
    AcceptanceDetector,
    AcceptanceResult,
)
from eval_workbench.shared.metrics.slack.escalation import (
    EscalationDetector,
    EscalationResult,
)
from eval_workbench.shared.metrics.slack.frustration import (
    FrustrationDetector,
    FrustrationResult,
)
from eval_workbench.shared.metrics.slack.override import (
    OverrideDetector,
    OverrideResult,
    OverrideSatisfactionAnalyzer,
)


@pytest.fixture
def conversation_with_recommendation():
    """Create a conversation with a recommendation."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Review case MGT-BOP-1234567'),
            AIMessage(content='Recommend Approve. Base Score: 85/100'),
            HumanMessage(content='Looks good, thanks!'),
        ]
    )


@pytest.fixture
def conversation_with_escalation():
    """Create a conversation with escalation."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Help with this case'),
            AIMessage(content='I apologize, I encountered an error.'),
            HumanMessage(content='@john_doe can you help instead?'),
        ]
    )


@pytest.fixture
def frustrated_conversation():
    """Create a conversation with frustration signals."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Help me with this'),
            AIMessage(content='Here is some info.'),
            HumanMessage(content='That doesnt work. STILL WRONG. What is going on???'),
            AIMessage(content='Let me try again.'),
            HumanMessage(content='Forget it, this is frustrating!!'),
        ]
    )


@pytest.fixture
def override_conversation():
    """Create a conversation with an override."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Review this case'),
            AIMessage(content='Recommend Decline. Risk factors detected.'),
            HumanMessage(
                content="I don't think decline is the right answer. The additional docs show this is a good risk."
            ),
            AIMessage(content='Understood. Let me update.'),
        ]
    )


class TestEscalationDetector:
    """Tests for EscalationDetector metric."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return EscalationDetector()

    @pytest.mark.asyncio
    async def test_heuristic_fallback_detects_mentions(
        self, detector, conversation_with_escalation
    ):
        """Should detect escalation via mentions using heuristic."""
        item = DatasetItem(conversation=conversation_with_escalation)

        # Mock LLM to fail, triggering heuristic fallback
        detector.escalation_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        assert result.score == 1.0  # Escalation detected
        signals = result.signals
        assert signals.is_escalated is True
        assert signals.escalation_type == 'team_mention'

    @pytest.mark.asyncio
    async def test_heuristic_fallback_detects_errors(self, detector):
        """Should detect error escalation using heuristic."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Help me'),
                AIMessage(
                    content='I apologize, I encountered an error processing this.'
                ),
            ]
        )
        item = DatasetItem(conversation=conversation)
        detector.escalation_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        signals = result.signals
        assert signals.is_escalated is True
        assert signals.escalation_type == 'error_escalation'

    @pytest.mark.asyncio
    async def test_no_escalation_detected(
        self, detector, conversation_with_recommendation
    ):
        """Should detect no escalation in normal conversation."""
        item = DatasetItem(conversation=conversation_with_recommendation)
        detector.escalation_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        assert result.score == 0.0
        signals = result.signals
        assert signals.is_escalated is False

    @pytest.mark.asyncio
    async def test_handles_no_conversation(self, detector):
        """Should handle missing conversation."""
        item = DatasetItem(query='No conversation')

        result = await detector.execute(item)

        assert result.score == 0.0
        signals = result.signals
        assert signals.is_escalated is False


class TestFrustrationDetector:
    """Tests for FrustrationDetector metric."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return FrustrationDetector()

    @pytest.mark.asyncio
    async def test_heuristic_detects_frustration_patterns(
        self, detector, frustrated_conversation
    ):
        """Should detect frustration using heuristic patterns."""
        item = DatasetItem(conversation=frustrated_conversation)
        detector.frustration_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        signals = result.signals
        # Should detect: multiple question marks, exclamation marks, ALL CAPS, frustration words
        assert signals.frustration_score > 0
        assert len(signals.frustration_indicators) > 0

    @pytest.mark.asyncio
    async def test_calm_conversation_low_score(
        self, detector, conversation_with_recommendation
    ):
        """Should return low score for calm conversation."""
        item = DatasetItem(conversation=conversation_with_recommendation)
        detector.frustration_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        signals = result.signals
        assert signals.frustration_score < 0.6
        assert signals.is_frustrated is False

    @pytest.mark.asyncio
    async def test_configurable_threshold(self):
        """Should use configurable frustration threshold."""
        detector = FrustrationDetector(frustration_threshold=0.3)

        assert detector.frustration_threshold == 0.3

    @pytest.mark.asyncio
    async def test_handles_no_human_messages(self, detector):
        """Should handle conversations without human messages."""
        conversation = MultiTurnConversation(
            messages=[AIMessage(content='AI only message')]
        )
        item = DatasetItem(conversation=conversation)

        result = await detector.execute(item)

        assert result.score == 0.0


class TestAcceptanceDetector:
    """Tests for AcceptanceDetector metric."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return AcceptanceDetector()

    @pytest.mark.asyncio
    async def test_handles_no_recommendation(self, detector):
        """Should handle conversation without recommendation."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi there!'),
            ]
        )
        item = DatasetItem(conversation=conversation)

        result = await detector.execute(item)

        assert result.score == 0.0
        signals = result.signals
        assert signals.has_recommendation is False

    @pytest.mark.asyncio
    async def test_finds_recommendation_context(
        self, detector, conversation_with_recommendation
    ):
        """Should find and record recommendation context."""
        item = DatasetItem(conversation=conversation_with_recommendation)

        # Mock LLM failure to test fallback
        detector.acceptance_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        signals = result.signals
        assert signals.has_recommendation is True
        assert signals.recommendation_type == 'approve'

    @pytest.mark.asyncio
    async def test_handles_no_conversation(self, detector):
        """Should handle missing conversation."""
        item = DatasetItem(query='No conversation')

        result = await detector.execute(item)

        assert result.score == 0.0


class TestOverrideDetector:
    """Tests for OverrideDetector metric."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return OverrideDetector()

    @pytest.mark.asyncio
    async def test_handles_no_recommendation(self, detector):
        """Should handle conversation without recommendation."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi!'),
            ]
        )
        item = DatasetItem(conversation=conversation)

        result = await detector.execute(item)

        assert result.score == 0.0
        signals = result.signals
        assert signals.has_recommendation is False

    @pytest.mark.asyncio
    async def test_records_recommendation_context(
        self, detector, override_conversation
    ):
        """Should record recommendation context for override analysis."""
        item = DatasetItem(conversation=override_conversation)
        detector.override_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await detector.execute(item)

        signals = result.signals
        assert signals.has_recommendation is True
        assert signals.original_recommendation == 'decline'

    @pytest.mark.asyncio
    async def test_handles_no_conversation(self, detector):
        """Should handle missing conversation."""
        item = DatasetItem(query='No conversation')

        result = await detector.execute(item)

        assert result.score == 0.0


class TestOverrideSatisfactionAnalyzer:
    """Tests for OverrideSatisfactionAnalyzer metric."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return OverrideSatisfactionAnalyzer()

    @pytest.mark.asyncio
    async def test_skips_when_no_override(
        self, analyzer, conversation_with_recommendation
    ):
        """Should skip analysis when no override detected."""
        item = DatasetItem(conversation=conversation_with_recommendation)

        # Mock override detector to return no override
        with patch.object(
            analyzer.override_detector, 'execute', new_callable=AsyncMock
        ) as mock_override:
            mock_override.return_value = MagicMock(
                signals=OverrideResult(
                    has_recommendation=True,
                    is_overridden=False,
                    override_type='no_override',
                )
            )

            result = await analyzer.execute(item)

            assert result.score == 0.0
            signals = result.signals
            assert signals.has_override is False

    @pytest.mark.asyncio
    async def test_configurable_threshold(self):
        """Should use configurable satisfaction threshold."""
        analyzer = OverrideSatisfactionAnalyzer(satisfaction_threshold=0.5)

        assert analyzer.satisfaction_threshold == 0.5

    @pytest.mark.asyncio
    async def test_handles_no_conversation(self, analyzer):
        """Should handle missing conversation."""
        item = DatasetItem(query='No conversation')

        result = await analyzer.execute(item)

        assert result.score == 0.0


class TestSignalDescriptors:
    """Tests for signal descriptors across LLM metrics."""

    @pytest.mark.asyncio
    async def test_escalation_signals(self):
        """Should generate proper escalation signals."""
        detector = EscalationDetector()
        result = EscalationResult(
            is_escalated=True,
            escalation_type='team_mention',
            escalation_targets=['user1', 'user2'],
        )

        signals = detector.get_signals(result)

        signal_names = [s.name for s in signals]
        assert 'is_escalated' in signal_names
        assert 'escalation_type' in signal_names
        assert 'escalation_targets' in signal_names

        # Check headline signals are marked
        headline_signals = [s for s in signals if s.headline_display]
        assert len(headline_signals) >= 2

    @pytest.mark.asyncio
    async def test_frustration_signals(self):
        """Should generate proper frustration signals."""
        detector = FrustrationDetector()
        result = FrustrationResult(
            frustration_score=0.7,
            is_frustrated=True,
            frustration_cause='ai_error',
        )

        signals = detector.get_signals(result)

        signal_names = [s.name for s in signals]
        assert 'frustration_score' in signal_names
        assert 'is_frustrated' in signal_names
        assert 'frustration_cause' in signal_names

    @pytest.mark.asyncio
    async def test_acceptance_signals(self):
        """Should generate proper acceptance signals."""
        detector = AcceptanceDetector()
        result = AcceptanceResult(
            has_recommendation=True,
            acceptance_status='accepted',
            is_accepted=True,
        )

        signals = detector.get_signals(result)

        signal_names = [s.name for s in signals]
        assert 'acceptance_status' in signal_names
        assert 'is_accepted' in signal_names
        assert 'recommendation_type' in signal_names

    @pytest.mark.asyncio
    async def test_override_signals(self):
        """Should generate proper override signals."""
        detector = OverrideDetector()
        result = OverrideResult(
            has_recommendation=True,
            is_overridden=True,
            override_type='full_override',
            override_reason_category='additional_info',
        )

        signals = detector.get_signals(result)

        signal_names = [s.name for s in signals]
        assert 'is_overridden' in signal_names
        assert 'override_type' in signal_names
        assert 'override_reason_category' in signal_names
