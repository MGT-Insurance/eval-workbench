"""
Test script for Feedback Metrics (Intervention, Sentiment, Resolution).

These metrics are in shared/metrics/slack/ alongside other Slack metrics.

Run with: pytest tests/shared/metrics/slack/test_feedback_metrics.py -v

Note: Tests requiring LLM calls are skipped when OPENAI_API_KEY is not set
or is set to a test placeholder value.
"""

import os

import pytest
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation

from shared.metrics.slack import (
    InterventionDetector,
    ResolutionDetector,
    SentimentDetector,
)

# Skip LLM-dependent tests in CI where API keys are fake
_api_key = os.environ.get('OPENAI_API_KEY', '')
_has_real_api_key = _api_key and not _api_key.startswith('test-')
requires_llm = pytest.mark.skipif(
    not _has_real_api_key,
    reason='Requires real OPENAI_API_KEY (not a test placeholder)',
)

# =============================================================================
# Sample Data
# =============================================================================


@pytest.fixture
def factual_correction_item():
    """Conversation with factual correction (hard escalation)."""
    return DatasetItem(
        id='test-factual-correction',
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content='**Case: MGT-BOP-12345**\nRecommendation: **Decline**\nRoof age: 25 years'
                ),
                HumanMessage(
                    content='This is incorrect. The roof was replaced in 2022.'
                ),
                AIMessage(
                    content='I apologize. Let me recalculate with the updated information.'
                ),
                HumanMessage(content='Approved. Thanks for the correction.'),
            ]
        ),
    )


@pytest.fixture
def stp_item():
    """Conversation with no human interaction (STP)."""
    return DatasetItem(
        id='test-stp',
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content='**Case: MGT-BOP-67890**\nRecommendation: **Approve**\nBase Score: 85/100'
                ),
            ]
        ),
    )


@pytest.fixture
def frustrated_user_item():
    """Conversation with frustrated user."""
    return DatasetItem(
        id='test-frustrated',
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content='**Case: MGT-BOP-11111**\nRecommendation: **Review**'
                ),
                HumanMessage(
                    content="I already submitted this TWICE!! Why isn't it showing???"
                ),
                AIMessage(content='I understand your frustration. Let me check.'),
                HumanMessage(
                    content="This is ridiculous. The system STILL doesn't work!"
                ),
            ]
        ),
    )


@pytest.fixture
def approval_required_item():
    """Conversation requiring authority approval."""
    return DatasetItem(
        id='test-authority',
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content='**Large Account** - Requires senior underwriter approval.'
                ),
                HumanMessage(content="As senior UW, I'm approving this account."),
                AIMessage(content='Thank you for the approval.'),
            ]
        ),
    )


@pytest.fixture
def declined_item():
    """Conversation ending in decline."""
    return DatasetItem(
        id='test-declined',
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content='**Fireworks Warehouse**\nRecommendation: **Decline**\nHigh risk factors.'
                ),
                HumanMessage(content='Agreed. Outside our risk appetite. Declining.'),
                AIMessage(content='Confirmed. Case declined.'),
            ]
        ),
    )


# =============================================================================
# InterventionDetector Tests
# =============================================================================


class TestInterventionDetector:
    """Tests for InterventionDetector metric."""

    @requires_llm
    @pytest.mark.asyncio
    async def test_factual_correction(self, factual_correction_item):
        """Test detection of factual correction intervention."""
        detector = InterventionDetector()
        result = await detector.execute(factual_correction_item)

        assert result.signals.has_intervention is True
        assert result.signals.intervention_type == 'correction_factual'
        assert result.signals.escalation_type == 'hard'
        assert result.signals.is_stp is False

    @requires_llm
    @pytest.mark.asyncio
    async def test_stp_no_human(self, stp_item):
        """Test STP detection when no human messages."""
        detector = InterventionDetector()
        result = await detector.execute(stp_item)

        assert result.signals.has_intervention is False
        assert result.signals.intervention_type == 'no_intervention'
        assert result.signals.escalation_type == 'none'
        assert result.signals.is_stp is True

    @requires_llm
    @pytest.mark.asyncio
    async def test_authority_approval(self, approval_required_item):
        """Test detection of authority approval."""
        detector = InterventionDetector()
        result = await detector.execute(approval_required_item)

        assert result.signals.has_intervention is True
        assert result.signals.intervention_type == 'approval'
        assert result.signals.escalation_type == 'authority'


# =============================================================================
# SentimentDetector Tests
# =============================================================================


class TestSentimentDetector:
    """Tests for SentimentDetector metric."""

    @requires_llm
    @pytest.mark.asyncio
    async def test_frustrated_user(self, frustrated_user_item):
        """Test detection of frustrated user."""
        detector = SentimentDetector()
        result = await detector.execute(frustrated_user_item)

        assert result.signals.is_frustrated is True
        assert result.signals.sentiment_score < 0.4

    @requires_llm
    @pytest.mark.asyncio
    async def test_neutral_interaction(self, factual_correction_item):
        """Test detection of neutral interaction."""
        detector = SentimentDetector()
        result = await detector.execute(factual_correction_item)

        # Factual correction should be neutral/professional
        assert result.signals.sentiment in ['neutral', 'positive']

    @requires_llm
    @pytest.mark.asyncio
    async def test_no_human_messages(self, stp_item):
        """Test handling of no human messages."""
        detector = SentimentDetector()
        result = await detector.execute(stp_item)

        # Default to neutral when no human messages
        assert result.signals.sentiment_score == 0.5


# =============================================================================
# ResolutionDetector Tests
# =============================================================================


class TestResolutionDetector:
    """Tests for ResolutionDetector metric."""

    @requires_llm
    @pytest.mark.asyncio
    async def test_approved_resolution(self, factual_correction_item):
        """Test detection of approved resolution."""
        detector = ResolutionDetector()
        result = await detector.execute(factual_correction_item)

        assert result.signals.is_resolved is True
        assert result.signals.final_status == 'approved'
        assert result.signals.resolution_type == 'approved'

    @requires_llm
    @pytest.mark.asyncio
    async def test_declined_resolution(self, declined_item):
        """Test detection of declined resolution."""
        detector = ResolutionDetector()
        result = await detector.execute(declined_item)

        assert result.signals.is_resolved is True
        assert result.signals.final_status == 'declined'
        assert result.signals.resolution_type == 'declined'

    @requires_llm
    @pytest.mark.asyncio
    async def test_unresolved(self, frustrated_user_item):
        """Test detection of unresolved conversation."""
        detector = ResolutionDetector()
        result = await detector.execute(frustrated_user_item)

        # Frustrated conversation doesn't reach resolution
        assert (
            result.signals.is_resolved is False
            or result.signals.final_status == 'pending'
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricIntegration:
    """Integration tests running all metrics together."""

    @requires_llm
    @pytest.mark.asyncio
    async def test_all_metrics_on_single_item(self, factual_correction_item):
        """Test running all three metrics on a single item."""
        intervention = InterventionDetector()
        sentiment = SentimentDetector()
        resolution = ResolutionDetector()

        int_result = await intervention.execute(factual_correction_item)
        sent_result = await sentiment.execute(factual_correction_item)
        res_result = await resolution.execute(factual_correction_item)

        # All should complete without error
        assert int_result.signals is not None
        assert sent_result.signals is not None
        assert res_result.signals is not None

        # Combined analysis should be coherent
        assert int_result.signals.has_intervention is True
        assert sent_result.signals.is_frustrated is False
        assert res_result.signals.is_resolved is True

    @requires_llm
    @pytest.mark.asyncio
    async def test_signal_descriptors(self, factual_correction_item):
        """Test that signal descriptors are properly defined."""
        detector = InterventionDetector()
        result = await detector.execute(factual_correction_item)

        signals = detector.get_signals(result.signals)
        signal_names = [s.name for s in signals]

        assert 'has_intervention' in signal_names
        assert 'intervention_type' in signal_names
        assert 'escalation_type' in signal_names
        assert 'is_stp' in signal_names
