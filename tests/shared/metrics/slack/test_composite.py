"""Tests for SlackConversationAnalyzer composite metric."""

from unittest.mock import AsyncMock

import pytest
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation

from shared.metrics.slack.composite import (
    AcceptanceSignals,
    EngagementSignals,
    EscalationSignals,
    FrustrationSignals,
    InteractionSignals,
    OverrideSignals,
    RecommendationSignals,
    SatisfactionSignals,
    SlackAnalysisResult,
    SlackConversationAnalyzer,
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
def conversation_with_override():
    """Create a conversation with an override."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Review this case MGT-BOP-9999'),
            AIMessage(
                content='Recommend Decline. Risk factors detected. Base Score: 35/100'
            ),
            HumanMessage(
                content="I don't think decline is right. The additional docs show good history."
            ),
            AIMessage(content='Understood, updating recommendation.'),
        ]
    )


@pytest.fixture
def frustrated_conversation():
    """Create a conversation with frustration signals."""
    return MultiTurnConversation(
        messages=[
            HumanMessage(content='Help me with this case'),
            AIMessage(content='Here is some information.'),
            HumanMessage(content='That DOESNT WORK. Why is this so hard???'),
            AIMessage(content='Let me try again.'),
            HumanMessage(content='Forget it, this is frustrating!!'),
        ]
    )


class TestSlackConversationAnalyzer:
    """Tests for SlackConversationAnalyzer composite metric."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SlackConversationAnalyzer()

    @pytest.mark.asyncio
    async def test_heuristic_signals_extraction(self, analyzer, simple_conversation):
        """Should extract all heuristic signals in single pass."""
        item = DatasetItem(
            conversation=simple_conversation,
            additional_input={
                'thread_ts': '123.456',
                'channel_id': 'C123',
                'sender': 'user123',
            },
        )

        # Mock LLM to fail so we test heuristic path
        analyzer.comprehensive_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await analyzer.execute(item)

        assert result.signals is not None
        signals = result.signals

        # Metadata
        assert signals.thread_id == '123.456'
        assert signals.channel_id == 'C123'
        assert signals.sender == 'user123'

        # Interaction signals
        assert signals.interaction.ai_message_count == 2
        assert signals.interaction.human_message_count == 2
        assert signals.interaction.total_turn_count == 4
        assert signals.interaction.is_interactive is True

        # Engagement signals
        assert signals.engagement.has_multiple_interactions is True
        assert signals.engagement.interaction_depth >= 1

        # Recommendation signals
        assert signals.recommendation.has_recommendation is True
        assert signals.recommendation.recommendation_type == 'approve'
        assert signals.recommendation.case_id == 'MGT-BOP-1234567'
        assert signals.recommendation.case_priority == 85

    @pytest.mark.asyncio
    async def test_heuristic_frustration_detection(
        self, analyzer, frustrated_conversation
    ):
        """Should detect frustration via heuristic patterns."""
        item = DatasetItem(conversation=frustrated_conversation)
        analyzer.comprehensive_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await analyzer.execute(item)

        signals = result.signals
        # Should detect: ALL CAPS, multiple question marks, exclamation marks
        assert signals.frustration.frustration_score > 0
        assert len(signals.frustration.frustration_indicators) > 0

    @pytest.mark.asyncio
    async def test_heuristic_escalation_detection(self, analyzer):
        """Should detect escalation via heuristic patterns."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Help with this'),
                AIMessage(
                    content='I apologize, I encountered an error processing this.'
                ),
                HumanMessage(content='@john_doe can you help?'),
            ]
        )
        item = DatasetItem(conversation=conversation)
        analyzer.comprehensive_analyzer.execute = AsyncMock(
            side_effect=Exception('LLM unavailable')
        )

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.escalation.is_escalated is True

    @pytest.mark.asyncio
    async def test_handles_no_conversation(self, analyzer):
        """Should handle missing conversation."""
        item = DatasetItem(query='No conversation')

        result = await analyzer.execute(item)

        assert result.score is None
        signals = result.signals
        assert signals.interaction.is_interactive is False
        assert signals.recommendation.has_recommendation is False

    @pytest.mark.asyncio
    async def test_handles_no_human_messages(self, analyzer):
        """Should handle conversations without human messages."""
        conversation = MultiTurnConversation(
            messages=[AIMessage(content='AI only message')]
        )
        item = DatasetItem(conversation=conversation)

        result = await analyzer.execute(item)

        signals = result.signals
        assert signals.interaction.human_message_count == 0
        assert signals.interaction.is_interactive is False


class TestSlackAnalysisResult:
    """Tests for SlackAnalysisResult model methods."""

    def test_to_rows_returns_all_metrics(self):
        """Should return rows for all metrics that have data."""
        result = SlackAnalysisResult(
            thread_id='123',
            channel_id='C123',
            sender='user1',
            interaction=InteractionSignals(is_interactive=True),
            engagement=EngagementSignals(interaction_depth=2),
            recommendation=RecommendationSignals(has_recommendation=True),
            escalation=EscalationSignals(is_escalated=False),
            frustration=FrustrationSignals(frustration_score=0.2),
            acceptance=AcceptanceSignals(is_accepted=True),
            override=OverrideSignals(is_overridden=False),
            satisfaction=SatisfactionSignals(satisfaction_score=0.8),
        )

        rows = result.to_rows()

        assert len(rows) == 8
        metric_names = [r['metric'] for r in rows]
        assert 'interaction' in metric_names
        assert 'engagement' in metric_names
        assert 'recommendation' in metric_names
        assert 'escalation' in metric_names
        assert 'frustration' in metric_names
        assert 'acceptance' in metric_names
        assert 'override' in metric_names
        assert 'satisfaction' in metric_names

    def test_to_rows_includes_metadata(self):
        """Should include metadata in each row when requested."""
        result = SlackAnalysisResult(
            thread_id='123',
            channel_id='C123',
            sender='user1',
        )

        rows = result.to_rows(include_metadata=True)

        for row in rows:
            assert row['thread_id'] == '123'
            assert row['channel_id'] == 'C123'
            assert row['sender'] == 'user1'

    def test_to_rows_excludes_metadata(self):
        """Should exclude metadata when requested."""
        result = SlackAnalysisResult(
            thread_id='123',
            sender='user1',
        )

        rows = result.to_rows(include_metadata=False)

        for row in rows:
            assert 'thread_id' not in row
            assert 'sender' not in row

    def test_to_kpi_summary(self):
        """Should return KPI-relevant values."""
        result = SlackAnalysisResult(
            thread_id='123',
            sender='user1',
            interaction=InteractionSignals(is_interactive=True),
            engagement=EngagementSignals(
                has_multiple_interactions=True, interaction_depth=3
            ),
            escalation=EscalationSignals(
                is_escalated=True, escalation_type='team_mention'
            ),
            frustration=FrustrationSignals(frustration_score=0.7, is_frustrated=True),
            recommendation=RecommendationSignals(has_recommendation=True),
            acceptance=AcceptanceSignals(
                is_accepted=True, acceptance_status='accepted'
            ),
            override=OverrideSignals(is_overridden=False, override_type='no_override'),
            satisfaction=SatisfactionSignals(
                satisfaction_score=0.8, is_satisfactory=True
            ),
        )

        summary = result.to_kpi_summary()

        assert summary['thread_id'] == '123'
        assert summary['sender'] == 'user1'
        assert summary['is_interactive'] is True
        assert summary['has_multiple_interactions'] is True
        assert summary['interaction_depth'] == 3
        assert summary['is_escalated'] is True
        assert summary['frustration_score'] == 0.7
        assert summary['is_frustrated'] is True
        assert summary['has_recommendation'] is True
        assert summary['is_accepted'] is True
        assert summary['is_overridden'] is False
        assert summary['is_satisfactory'] is True


class TestSignalModels:
    """Tests for individual signal models."""

    def test_interaction_signals_defaults(self):
        """Should have sensible defaults."""
        signals = InteractionSignals()

        assert signals.ai_message_count == 0
        assert signals.human_message_count == 0
        assert signals.is_interactive is False

    def test_recommendation_signals_defaults(self):
        """Should have sensible defaults."""
        signals = RecommendationSignals()

        assert signals.has_recommendation is False
        assert signals.recommendation_type == 'none'
        assert signals.case_id is None

    def test_escalation_signals_defaults(self):
        """Should have sensible defaults."""
        signals = EscalationSignals()

        assert signals.is_escalated is False
        assert signals.escalation_type == 'no_escalation'

    def test_frustration_signals_bounds(self):
        """Should enforce score bounds."""
        signals = FrustrationSignals(frustration_score=0.5)

        assert 0.0 <= signals.frustration_score <= 1.0


class TestConfigurableThresholds:
    """Tests for configurable thresholds."""

    def test_frustration_threshold(self):
        """Should use custom frustration threshold."""
        analyzer = SlackConversationAnalyzer(frustration_threshold=0.3)

        assert analyzer.frustration_threshold == 0.3

    def test_satisfaction_threshold(self):
        """Should use custom satisfaction threshold."""
        analyzer = SlackConversationAnalyzer(satisfaction_threshold=0.5)

        assert analyzer.satisfaction_threshold == 0.5


class TestSignalDescriptors:
    """Tests for signal descriptors."""

    @pytest.mark.asyncio
    async def test_generates_signal_descriptors(self):
        """Should generate proper signal descriptors."""
        analyzer = SlackConversationAnalyzer()
        result = SlackAnalysisResult()

        signals = analyzer.get_signals(result)

        signal_names = [s.name for s in signals]
        assert 'thread_id' in signal_names
        assert 'is_interactive' in signal_names
        assert 'is_escalated' in signal_names
        assert 'is_frustrated' in signal_names
        assert 'has_recommendation' in signal_names
        assert 'is_accepted' in signal_names
        assert 'is_overridden' in signal_names

        # Check headline signals
        headline_signals = [s for s in signals if s.headline_display]
        assert len(headline_signals) >= 5


class TestExpandResults:
    """Tests for expand_results and results_to_kpi_dataframe methods."""

    @pytest.fixture
    def mock_evaluation_result(self):
        """Create a mock EvaluationResult that mimics real evaluation_runner output."""
        from dataclasses import dataclass
        from typing import Any, Dict, List, Optional

        @dataclass
        class MockMetricScore:
            name: str
            score: Optional[float]
            metadata: Dict[str, Any]
            source: str = 'axion'
            cost_estimate: float = 0.01
            timestamp: str = '2024-01-01T00:00:00'

        @dataclass
        class MockTestCase:
            id: str

        @dataclass
        class MockTestResult:
            test_case: MockTestCase
            score_results: List[MockMetricScore]

        @dataclass
        class MockEvaluationResult:
            results: List[MockTestResult]
            run_id: str = 'run_123'
            evaluation_name: str = 'test_eval'
            metadata: Dict[str, Any] = None
            timestamp: str = '2024-01-01T00:00:00'

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

        # Create sample SlackAnalysisResult objects
        result1 = SlackAnalysisResult(
            thread_id='thread1',
            channel_id='C123',
            sender='user1',
            interaction=InteractionSignals(is_interactive=True, ai_message_count=2),
            engagement=EngagementSignals(interaction_depth=3),
            escalation=EscalationSignals(is_escalated=False),
            frustration=FrustrationSignals(frustration_score=0.2, is_frustrated=False),
            recommendation=RecommendationSignals(has_recommendation=True),
            acceptance=AcceptanceSignals(is_accepted=True),
            override=OverrideSignals(is_overridden=False),
            satisfaction=SatisfactionSignals(satisfaction_score=0.0),
        )

        result2 = SlackAnalysisResult(
            thread_id='thread2',
            channel_id='C456',
            sender='user2',
            interaction=InteractionSignals(is_interactive=True, ai_message_count=3),
            escalation=EscalationSignals(
                is_escalated=True, escalation_type='team_mention'
            ),
            frustration=FrustrationSignals(frustration_score=0.8, is_frustrated=True),
        )

        return MockEvaluationResult(
            results=[
                MockTestResult(
                    test_case=MockTestCase(id='item1'),
                    score_results=[
                        MockMetricScore(
                            name='Slack Conversation Analyzer',
                            score=None,
                            metadata={'slack_analysis_result': result1.model_dump()},
                        )
                    ],
                ),
                MockTestResult(
                    test_case=MockTestCase(id='item2'),
                    score_results=[
                        MockMetricScore(
                            name='Slack Conversation Analyzer',
                            score=None,
                            metadata={'slack_analysis_result': result2.model_dump()},
                        )
                    ],
                ),
            ]
        )

    def test_expand_results_creates_rows_for_active_metrics(
        self, mock_evaluation_result
    ):
        """Should create rows for metrics that have data."""
        expanded = SlackConversationAnalyzer.expand_results(mock_evaluation_result)

        # item1 has 8 metrics, item2 has 5 metrics (missing acceptance, override, satisfaction)
        assert len(expanded) == 13

        # Check metric distribution - all 8 should appear for item1
        item1_metrics = expanded[expanded['id'] == 'item1']['metric_name'].tolist()
        assert len(item1_metrics) == 8

        # item2 should have 5 metrics
        item2_metrics = expanded[expanded['id'] == 'item2']['metric_name'].tolist()
        assert len(item2_metrics) == 5

    def test_expand_results_matches_to_dataframe_format(self, mock_evaluation_result):
        """Should have same columns as to_dataframe() would produce."""
        expanded = SlackConversationAnalyzer.expand_results(mock_evaluation_result)

        # These columns should match to_dataframe() format
        assert 'id' in expanded.columns
        assert 'metric_name' in expanded.columns
        assert 'metric_score' in expanded.columns
        assert 'metric_type' in expanded.columns
        assert 'threshold' in expanded.columns
        assert 'passed' in expanded.columns
        assert 'explanation' in expanded.columns
        assert 'signals' in expanded.columns
        assert 'metadata' in expanded.columns
        assert 'run_id' in expanded.columns
        assert 'evaluation_name' in expanded.columns

    def test_expand_results_metric_names(self, mock_evaluation_result):
        """Should have correct metric names."""
        expanded = SlackConversationAnalyzer.expand_results(mock_evaluation_result)

        expected_metrics = {
            'slack_interaction',
            'slack_engagement',
            'slack_recommendation',
            'slack_escalation',
            'slack_frustration',
            'slack_acceptance',
            'slack_override',
            'slack_satisfaction',
        }
        actual_metrics = set(expanded['metric_name'].unique())
        assert actual_metrics == expected_metrics

    def test_expand_results_scores(self, mock_evaluation_result):
        """Should have correct scores for each metric type."""
        expanded = SlackConversationAnalyzer.expand_results(mock_evaluation_result)

        # Frustration row for item2 should have score 0.8
        item2_frustration = expanded[
            (expanded['id'] == 'item2')
            & (expanded['metric_name'] == 'slack_frustration')
        ].iloc[0]
        assert item2_frustration['metric_score'] == 0.8
        assert item2_frustration['metric_type'] == 'analysis'

        # Escalation row for item2 - check signals
        item2_escalation = expanded[
            (expanded['id'] == 'item2')
            & (expanded['metric_name'] == 'slack_escalation')
        ].iloc[0]
        assert item2_escalation['signals']['is_escalated'] is True

    def test_expand_results_signals_contain_metric_data(self, mock_evaluation_result):
        """Should have metric-specific signals in each row."""
        expanded = SlackConversationAnalyzer.expand_results(mock_evaluation_result)

        # Interaction row should have interaction signals
        interaction_row = expanded[
            (expanded['id'] == 'item1')
            & (expanded['metric_name'] == 'slack_interaction')
        ].iloc[0]
        signals = interaction_row['signals']
        assert signals['is_interactive'] == True  # noqa: E712
        assert signals['ai_message_count'] == 2

        # Frustration row should have frustration signals
        frustration_row = expanded[
            (expanded['id'] == 'item2')
            & (expanded['metric_name'] == 'slack_frustration')
        ].iloc[0]
        signals = frustration_row['signals']
        assert signals['frustration_score'] == 0.8
        assert signals['is_frustrated'] == True  # noqa: E712

    def test_expand_results_metadata_contains_slack_info(self, mock_evaluation_result):
        """Should include slack metadata in each row's metadata."""
        expanded = SlackConversationAnalyzer.expand_results(mock_evaluation_result)

        item1_row = expanded[expanded['id'] == 'item1'].iloc[0]
        metadata = item1_row['metadata']
        assert metadata['thread_id'] == 'thread1'
        assert metadata['channel_id'] == 'C123'
        assert metadata['sender'] == 'user1'

    def test_expand_results_handles_empty_results(self):
        """Should handle empty EvaluationResult."""
        from dataclasses import dataclass

        @dataclass
        class MockEmptyResult:
            results: list
            run_id: str = 'run_123'
            evaluation_name: str = 'test'
            metadata: dict = None

        empty_result = MockEmptyResult(results=[])
        expanded = SlackConversationAnalyzer.expand_results(empty_result)
        assert len(expanded) == 0

    def test_results_to_kpi_dataframe(self, mock_evaluation_result):
        """Should create KPI summary DataFrame."""
        kpi_df = SlackConversationAnalyzer.results_to_kpi_dataframe(
            mock_evaluation_result
        )

        # 1 row per item
        assert len(kpi_df) == 2

        # Check KPI columns exist
        assert 'is_interactive' in kpi_df.columns
        assert 'is_escalated' in kpi_df.columns
        assert 'is_frustrated' in kpi_df.columns
        assert 'has_recommendation' in kpi_df.columns
        assert 'is_accepted' in kpi_df.columns
        assert 'is_overridden' in kpi_df.columns
        assert 'frustration_score' in kpi_df.columns

    def test_results_to_kpi_dataframe_values(self, mock_evaluation_result):
        """Should have correct KPI values."""
        kpi_df = SlackConversationAnalyzer.results_to_kpi_dataframe(
            mock_evaluation_result
        )

        # Check item1 values
        item1 = kpi_df[kpi_df['id'] == 'item1'].iloc[0]
        assert item1['is_interactive'] == True  # noqa: E712
        assert item1['is_escalated'] == False  # noqa: E712
        assert item1['is_frustrated'] == False  # noqa: E712
        assert item1['has_recommendation'] == True  # noqa: E712
        assert item1['is_accepted'] == True  # noqa: E712

        # Check item2 values
        item2 = kpi_df[kpi_df['id'] == 'item2'].iloc[0]
        assert item2['is_escalated'] == True  # noqa: E712
        assert item2['is_frustrated'] == True  # noqa: E712
        assert item2['frustration_score'] == 0.8
