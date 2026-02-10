import re
from typing import Any, Dict, List, Literal, Optional

from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SubMetricResult
from pydantic import Field

from eval_workbench.shared.metrics.slack.config import AnalyzerConfig
from eval_workbench.shared.metrics.slack.utils import (
    ReactionSignals,
    StalemateSignals,
    analyze_reactions,
    count_questions,
    detect_stalemate,
    extract_case_id,
    extract_mentions,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
    get_human_messages,
    parse_slack_metadata,
)


class InteractionSignals(RichBaseModel):
    """Interaction signals (heuristic)."""

    ai_message_count: int = Field(default=0)
    human_message_count: int = Field(default=0)
    total_turn_count: int = Field(default=0)
    reply_count: Optional[int] = Field(default=None)
    is_ai_initiated: bool = Field(default=False)
    has_human_response: bool = Field(default=False)
    is_interactive: bool = Field(default=False)


class EngagementSignals(RichBaseModel):
    """Engagement signals (heuristic)."""

    interaction_depth: int = Field(default=0)
    has_multiple_interactions: bool = Field(default=False)
    avg_human_response_length: float = Field(default=0.0)
    avg_ai_response_length: float = Field(default=0.0)
    question_count: int = Field(default=0)
    mention_count: int = Field(default=0)
    unique_participants: int = Field(default=0)


class RecommendationSignals(RichBaseModel):
    """Recommendation signals (heuristic)."""

    has_recommendation: bool = Field(default=False)
    recommendation_type: Literal['approve', 'decline', 'review', 'hold', 'none'] = (
        Field(default='none')
    )
    recommendation_turn_index: Optional[int] = Field(default=None)
    recommendation_confidence: Optional[float] = Field(default=None)
    case_id: Optional[str] = Field(default=None)
    case_priority: Optional[int] = Field(default=None)


class HeuristicAnalysisResult(RichBaseModel):
    """Result from heuristic analysis."""

    # Metadata
    thread_id: Optional[str] = Field(default=None)
    channel_id: Optional[str] = Field(default=None)
    sender: Optional[str] = Field(default=None)
    case_id: Optional[str] = Field(default=None)

    # Signals
    interaction: InteractionSignals = Field(default_factory=InteractionSignals)
    engagement: EngagementSignals = Field(default_factory=EngagementSignals)
    recommendation: RecommendationSignals = Field(default_factory=RecommendationSignals)
    reactions: Optional[ReactionSignals] = Field(default=None)
    stalemate: Optional[StalemateSignals] = Field(default=None)

    def to_kpi_summary(self) -> Dict[str, Any]:
        """Extract key KPI values."""
        kpi = {
            'thread_id': self.thread_id,
            'case_id': self.case_id,
            'is_interactive': self.interaction.is_interactive,
            'human_message_count': self.interaction.human_message_count,
            'interaction_depth': self.engagement.interaction_depth,
            'has_recommendation': self.recommendation.has_recommendation,
            'recommendation_type': self.recommendation.recommendation_type,
        }
        if self.reactions:
            kpi['reaction_sentiment_score'] = self.reactions.reaction_sentiment_score
        if self.stalemate:
            kpi['is_stalemate'] = self.stalemate.is_stalemate
        return kpi


@metric(
    name='Slack Heuristic Analyzer',
    key='slack_heuristic_analyzer',
    description='Zero-cost heuristic analysis of Slack conversations (no LLM).',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'heuristic', 'multi_turn'],
)
class SlackHeuristicAnalyzer(BaseMetric):
    """
    Heuristic-only analyzer for Slack conversations.

    Computes zero-cost metrics without any LLM calls:
    - `interaction`: Message counts, is_interactive
    - `engagement`: Depth, response lengths, questions
    - `recommendation`: Has recommendation, type, confidence
    - `reactions`: Slack emoji reaction sentiment
    - `stalemate`: Bot repeating same message detection

    Use this when you want fast, free analysis of conversation structure.
    """

    is_multi_metric = True
    include_parent_score = False
    sub_metric_prefix = False

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        include_reactions: bool = True,
        include_stalemate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.analyzer_config = config or AnalyzerConfig()
        self.include_reactions = include_reactions
        self.include_stalemate = include_stalemate

    @property
    def cost_estimate(self) -> float:
        """Return 0.0 - heuristic analyzer uses no LLM."""
        return 0.0

    @trace(name='SlackHeuristicAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute heuristic analysis."""
        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=HeuristicAnalysisResult(),
            )

        metadata = parse_slack_metadata(item.additional_input)
        additional = item.additional_input or {}
        all_messages = item.conversation.messages
        ai_messages = get_ai_messages(item.conversation)
        human_messages = get_human_messages(item.conversation)

        ai_count = len(ai_messages)
        human_count = len(human_messages)
        total_count = len(all_messages)

        # Interaction
        interaction = InteractionSignals(
            ai_message_count=ai_count,
            human_message_count=human_count,
            total_turn_count=total_count,
            reply_count=metadata.reply_count,
            is_ai_initiated=total_count > 0 and isinstance(all_messages[0], AIMessage),
            has_human_response=human_count > 0 and ai_count > 0,
            is_interactive=ai_count > 0 and human_count > 0,
        )

        # Engagement
        all_mentions = []
        for msg in all_messages:
            if msg.content:
                all_mentions.extend(extract_mentions(msg.content))
        unique_mentions = list(set(all_mentions))

        avg_human_len = sum(len(m.content or '') for m in human_messages) / max(
            human_count, 1
        )
        avg_ai_len = sum(len(m.content or '') for m in ai_messages) / max(ai_count, 1)

        engagement = EngagementSignals(
            interaction_depth=self._calculate_interaction_depth(all_messages),
            has_multiple_interactions=human_count > 1,
            avg_human_response_length=round(avg_human_len, 1),
            avg_ai_response_length=round(avg_ai_len, 1),
            question_count=sum(count_questions(m.content or '') for m in all_messages),
            mention_count=len(all_mentions),
            unique_participants=max(1, len(unique_mentions)) if human_count > 0 else 0,
        )

        # Recommendation
        rec_turn = find_recommendation_turn(item.conversation)
        has_recommendation = rec_turn is not None
        rec_type = 'none'
        rec_text = ''
        if has_recommendation:
            rec_message = all_messages[rec_turn]
            rec_text = rec_message.content or ''
            rec_type = extract_recommendation_type(rec_text) or 'none'

        case_id = additional.get('case_id') or self._extract_case_id_from_messages(
            all_messages
        )

        recommendation = RecommendationSignals(
            has_recommendation=has_recommendation,
            recommendation_type=rec_type,
            recommendation_turn_index=rec_turn,
            recommendation_confidence=self._extract_confidence(rec_text)
            if has_recommendation
            else None,
            case_id=case_id,
            case_priority=self._extract_priority_from_messages(ai_messages)
            if has_recommendation
            else None,
        )

        # Reactions
        reactions = None
        if self.include_reactions:
            reactions = analyze_reactions(metadata.reactions)

        # Stalemate
        stalemate = None
        if self.include_stalemate:
            stalemate = detect_stalemate(all_messages)

        result = HeuristicAnalysisResult(
            thread_id=metadata.thread_ts,
            channel_id=metadata.channel_id,
            sender=metadata.sender,
            case_id=case_id,
            interaction=interaction,
            engagement=engagement,
            recommendation=recommendation,
            reactions=reactions,
            stalemate=stalemate,
        )

        return MetricEvaluationResult(
            score=None,
            explanation=f'Analyzed {total_count} messages. Interactive: {interaction.is_interactive}.',
            signals=result,
            metadata={'heuristic_result': result.model_dump()},
        )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """Extract sub-metrics."""
        signals: HeuristicAnalysisResult = result.signals
        if not signals:
            return []

        sub_metrics = [
            SubMetricResult(
                name='interaction',
                score=1.0 if signals.interaction.is_interactive else 0.0,
                explanation=f'Depth: {signals.engagement.interaction_depth}',
                metric_category=MetricCategory.SCORE,
                group='heuristic',
                metadata=signals.interaction.model_dump(),
            ),
            SubMetricResult(
                name='engagement',
                score=min(signals.engagement.interaction_depth / 10.0, 1.0),
                explanation=f'Depth: {signals.engagement.interaction_depth}',
                metric_category=MetricCategory.SCORE,
                group='heuristic',
                metadata=signals.engagement.model_dump(),
            ),
            SubMetricResult(
                name='recommendation_type',
                score=None,
                explanation=signals.recommendation.recommendation_type,
                metric_category=MetricCategory.CLASSIFICATION,
                group='heuristic',
                metadata=signals.recommendation.model_dump(),
            ),
        ]

        if signals.reactions:
            sub_metrics.append(
                SubMetricResult(
                    name='reaction_sentiment',
                    score=signals.reactions.reaction_sentiment_score,
                    explanation=f'Score: {signals.reactions.reaction_sentiment_score:.2f}',
                    metric_category=MetricCategory.SCORE,
                    group='heuristic',
                    metadata=signals.reactions.model_dump(),
                )
            )

        if signals.stalemate:
            sub_metrics.append(
                SubMetricResult(
                    name='is_stalemate',
                    score=None,
                    explanation=str(signals.stalemate.is_stalemate).lower(),
                    metric_category=MetricCategory.CLASSIFICATION,
                    group='heuristic',
                    metadata=signals.stalemate.model_dump(),
                )
            )

        return sub_metrics

    def _calculate_interaction_depth(self, messages: list) -> int:
        if not messages:
            return 0
        exchanges = 0
        last_role = None
        for msg in messages:
            current_role = 'human' if isinstance(msg, HumanMessage) else 'ai'
            if last_role is not None and current_role != last_role:
                exchanges += 1
            last_role = current_role
        return (exchanges + 1) // 2 if exchanges > 0 else 0

    def _extract_case_id_from_messages(self, messages: list) -> Optional[str]:
        for msg in messages:
            if msg.content:
                case_id = extract_case_id(msg.content)
                if case_id:
                    return case_id
        return None

    def _extract_priority_from_messages(self, ai_messages: list) -> Optional[int]:
        for msg in ai_messages:
            if msg.content:
                priority = extract_priority_score(msg.content)
                if priority is not None:
                    return priority
        return None

    def _extract_confidence(self, text: str) -> Optional[float]:
        if not text:
            return None
        match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%?', text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return min(max(value / 100 if value > 1 else value, 0), 1)
        if 'high confidence' in text.lower():
            return 0.9
        if 'medium confidence' in text.lower():
            return 0.6
        if 'low confidence' in text.lower():
            return 0.3
        return None


__all__ = [
    'SlackHeuristicAnalyzer',
    'HeuristicAnalysisResult',
    'InteractionSignals',
    'EngagementSignals',
    'RecommendationSignals',
]
