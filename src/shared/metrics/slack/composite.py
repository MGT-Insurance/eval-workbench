from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import (
    build_transcript,
    count_questions,
    extract_case_id,
    extract_mentions,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
    get_human_messages,
    parse_slack_metadata,
)

# Heuristic metrics (no LLM cost)
HEURISTIC_METRICS = {'interaction', 'engagement', 'recommendation'}

# Interaction LLM metrics
INTERACTION_LLM_METRICS = {
    'escalation',
    'frustration',
    'acceptance',
    'override',
    'satisfaction',
}

# Outcome LLM metrics
OUTCOME_LLM_METRICS = {'intervention', 'sentiment', 'resolution'}

# All available metrics
ALL_METRICS = HEURISTIC_METRICS | INTERACTION_LLM_METRICS | OUTCOME_LLM_METRICS


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


class EscalationSignals(RichBaseModel):
    """Escalation signals (LLM) - team mentions, handoffs, errors."""

    is_escalated: bool = Field(default=False)
    escalation_type: Literal[
        'no_escalation',
        'team_mention',
        'explicit_handoff',
        'error_escalation',
        'complexity_escalation',
    ] = Field(default='no_escalation')
    escalation_turn_index: Optional[int] = Field(default=None)
    escalation_targets: List[str] = Field(default_factory=list)
    escalation_reason: str = Field(default='')


class FrustrationSignals(RichBaseModel):
    """Frustration signals (LLM)."""

    frustration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_frustrated: bool = Field(default=False)
    frustration_indicators: List[str] = Field(default_factory=list)
    peak_frustration_turn: Optional[int] = Field(default=None)
    frustration_cause: Literal[
        'none',
        'ai_error',
        'slow_response',
        'wrong_answer',
        'repeated_questions',
        'poor_understanding',
        'system_issue',
        'other',
    ] = Field(default='none')


class AcceptanceSignals(RichBaseModel):
    """Acceptance signals (LLM) - recommendation acceptance."""

    acceptance_status: Literal[
        'accepted',
        'accepted_with_discussion',
        'pending',
        'rejected',
        'modified',
    ] = Field(default='pending')
    is_accepted: bool = Field(default=False)
    acceptance_turn_index: Optional[int] = Field(default=None)
    decision_maker: Optional[str] = Field(default=None)
    turns_to_decision: Optional[int] = Field(default=None)


class OverrideSignals(RichBaseModel):
    """Override signals (LLM) - recommendation override."""

    is_overridden: bool = Field(default=False)
    override_type: Literal[
        'no_override',
        'full_override',
        'partial_override',
        'pending_override',
    ] = Field(default='no_override')
    original_recommendation: str = Field(default='')
    final_decision: Optional[str] = Field(default=None)
    override_reason: str = Field(default='')
    override_reason_category: Literal[
        'none',
        'additional_info',
        'risk_assessment',
        'policy_exception',
        'class_code_issue',
        'rate_issue',
        'experience_judgment',
        'other',
    ] = Field(default='none')


class SatisfactionSignals(RichBaseModel):
    """Satisfaction signals (LLM) - override quality."""

    satisfaction_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_satisfactory: bool = Field(default=False)
    has_clear_reason: bool = Field(default=False)
    has_supporting_evidence: bool = Field(default=False)
    is_actionable: bool = Field(default=False)
    improvement_suggestions: List[str] = Field(default_factory=list)


class InterventionSignals(RichBaseModel):
    """Intervention signals (LLM) - why human intervened."""

    has_intervention: bool = Field(default=False)
    intervention_type: Literal[
        'no_intervention',
        'correction_factual',
        'missing_context',
        'risk_appetite',
        'tech_issue',
        'data_quality',
        'clarification',
        'support',
        'approval',
    ] = Field(default='no_intervention')
    # Escalation classification derived from intervention_type
    intervention_escalation: Literal['hard', 'soft', 'authority', 'none'] = Field(
        default='none'
    )
    is_stp: bool = Field(default=True, description='Straight-through processing')
    intervention_summary: str = Field(default='')
    friction_point: Optional[str] = Field(default=None)
    issue_details: Optional[str] = Field(default=None)


class SentimentSignals(RichBaseModel):
    """Sentiment signals (LLM) - overall user sentiment."""

    sentiment: Literal['positive', 'neutral', 'frustrated', 'confused'] = Field(
        default='neutral'
    )
    sentiment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_positive: bool = Field(default=False)
    is_negative: bool = Field(default=False)
    sentiment_indicators: List[str] = Field(default_factory=list)


class ResolutionSignals(RichBaseModel):
    """Resolution signals (LLM) - conversation outcome."""

    final_status: Literal[
        'approved',
        'declined',
        'blocked',
        'needs_info',
        'stalemate',
        'pending',
    ] = Field(default='pending')
    is_resolved: bool = Field(default=False)
    resolution_type: Optional[str] = Field(default=None)
    is_stalemate: bool = Field(default=False)
    time_to_resolution_seconds: Optional[float] = Field(default=None)


class SlackAnalysisResult(RichBaseModel):
    """Complete analysis result containing all signal groups."""

    # Metadata
    thread_id: Optional[str] = Field(default=None)
    channel_id: Optional[str] = Field(default=None)
    sender: Optional[str] = Field(default=None)
    case_id: Optional[str] = Field(default=None)

    # Heuristic signals (always computed)
    interaction: InteractionSignals = Field(default_factory=InteractionSignals)
    engagement: EngagementSignals = Field(default_factory=EngagementSignals)
    recommendation: RecommendationSignals = Field(default_factory=RecommendationSignals)

    # Original LLM signals (optional)
    escalation: Optional[EscalationSignals] = Field(default=None)
    frustration: Optional[FrustrationSignals] = Field(default=None)
    acceptance: Optional[AcceptanceSignals] = Field(default=None)
    override: Optional[OverrideSignals] = Field(default=None)
    satisfaction: Optional[SatisfactionSignals] = Field(default=None)

    # Feedback LLM signals (optional)
    intervention: Optional[InterventionSignals] = Field(default=None)
    sentiment: Optional[SentimentSignals] = Field(default=None)
    resolution: Optional[ResolutionSignals] = Field(default=None)

    # Metadata
    enabled_metrics: List[str] = Field(default_factory=list)
    llm_analysis_performed: bool = Field(default=False)
    analysis_reasoning: str = Field(default='')

    def _get_active_metrics(self) -> List[str]:
        """Get list of metrics that have data, auto-detecting if enabled_metrics is empty."""
        if self.enabled_metrics:
            return self.enabled_metrics

        # Auto-detect from non-None signal fields
        all_metric_fields = [
            'interaction',
            'engagement',
            'recommendation',
            'escalation',
            'frustration',
            'acceptance',
            'override',
            'satisfaction',
            'intervention',
            'sentiment',
            'resolution',
        ]
        return [m for m in all_metric_fields if getattr(self, m, None) is not None]

    def to_rows(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Convert to separate rows per signal group."""
        metadata = {}
        if include_metadata:
            metadata = {
                'thread_id': self.thread_id,
                'channel_id': self.channel_id,
                'sender': self.sender,
                'case_id': self.case_id,
            }

        rows = []
        for metric_name in self._get_active_metrics():
            signals = getattr(self, metric_name, None)
            if signals is not None:
                rows.append(
                    {
                        'metric': metric_name,
                        **metadata,
                        **signals.model_dump(),
                    }
                )
        return rows

    def to_kpi_summary(self) -> Dict[str, Any]:
        """Extract key KPI values for aggregation."""
        kpi: Dict[str, Any] = {
            'thread_id': self.thread_id,
            'sender': self.sender,
            'case_id': self.case_id,
            # Interaction
            'is_interactive': self.interaction.is_interactive,
            'human_message_count': self.interaction.human_message_count,
            # Engagement
            'interaction_depth': self.engagement.interaction_depth,
            'has_multiple_interactions': self.engagement.has_multiple_interactions,
            # Recommendation
            'has_recommendation': self.recommendation.has_recommendation,
            'recommendation_type': self.recommendation.recommendation_type,
        }

        # Original LLM metrics
        if self.escalation:
            kpi.update(
                {
                    'is_escalated': self.escalation.is_escalated,
                    'escalation_type': self.escalation.escalation_type,
                }
            )
        if self.frustration:
            kpi.update(
                {
                    'frustration_score': self.frustration.frustration_score,
                    'is_frustrated': self.frustration.is_frustrated,
                }
            )
        if self.acceptance:
            kpi.update(
                {
                    'is_accepted': self.acceptance.is_accepted,
                    'acceptance_status': self.acceptance.acceptance_status,
                }
            )
        if self.override:
            kpi.update(
                {
                    'is_overridden': self.override.is_overridden,
                    'override_type': self.override.override_type,
                }
            )
        if self.satisfaction:
            kpi.update(
                {
                    'satisfaction_score': self.satisfaction.satisfaction_score,
                    'is_satisfactory': self.satisfaction.is_satisfactory,
                }
            )

        # Feedback LLM metrics
        if self.intervention:
            kpi.update(
                {
                    'has_intervention': self.intervention.has_intervention,
                    'intervention_type': self.intervention.intervention_type,
                    'intervention_escalation': self.intervention.intervention_escalation,
                    'is_stp': self.intervention.is_stp,
                    'friction_point': self.intervention.friction_point,
                }
            )
        if self.sentiment:
            kpi.update(
                {
                    'sentiment': self.sentiment.sentiment,
                    'sentiment_score': self.sentiment.sentiment_score,
                    'is_positive': self.sentiment.is_positive,
                    'is_negative': self.sentiment.is_negative,
                }
            )
        if self.resolution:
            kpi.update(
                {
                    'final_status': self.resolution.final_status,
                    'is_resolved': self.resolution.is_resolved,
                    'resolution_type': self.resolution.resolution_type,
                    'is_stalemate': self.resolution.is_stalemate,
                }
            )

        return kpi


class ComprehensiveAnalysisInput(RichBaseModel):
    """Input for comprehensive LLM analysis."""

    conversation_transcript: str = Field(description='Full conversation transcript')
    has_recommendation: bool = Field(
        description='Whether a recommendation was detected'
    )
    recommendation_type: str = Field(description='Type of recommendation if found')
    recommendation_turn: Optional[int] = Field(
        description='Turn where recommendation was made'
    )
    detected_mentions: List[str] = Field(description='@mentions detected')
    human_message_count: int = Field(description='Number of human messages')
    # Which analyses to perform
    analyze_escalation: bool = Field(default=True)
    analyze_frustration: bool = Field(default=True)
    analyze_acceptance: bool = Field(default=True)
    analyze_override: bool = Field(default=True)
    analyze_satisfaction: bool = Field(default=True)
    analyze_intervention: bool = Field(default=True)
    analyze_sentiment: bool = Field(default=True)
    analyze_resolution: bool = Field(default=True)


class ComprehensiveAnalysisOutput(RichBaseModel):
    """Output for comprehensive LLM analysis."""

    # Original metrics
    is_escalated: bool = Field(default=False)
    escalation_type: Literal[
        'no_escalation',
        'team_mention',
        'explicit_handoff',
        'error_escalation',
        'complexity_escalation',
    ] = Field(default='no_escalation')
    escalation_turn_index: Optional[int] = Field(default=None)
    escalation_reason: str = Field(default='')

    frustration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    frustration_indicators: List[str] = Field(default_factory=list)
    peak_frustration_turn: Optional[int] = Field(default=None)
    frustration_cause: Literal[
        'none',
        'ai_error',
        'slow_response',
        'wrong_answer',
        'repeated_questions',
        'poor_understanding',
        'system_issue',
        'other',
    ] = Field(default='none')

    acceptance_status: Literal[
        'accepted',
        'accepted_with_discussion',
        'pending',
        'rejected',
        'modified',
    ] = Field(default='pending')
    acceptance_turn_index: Optional[int] = Field(default=None)
    decision_maker: Optional[str] = Field(default=None)

    is_overridden: bool = Field(default=False)
    override_type: Literal[
        'no_override',
        'full_override',
        'partial_override',
        'pending_override',
    ] = Field(default='no_override')
    final_decision: Optional[str] = Field(default=None)
    override_reason: str = Field(default='')
    override_reason_category: Literal[
        'none',
        'additional_info',
        'risk_assessment',
        'policy_exception',
        'class_code_issue',
        'rate_issue',
        'experience_judgment',
        'other',
    ] = Field(default='none')

    satisfaction_score: float = Field(default=0.0, ge=0.0, le=1.0)
    has_clear_reason: bool = Field(default=False)
    has_supporting_evidence: bool = Field(default=False)
    is_actionable: bool = Field(default=False)
    improvement_suggestions: List[str] = Field(default_factory=list)

    # Feedback metrics (new)
    has_human_intervention: bool = Field(default=False)
    intervention_category: Literal[
        'no_intervention',
        'correction_factual',
        'missing_context',
        'risk_appetite',
        'tech_issue',
        'data_quality',
        'clarification',
        'support',
        'approval',
    ] = Field(default='no_intervention')
    intervention_summary: str = Field(default='')
    friction_point: Optional[str] = Field(default=None)
    issue_details: Optional[str] = Field(default=None)

    user_sentiment: Literal['positive', 'neutral', 'frustrated', 'confused'] = Field(
        default='neutral'
    )
    sentiment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    sentiment_indicators: List[str] = Field(default_factory=list)

    conversation_status: Literal[
        'approved',
        'declined',
        'blocked',
        'needs_info',
        'stalemate',
        'pending',
    ] = Field(default='pending')

    reasoning: str = Field(default='')


class ComprehensiveAnalyzer(
    BaseMetric[ComprehensiveAnalysisInput, ComprehensiveAnalysisOutput]
):
    """Internal LLM-based analyzer for comprehensive conversation analysis."""

    instruction = """You are an expert analyzer for Slack conversations between users and an AI assistant (Athena) in an insurance underwriting workflow.

**TASK**: Analyze the conversation comprehensively. Only provide analysis for the metrics requested.

---

## ESCALATION ANALYSIS (if requested)
Determine if conversation was escalated to human team members.
- `no_escalation`: Normal AI-handled
- `team_mention`: User @mentioned team members
- `explicit_handoff`: AI handed off to human
- `error_escalation`: Due to AI error
- `complexity_escalation`: Due to case complexity

## FRUSTRATION ANALYSIS (if requested)
Score user frustration (0.0 = calm, 1.0 = very frustrated).
Look for: ???, !!!, ALL CAPS, complaints, sarcasm, giving up language.
Causes: ai_error, slow_response, wrong_answer, repeated_questions, poor_understanding, system_issue, other, none

## ACCEPTANCE ANALYSIS (if requested, requires recommendation)
Was the AI recommendation accepted?
- `accepted`: Without change
- `accepted_with_discussion`: After discussion
- `pending`: No clear resolution
- `rejected`: Explicitly rejected
- `modified`: Accepted with modifications

## OVERRIDE ANALYSIS (if requested, requires recommendation)
Was the recommendation overridden?
- `no_override`, `full_override`, `partial_override`, `pending_override`
Override reasons: additional_info, risk_assessment, policy_exception, class_code_issue, rate_issue, experience_judgment, other

## SATISFACTION ANALYSIS (if requested, requires override)
Score override explanation quality (0.0 = poor, 1.0 = excellent).
Check: has_clear_reason, has_supporting_evidence, is_actionable

## INTERVENTION ANALYSIS (if requested)
**Why did the human intervene?**
- `no_intervention`: No human involvement (STP)
- `correction_factual`: Human correcting AI's factual errors
- `missing_context`: Human providing context AI didn't have
- `risk_appetite`: Human applying risk judgment
- `tech_issue`: Human reporting a bug/technical problem
- `data_quality`: Human fixing bad/missing data
- `clarification`: Human asking for clarification
- `support`: Human requesting help
- `approval`: Human providing required approval

Extract `friction_point` (specific concept causing discussion) and `issue_details` (technical details).

## SENTIMENT ANALYSIS (if requested)
Overall user sentiment:
- `positive`: Satisfied, appreciative
- `neutral`: Matter-of-fact, professional
- `frustrated`: Annoyed, upset, complaining
- `confused`: Uncertain, misunderstanding

Score: 0.0 = very negative, 0.5 = neutral, 1.0 = very positive

## RESOLUTION ANALYSIS (if requested)
Final conversation outcome:
- `approved` / `declined`: Clear decision
- `blocked`: Pending something
- `needs_info`: Waiting for information
- `stalemate`: No resolution after extended time
- `pending`: No clear outcome

---

**OUTPUT**: Provide all requested fields. Use defaults for unrequested analyses."""

    input_model = ComprehensiveAnalysisInput
    output_model = ComprehensiveAnalysisOutput


@metric(
    name='Slack Conversation Analyzer',
    key='slack_conversation_analyzer',
    description='Unified analysis of Slack conversations with configurable metrics.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'multi_turn', 'analysis', 'configurable'],
)
class SlackConversationAnalyzer(BaseMetric):
    """
    Unified Slack conversation analyzer with configurable metrics.

    **Available Metrics:**

    Heuristic (no LLM cost):
    - `interaction`: Message counts, is_interactive
    - `engagement`: Depth, response lengths, questions
    - `recommendation`: Has recommendation, type, confidence

    Interaction (LLM):
    - `escalation`: Team mentions, handoffs, errors
    - `frustration`: Frustration score and indicators
    - `acceptance`: Recommendation acceptance status
    - `override`: Recommendation override analysis
    - `satisfaction`: Override explanation quality

    Outcome (LLM):
    - `intervention`: Why human intervened (hard/soft/authority escalation)
    - `sentiment`: Overall sentiment (positive/neutral/frustrated/confused)
    - `resolution`: Final outcome (approved/declined/stalemate/pending)

    **Usage:**
    ```python
    # All metrics (default)
    analyzer = SlackConversationAnalyzer()

    # Only heuristic metrics (no LLM cost)
    analyzer = SlackConversationAnalyzer(metrics=['interaction', 'engagement', 'recommendation'])

    # Outcome analysis only
    analyzer = SlackConversationAnalyzer(metrics=['intervention', 'sentiment', 'resolution'])

    # Interaction + specific outcomes
    analyzer = SlackConversationAnalyzer(metrics=[
        'escalation', 'frustration',  # Interaction
        'intervention', 'resolution',  # Outcome
    ])
    ```
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        frustration_threshold: float = 0.6,
        satisfaction_threshold: float = 0.7,
        sentiment_threshold: float = 0.4,
        **kwargs,
    ):
        """
        Initialize the analyzer with configurable metrics.

        Args:
            metrics: List of metrics to enable. Default: all metrics.
                     Options: interaction, engagement, recommendation,
                              escalation, frustration, acceptance, override, satisfaction,
                              intervention, sentiment, resolution
            frustration_threshold: Score threshold for "frustrated" (default: 0.6)
            satisfaction_threshold: Score threshold for "satisfactory" (default: 0.7)
            sentiment_threshold: Score below which sentiment is "negative" (default: 0.4)
        """
        super().__init__(**kwargs)

        # Validate and set metrics
        if metrics is None:
            self.enabled_metrics = ALL_METRICS
        else:
            invalid = set(metrics) - ALL_METRICS
            if invalid:
                raise ValueError(f'Invalid metrics: {invalid}. Valid: {ALL_METRICS}')
            self.enabled_metrics = set(metrics)

        # Always include heuristic metrics (they're free)
        self.enabled_metrics = self.enabled_metrics | HEURISTIC_METRICS

        self.frustration_threshold = frustration_threshold
        self.satisfaction_threshold = satisfaction_threshold
        self.sentiment_threshold = sentiment_threshold

        # Check if any LLM metrics are enabled
        self.needs_llm = bool(
            (self.enabled_metrics & INTERACTION_LLM_METRICS)
            | (self.enabled_metrics & OUTCOME_LLM_METRICS)
        )

        if self.needs_llm:
            self.comprehensive_analyzer = ComprehensiveAnalyzer(**kwargs)

    @trace(name='SlackConversationAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute analysis on conversation with enabled metrics."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=SlackAnalysisResult(enabled_metrics=list(self.enabled_metrics)),
            )

        # Parse metadata
        metadata = parse_slack_metadata(item.additional_input)
        additional = item.additional_input or {}

        # Get messages
        ai_messages = get_ai_messages(item.conversation)
        human_messages = get_human_messages(item.conversation)
        all_messages = item.conversation.messages

        # Build transcript
        transcript = build_transcript(item.conversation)

        # Extract mentions
        all_mentions = []
        for msg in all_messages:
            if msg.content:
                all_mentions.extend(extract_mentions(msg.content))
        unique_mentions = list(set(all_mentions))

        # Find recommendation
        rec_turn = find_recommendation_turn(item.conversation)
        has_recommendation = rec_turn is not None
        rec_type = 'none'
        rec_text = ''
        if has_recommendation:
            rec_message = all_messages[rec_turn]
            rec_text = rec_message.content or ''
            rec_type = extract_recommendation_type(rec_text) or 'none'

        ai_count = len(ai_messages)
        human_count = len(human_messages)
        total_count = len(all_messages)

        interaction = InteractionSignals(
            ai_message_count=ai_count,
            human_message_count=human_count,
            total_turn_count=total_count,
            reply_count=metadata.reply_count,
            is_ai_initiated=total_count > 0 and isinstance(all_messages[0], AIMessage),
            has_human_response=human_count > 0 and ai_count > 0,
            is_interactive=ai_count > 0 and human_count > 0,
        )

        interaction_depth = self._calculate_interaction_depth(all_messages)
        avg_human_len = sum(len(m.content or '') for m in human_messages) / max(
            len(human_messages), 1
        )
        avg_ai_len = sum(len(m.content or '') for m in ai_messages) / max(
            len(ai_messages), 1
        )

        engagement = EngagementSignals(
            interaction_depth=interaction_depth,
            has_multiple_interactions=human_count > 1,
            avg_human_response_length=round(avg_human_len, 1),
            avg_ai_response_length=round(avg_ai_len, 1),
            question_count=sum(count_questions(m.content or '') for m in all_messages),
            mention_count=len(all_mentions),
            unique_participants=max(1, len(unique_mentions)) if human_count > 0 else 0,
        )

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

        escalation = None
        frustration = None
        acceptance = None
        override = None
        satisfaction = None
        intervention = None
        sentiment = None
        resolution = None
        llm_performed = False
        reasoning = ''

        if self.needs_llm and human_count > 0:
            try:
                analysis_input = ComprehensiveAnalysisInput(
                    conversation_transcript=transcript,
                    has_recommendation=has_recommendation,
                    recommendation_type=rec_type,
                    recommendation_turn=rec_turn,
                    detected_mentions=unique_mentions,
                    human_message_count=human_count,
                    analyze_escalation='escalation' in self.enabled_metrics,
                    analyze_frustration='frustration' in self.enabled_metrics,
                    analyze_acceptance='acceptance' in self.enabled_metrics,
                    analyze_override='override' in self.enabled_metrics,
                    analyze_satisfaction='satisfaction' in self.enabled_metrics,
                    analyze_intervention='intervention' in self.enabled_metrics,
                    analyze_sentiment='sentiment' in self.enabled_metrics,
                    analyze_resolution='resolution' in self.enabled_metrics,
                )

                llm_result = await self.comprehensive_analyzer.execute(analysis_input)
                llm_performed = True
                reasoning = llm_result.reasoning

                # Map results to signal models
                if 'escalation' in self.enabled_metrics:
                    escalation = EscalationSignals(
                        is_escalated=llm_result.is_escalated,
                        escalation_type=llm_result.escalation_type,
                        escalation_turn_index=llm_result.escalation_turn_index,
                        escalation_targets=unique_mentions
                        if llm_result.is_escalated
                        else [],
                        escalation_reason=llm_result.escalation_reason,
                    )

                if 'frustration' in self.enabled_metrics:
                    frustration = FrustrationSignals(
                        frustration_score=llm_result.frustration_score,
                        is_frustrated=llm_result.frustration_score
                        >= self.frustration_threshold,
                        frustration_indicators=llm_result.frustration_indicators,
                        peak_frustration_turn=llm_result.peak_frustration_turn,
                        frustration_cause=llm_result.frustration_cause,
                    )

                if 'acceptance' in self.enabled_metrics and has_recommendation:
                    acceptance = AcceptanceSignals(
                        acceptance_status=llm_result.acceptance_status,
                        is_accepted=llm_result.acceptance_status
                        in ['accepted', 'accepted_with_discussion'],
                        acceptance_turn_index=llm_result.acceptance_turn_index,
                        decision_maker=llm_result.decision_maker,
                        turns_to_decision=(
                            llm_result.acceptance_turn_index - rec_turn
                            if llm_result.acceptance_turn_index and rec_turn
                            else None
                        ),
                    )

                if 'override' in self.enabled_metrics and has_recommendation:
                    override = OverrideSignals(
                        is_overridden=llm_result.is_overridden,
                        override_type=llm_result.override_type,
                        original_recommendation=rec_type,
                        final_decision=llm_result.final_decision,
                        override_reason=llm_result.override_reason,
                        override_reason_category=llm_result.override_reason_category,
                    )

                if 'satisfaction' in self.enabled_metrics and llm_result.is_overridden:
                    satisfaction = SatisfactionSignals(
                        satisfaction_score=llm_result.satisfaction_score,
                        is_satisfactory=llm_result.satisfaction_score
                        >= self.satisfaction_threshold,
                        has_clear_reason=llm_result.has_clear_reason,
                        has_supporting_evidence=llm_result.has_supporting_evidence,
                        is_actionable=llm_result.is_actionable,
                        improvement_suggestions=llm_result.improvement_suggestions,
                    )

                if 'intervention' in self.enabled_metrics:
                    int_type = llm_result.intervention_category
                    esc_type, is_stp = self._classify_intervention_escalation(int_type)
                    intervention = InterventionSignals(
                        has_intervention=llm_result.has_human_intervention,
                        intervention_type=int_type,
                        intervention_escalation=esc_type,
                        is_stp=is_stp,
                        intervention_summary=llm_result.intervention_summary,
                        friction_point=llm_result.friction_point,
                        issue_details=llm_result.issue_details,
                    )

                if 'sentiment' in self.enabled_metrics:
                    sentiment = SentimentSignals(
                        sentiment=llm_result.user_sentiment,
                        sentiment_score=llm_result.sentiment_score,
                        is_positive=llm_result.sentiment_score >= 0.7,
                        is_negative=llm_result.sentiment_score
                        < self.sentiment_threshold,
                        sentiment_indicators=llm_result.sentiment_indicators,
                    )

                if 'resolution' in self.enabled_metrics:
                    status = llm_result.conversation_status
                    is_resolved = status in ['approved', 'declined', 'stalemate']
                    resolution = ResolutionSignals(
                        final_status=status,
                        is_resolved=is_resolved,
                        resolution_type=status if is_resolved else None,
                        is_stalemate=status == 'stalemate',
                    )

            except Exception as e:
                reasoning = f'LLM analysis failed: {e}. Using heuristic fallback.'
                # Fallback to heuristics for critical metrics
                if 'frustration' in self.enabled_metrics:
                    frustration = self._heuristic_frustration(human_messages)
                if 'escalation' in self.enabled_metrics:
                    escalation = self._heuristic_escalation(
                        all_messages, unique_mentions
                    )

        # Build result
        result = SlackAnalysisResult(
            thread_id=metadata.thread_ts,
            channel_id=metadata.channel_id,
            sender=metadata.sender,
            case_id=case_id,
            interaction=interaction,
            engagement=engagement,
            recommendation=recommendation,
            escalation=escalation,
            frustration=frustration,
            acceptance=acceptance,
            override=override,
            satisfaction=satisfaction,
            intervention=intervention,
            sentiment=sentiment,
            resolution=resolution,
            enabled_metrics=list(self.enabled_metrics),
            llm_analysis_performed=llm_performed,
            analysis_reasoning=reasoning,
        )

        # Build explanation
        parts = [f'Analyzed {total_count} messages.']
        if interaction.is_interactive:
            parts.append('Interactive.')
        if escalation and escalation.is_escalated:
            parts.append(f'Escalated: {escalation.escalation_type}.')
        if frustration and frustration.is_frustrated:
            parts.append('Frustrated.')
        if intervention and intervention.has_intervention:
            parts.append(f'Intervention: {intervention.intervention_type}.')
        if resolution and resolution.is_resolved:
            parts.append(f'Resolved: {resolution.final_status}.')

        return MetricEvaluationResult(
            score=None,
            explanation=' '.join(parts),
            signals=result,
            metadata={'slack_analysis_result': result.model_dump()},
        )

    def _classify_intervention_escalation(self, intervention_type: str) -> tuple:
        """Map intervention type to escalation category."""
        HARD = {'correction_factual', 'tech_issue', 'data_quality'}
        SOFT = {'missing_context', 'risk_appetite', 'clarification', 'support'}
        AUTHORITY = {'approval'}

        if intervention_type in HARD:
            return 'hard', False
        elif intervention_type in SOFT:
            return 'soft', False
        elif intervention_type in AUTHORITY:
            return 'authority', False
        else:
            return 'none', True

    def _calculate_interaction_depth(self, messages: list) -> int:
        """Calculate number of back-and-forth exchanges."""
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
        import re

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

    def _heuristic_escalation(
        self, messages: list, mentions: List[str]
    ) -> EscalationSignals:
        """Heuristic fallback for escalation."""
        import re

        is_escalated = False
        escalation_type = 'no_escalation'
        escalation_turn = None
        escalation_reason = ''

        ai_seen = False
        for idx, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                ai_seen = True
                if msg.content:
                    for pattern in [
                        r'apologize.*error',
                        r'encountered.*(?:error|issue)',
                        r'unable\s+to',
                    ]:
                        if re.search(pattern, msg.content, re.IGNORECASE):
                            is_escalated = True
                            escalation_type = 'error_escalation'
                            escalation_turn = idx
                            escalation_reason = 'AI encountered an error'
                            break
            elif ai_seen and isinstance(msg, HumanMessage) and msg.content:
                if extract_mentions(msg.content):
                    is_escalated = True
                    escalation_type = 'team_mention'
                    escalation_turn = idx
                    escalation_reason = 'User @mentioned team members'
                    break
            if is_escalated:
                break

        return EscalationSignals(
            is_escalated=is_escalated,
            escalation_type=escalation_type,
            escalation_turn_index=escalation_turn,
            escalation_targets=mentions if is_escalated else [],
            escalation_reason=escalation_reason,
        )

    def _heuristic_frustration(self, human_messages: list) -> FrustrationSignals:
        """Heuristic fallback for frustration."""
        import re

        indicators = []
        scores = []
        peak_turn = None
        peak_score = 0.0

        for idx, msg in enumerate(human_messages):
            if not msg.content:
                continue
            text = msg.content
            turn_score = 0.0
            turn_indicators = []

            if re.search(r'\?\?+', text):
                turn_score += 0.2
                turn_indicators.append('multiple question marks')
            if re.search(r'!!+', text):
                turn_score += 0.2
                turn_indicators.append('multiple exclamation marks')
            caps = len(re.findall(r'\b[A-Z]{3,}\b', text))
            if caps > 0:
                turn_score += min(0.3, caps * 0.1)
                turn_indicators.append(f'{caps} ALL CAPS')

            for pattern, indicator in [
                (r'frustrat', 'frustration'),
                (r"doesn'?t?\s+work", 'not working'),
            ]:
                if re.search(pattern, text, re.IGNORECASE):
                    turn_score += 0.2
                    turn_indicators.append(indicator)

            if turn_indicators:
                indicators.extend([f'Turn {idx}: {i}' for i in turn_indicators])
                scores.append(turn_score)
                if turn_score > peak_score:
                    peak_score = turn_score
                    peak_turn = idx

        overall = min(1.0, sum(scores) / max(len(human_messages), 1)) if scores else 0.0

        return FrustrationSignals(
            frustration_score=round(overall, 2),
            is_frustrated=overall >= self.frustration_threshold,
            frustration_indicators=indicators[:10],
            peak_frustration_turn=peak_turn,
            frustration_cause='other' if indicators else 'none',
        )

    def get_signals(self, result: SlackAnalysisResult) -> List[SignalDescriptor]:
        """Generate signal descriptors for all metrics."""
        # Always include base descriptors for the original 8 metrics
        descriptors = [
            SignalDescriptor(name='thread_id', extractor=lambda r: r.thread_id),
            SignalDescriptor(
                name='is_interactive',
                extractor=lambda r: r.interaction.is_interactive,
                headline_display=True,
            ),
            SignalDescriptor(
                name='is_escalated',
                extractor=lambda r: r.escalation.is_escalated if r.escalation else None,
                headline_display=True,
            ),
            SignalDescriptor(
                name='is_frustrated',
                extractor=lambda r: r.frustration.is_frustrated
                if r.frustration
                else None,
                headline_display=True,
            ),
            SignalDescriptor(
                name='has_recommendation',
                extractor=lambda r: r.recommendation.has_recommendation,
                headline_display=True,
            ),
            SignalDescriptor(
                name='is_accepted',
                extractor=lambda r: r.acceptance.is_accepted if r.acceptance else None,
                headline_display=True,
            ),
            SignalDescriptor(
                name='is_overridden',
                extractor=lambda r: r.override.is_overridden if r.override else None,
                headline_display=True,
            ),
        ]

        # Add feedback metric descriptors if present
        if result.intervention:
            descriptors.extend(
                [
                    SignalDescriptor(
                        name='has_intervention',
                        extractor=lambda r: r.intervention.has_intervention
                        if r.intervention
                        else None,
                        headline_display=True,
                    ),
                    SignalDescriptor(
                        name='intervention_type',
                        extractor=lambda r: r.intervention.intervention_type
                        if r.intervention
                        else None,
                    ),
                    SignalDescriptor(
                        name='is_stp',
                        extractor=lambda r: r.intervention.is_stp
                        if r.intervention
                        else None,
                        headline_display=True,
                    ),
                ]
            )
        if result.sentiment:
            descriptors.append(
                SignalDescriptor(
                    name='sentiment',
                    extractor=lambda r: r.sentiment.sentiment if r.sentiment else None,
                    headline_display=True,
                )
            )
        if result.resolution:
            descriptors.extend(
                [
                    SignalDescriptor(
                        name='is_resolved',
                        extractor=lambda r: r.resolution.is_resolved
                        if r.resolution
                        else None,
                        headline_display=True,
                    ),
                    SignalDescriptor(
                        name='final_status',
                        extractor=lambda r: r.resolution.final_status
                        if r.resolution
                        else None,
                    ),
                ]
            )

        return descriptors

    @staticmethod
    def expand_results(results) -> pd.DataFrame:
        """Expand results into rows per metric, matching to_dataframe() format."""
        if not hasattr(results, 'results'):
            raise TypeError(f'Expected EvaluationResult, got {type(results)}')

        all_rows = []
        for test_result in results.results:
            test_case_id = getattr(test_result.test_case, 'id', None)

            for score_result in test_result.score_results:
                if score_result.name != 'Slack Conversation Analyzer':
                    continue

                if (
                    not score_result.metadata
                    or 'slack_analysis_result' not in score_result.metadata
                ):
                    continue

                result_obj = SlackAnalysisResult(
                    **score_result.metadata['slack_analysis_result']
                )

                for metric_name in result_obj._get_active_metrics():
                    signals = getattr(result_obj, metric_name, None)
                    if signals is None:
                        continue

                    signals_dict = signals.model_dump()

                    # Extract primary score based on metric type
                    metric_score = None
                    if metric_name == 'frustration' and hasattr(
                        signals, 'frustration_score'
                    ):
                        metric_score = signals.frustration_score
                    elif metric_name == 'satisfaction' and hasattr(
                        signals, 'satisfaction_score'
                    ):
                        metric_score = signals.satisfaction_score
                    elif metric_name == 'sentiment' and hasattr(
                        signals, 'sentiment_score'
                    ):
                        metric_score = signals.sentiment_score

                    row = {
                        'id': test_case_id,
                        'metric_name': f'slack_{metric_name}',
                        'metric_score': metric_score,
                        'metric_type': 'analysis',
                        'threshold': None,
                        'passed': None,
                        'explanation': None,
                        'signals': signals_dict,
                        'metadata': {
                            'thread_id': result_obj.thread_id,
                            'channel_id': result_obj.channel_id,
                            'sender': result_obj.sender,
                        },
                        'run_id': results.run_id,
                        'evaluation_name': results.evaluation_name,
                    }
                    all_rows.append(row)

        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    @staticmethod
    def results_to_kpi_dataframe(results) -> pd.DataFrame:
        """Convert results to KPI summary DataFrame."""
        if not hasattr(results, 'results'):
            raise TypeError(f'Expected EvaluationResult, got {type(results)}')

        all_rows = []
        for test_result in results.results:
            test_case_id = getattr(test_result.test_case, 'id', None)

            for score_result in test_result.score_results:
                if score_result.name != 'Slack Conversation Analyzer':
                    continue

                if (
                    not score_result.metadata
                    or 'slack_analysis_result' not in score_result.metadata
                ):
                    continue

                result_obj = SlackAnalysisResult(
                    **score_result.metadata['slack_analysis_result']
                )
                kpi_row = result_obj.to_kpi_summary()
                kpi_row['id'] = test_case_id
                all_rows.append(kpi_row)

        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
