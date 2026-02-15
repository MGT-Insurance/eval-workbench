from typing import Any, Dict, List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion._handlers.llm.handler import LLMHandler
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SubMetricResult
from pydantic import Field

from eval_workbench.shared.metrics.slack.config import AnalyzerConfig
from eval_workbench.shared.metrics.slack.utils import (
    build_transcript,
    get_human_messages,
)


class SubjectiveAnalysisInput(RichBaseModel):
    """Input for subjective analysis."""

    conversation_transcript: str = Field(description='Full conversation transcript')
    has_recommendation: bool = Field(
        description='Whether a recommendation was detected'
    )
    recommendation_type: str = Field(description='Type of recommendation if found')
    recommendation_turn: Optional[int] = Field(
        description='Turn where recommendation was made'
    )
    human_message_count: int = Field(description='Number of human messages')
    truncation_summary: str = Field(
        default='',
        description='Summary of any truncation applied',
    )
    bot_name: str = Field(default='Athena', description='Name of the AI assistant')
    domain_context: str = Field(
        default='insurance underwriting',
        description='Domain context for the conversation',
    )

    # Context from objective analysis (Pass 1)
    objective_is_escalated: bool = Field(
        default=False,
        description='Escalation status from objective analysis',
    )
    objective_has_intervention: bool = Field(
        default=False,
        description='Intervention status from objective analysis',
    )
    objective_intervention_type: str = Field(
        default='no_intervention',
        description='Intervention type from objective analysis',
    )
    objective_final_status: str = Field(
        default='pending',
        description='Resolution status from objective analysis',
    )


class SubjectiveAnalysisOutput(RichBaseModel):
    """Output for subjective analysis - sentiment and quality metrics."""

    # Analysis applicability/state
    is_applicable: bool = Field(
        default=True,
        description='Whether subjective analysis is applicable for this conversation',
    )
    analysis_status: Literal['analyzed', 'skipped_no_human'] = Field(
        default='analyzed',
        description='Execution status for subjective analysis',
    )

    # Sentiment (classification only)
    sentiment: Literal['positive', 'neutral', 'frustrated', 'confused'] = Field(
        default='neutral',
        description='Overall user sentiment throughout conversation',
    )
    sentiment_trajectory: Literal['improving', 'stable', 'worsening'] = Field(
        default='stable',
        description='How sentiment changed over the conversation',
    )
    sentiment_indicators: List[str] = Field(
        default_factory=list,
        description='Specific phrases/behaviors indicating sentiment',
    )

    # Frustration (classification only)
    frustration_cause: Literal[
        'none',
        'ai_error',  # Bot hallucinated or gave wrong info
        'data_quality',  # Magic Dust/3rd party data was wrong
        'tooling_friction',  # Platform (Socotra/SFX) bugs or UI issues
        'rule_rigidity',  # Hard blocks that user disagrees with
        'slow_response',  # Latency issues
        'other',
    ] = Field(
        default='none',
        description='Primary cause of frustration if present',
    )
    frustration_indicators: List[str] = Field(
        default_factory=list,
        description='Specific frustration indicators found',
    )
    peak_frustration_turn: Optional[int] = Field(
        default=None,
        description='Turn index where frustration peaked',
    )

    # Satisfaction (for override quality)
    satisfaction_score: Optional[float] = Field(
        default=None,
        description='Override explanation quality (if applicable). Value between 0.0 and 1.0.',
    )
    has_clear_reason: bool = Field(
        default=False,
        description='Whether override has a clear reason',
    )
    has_supporting_evidence: bool = Field(
        default=False,
        description='Whether override has supporting evidence',
    )
    is_actionable: bool = Field(
        default=False,
        description='Whether feedback is actionable for improvement',
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description='Suggested improvements based on conversation',
    )

    # Acceptance (for recommendations)
    acceptance_status: Literal[
        'accepted',
        'accepted_with_discussion',
        'pending',
        'rejected',
        'modified',
    ] = Field(
        default='pending',
        description='Whether recommendation was accepted',
    )
    is_accepted: bool = Field(
        default=False,
        description='Boolean for whether recommendation was accepted',
    )
    acceptance_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where acceptance/rejection occurred',
    )
    decision_maker: Optional[str] = Field(
        default=None,
        description='Who made the final decision',
    )

    # Override
    is_overridden: bool = Field(
        default=False,
        description='Whether the recommendation was overridden',
    )
    override_type: Literal[
        'no_override',
        'full_override',
        'partial_override',
        'pending_override',
    ] = Field(
        default='no_override',
        description='Type of override if present',
    )
    final_decision: Optional[str] = Field(
        default=None,
        description='The final decision made',
    )
    override_reason: str = Field(
        default='',
        description='Reason for the override',
    )
    override_reason_category: Literal[
        'none',
        'additional_info',
        'risk_assessment',
        'policy_exception',
        'class_code_issue',
        'rate_issue',
        'experience_judgment',
        'other',
    ] = Field(
        default='none',
        description='Category of override reason',
    )

    # Chain-of-Thought
    reasoning_trace: str = Field(
        default='',
        description='Step-by-step reasoning for assessments',
    )


@metric(
    name='Slack Subjective Analyzer',
    key='slack_subjective_analyzer',
    description='Subjective sentiment and quality analysis of Slack conversations.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=0.5,
    score_range=(0.0, 1.0),
    tags=['slack', 'multi_turn'],
)
class SlackSubjectiveAnalyzer(
    BaseMetric[SubjectiveAnalysisInput, SubjectiveAnalysisOutput]
):
    """
    Subjective Analyzer: Nuanced sentiment and quality assessment.

    Uses standard temperature (0.3) for nuanced interpretation.
    Can receive context from objective analysis to inform assessment.
    Focuses on subjective qualities:
    - User sentiment towards the Bot vs. the System
    - Frustration causes (Data, Tooling, Rules)
    - Recommendation acceptance and overrides
    """

    instruction = """You are an Underwriting Experience Analyst for an AI Assistant.

## YOUR TASK
Analyze the conversation for subjective qualities using the context provided in the input.

Crucially, distinguish between frustration with **Athena (the Bot)** vs frustration with **The Platform (Socotra/SFX/Magic Dust)**.

---

## FRUSTRATION CAUSES

**tooling_friction** - Platform Bugs/UX:
- User is annoyed by the software, not the bot's logic.
- Evidence: "Failed to decline", "Button not working", "Can't override in SFX".

**data_quality** - Bad Inputs:
- User is annoyed that pre-filled data is wrong.
- Evidence: "Magic Dust is wrong again", "Why does it say 1978?".

**rule_rigidity** - Policy Blocks:
- User is annoyed by a hard decline rule they disagree with.
- Evidence: "This shouldn't be blocked", "Why is coastal coverage restricted?".

**ai_error** - Bot Logic:
- The bot misunderstood the prompt or gave a bad answer.
- Evidence: "You missed the payroll cap", "That's not what I asked".

---

## ACCEPTANCE & OVERRIDE ANALYSIS

**Acceptance Status** (for recommendations):
- `accepted`: User explicitly agreed without changes
- `accepted_with_discussion`: Agreed after discussion/clarification
- `pending`: No clear decision made
- `rejected`: User explicitly declined
- `modified`: User accepted with modifications

**Override** (human changed bot's recommendation):
- `no_override`: Final decision matches recommendation
- `full_override`: Decision completely opposite to recommendation
- `partial_override`: Decision partially differs from recommendation
- `pending_override`: Override being discussed but not finalized

**Override Reason Categories**:
- `additional_info`: Human had information bot didn't have
- `risk_assessment`: Different risk judgment
- `policy_exception`: Applying policy exception
- `class_code_issue`: Class code disagreement
- `rate_issue`: Rate/pricing disagreement
- `experience_judgment`: Professional experience override

---

## OUTPUT FORMAT
Provide your reasoning trace first, walking through the conversation chronologically.
Note specific turns and quotes that inform your assessments."""

    input_model = SubjectiveAnalysisInput
    output_model = SubjectiveAnalysisOutput
    description = 'Subjective sentiment and quality analysis'

    examples = [
        (
            SubjectiveAnalysisInput(
                conversation_transcript="""
[Turn 0] Athena: Status: Approved.
[Turn 1] User: Taylor reached out because this condo is owner occupied. I will have to change to decline.
[Turn 2] User: I tried that and got "failed to decline".
[Turn 3] User: I can't do it in SFX without editing something to make it not auto approve.
""",
                has_recommendation=True,
                recommendation_type='approve',
                recommendation_turn=0,
                human_message_count=3,
                bot_name='Athena',
                domain_context='insurance underwriting',
                objective_is_escalated=False,
                objective_has_intervention=True,
                objective_intervention_type='system_workaround',
                objective_final_status='declined',
            ),
            SubjectiveAnalysisOutput(
                sentiment='frustrated',
                sentiment_trajectory='worsening',
                sentiment_indicators=['"failed to decline"', '"can\'t do it in SFX"'],
                frustration_cause='tooling_friction',
                frustration_indicators=['failed to decline', "can't do it in SFX"],
                peak_frustration_turn=2,
                acceptance_status='rejected',
                is_accepted=False,
                is_overridden=True,
                override_type='full_override',
                final_decision='decline',
                override_reason='Owner occupied condo requiring decline, but system blocked the action.',
                override_reason_category='additional_info',
                reasoning_trace='User wanted to decline but was blocked by a system error ("failed to decline"). Frustration is directed at the tool (SFX), not the bot.',
            ),
        ),
        (
            SubjectiveAnalysisInput(
                conversation_transcript="""
[Turn 0] Athena: Blocked. Referrals: property_rate_low_Refer.
[Turn 1] User: So do I understand correct that tier 1 rule is now off? Is there an easy way for me to see if a risk is tier 1?
[Turn 2] User: I haven't seen AAL come thru at all today.
[Turn 3] User: AAL is broken again.
""",
                has_recommendation=False,
                recommendation_type='refer',
                recommendation_turn=0,
                human_message_count=3,
                bot_name='Athena',
                domain_context='insurance underwriting',
                objective_is_escalated=False,
                objective_has_intervention=True,
                objective_intervention_type='system_workaround',
                objective_final_status='pending',
            ),
            SubjectiveAnalysisOutput(
                sentiment='frustrated',
                sentiment_trajectory='stable',
                sentiment_indicators=['"broken again"', '"haven\'t seen AAL"'],
                frustration_cause='tooling_friction',
                frustration_indicators=['AAL is broken again'],
                peak_frustration_turn=3,
                acceptance_status='pending',
                is_accepted=False,
                is_overridden=False,
                override_type='no_override',
                final_decision=None,
                override_reason='',
                override_reason_category='none',
                reasoning_trace='User is frustrated by repeated service failure ("AAL broken again"). This is a platform stability issue.',
            ),
        ),
    ]

    is_multi_metric = True
    include_parent_score = False
    sub_metric_prefix = False

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        objective_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the SlackSubjectiveAnalyzer.

        Args:
            config: Optional analyzer configuration for bot name and domain context
            objective_context: Optional context from objective analysis
            **kwargs: Additional arguments passed to BaseMetric
        """
        kwargs['temperature'] = 0.3
        super().__init__(**kwargs)
        self.analyzer_config = config or AnalyzerConfig()
        self.objective_context = objective_context or {}

    @trace(name='SlackSubjectiveAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute subjective analysis."""
        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=SubjectiveAnalysisOutput(),
            )

        human_messages = get_human_messages(item.conversation)
        additional = item.additional_input or {}

        # Prepare transcript
        transcript = build_transcript(item.conversation)

        # Get objective context from kwargs or stored context
        obj_ctx = kwargs.get('objective_context', self.objective_context)

        # Build input for LLM
        input_data = SubjectiveAnalysisInput(
            conversation_transcript=transcript,
            has_recommendation=additional.get('has_recommendation', False),
            recommendation_type=additional.get('recommendation_type', 'none'),
            recommendation_turn=additional.get('recommendation_turn'),
            human_message_count=len(human_messages),
            truncation_summary=additional.get('truncation_summary', ''),
            bot_name=self.analyzer_config.bot_name,
            domain_context=self.analyzer_config.domain_context,
            objective_is_escalated=obj_ctx.get('is_escalated', False),
            objective_has_intervention=obj_ctx.get('has_intervention', False),
            objective_intervention_type=obj_ctx.get(
                'intervention_type', 'no_intervention'
            ),
            objective_final_status=obj_ctx.get('final_status', 'pending'),
        )

        try:
            # Call LLMHandler.execute() directly (not BaseMetric.execute() which expects DatasetItem)
            # This handles LLM call, retries, tracing, and cost tracking
            output = await LLMHandler.execute(self, input_data)

            return MetricEvaluationResult(
                score=None,
                explanation=f'Sentiment: {output.sentiment}, Frustration cause: {output.frustration_cause}',
                signals=output,
                metadata={'subjective_result': output.model_dump()},
            )

        except Exception as e:
            return MetricEvaluationResult(
                score=None,
                explanation=f'Analysis failed: {str(e)}',
                signals=SubjectiveAnalysisOutput(
                    reasoning_trace=f'Error during analysis: {str(e)}'
                ),
            )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """Extract sub-metrics from the analysis result."""
        signals: SubjectiveAnalysisOutput = result.signals
        if not signals:
            return []

        sub_metrics = [
            # Sentiment category
            SubMetricResult(
                name='sentiment_category',
                score=None,
                explanation='none' if not signals.is_applicable else signals.sentiment,
                metric_category=MetricCategory.CLASSIFICATION,
                group='subjective',
                metadata={
                    'sentiment': signals.sentiment,
                    'is_applicable': signals.is_applicable,
                    'analysis_status': signals.analysis_status,
                    'trajectory': signals.sentiment_trajectory,
                    'indicators': signals.sentiment_indicators,
                },
            ),
            # Frustration cause
            SubMetricResult(
                name='frustration_cause',
                score=None,
                explanation=signals.frustration_cause,
                metric_category=MetricCategory.CLASSIFICATION,
                group='subjective',
                metadata={
                    'cause': signals.frustration_cause,
                    'peak_turn': signals.peak_frustration_turn,
                    'indicators': signals.frustration_indicators,
                },
            ),
            # Acceptance - classification
            SubMetricResult(
                name='acceptance_status',
                score=None,
                explanation=signals.acceptance_status,
                metric_category=MetricCategory.CLASSIFICATION,
                group='subjective',
                metadata={
                    'status': signals.acceptance_status,
                    'turn_index': signals.acceptance_turn_index,
                    'decision_maker': signals.decision_maker,
                },
            ),
            # Override - classification
            SubMetricResult(
                name='override_type',
                score=None,
                explanation=signals.override_type,
                metric_category=MetricCategory.CLASSIFICATION,
                group='subjective',
                metadata={
                    'is_overridden': signals.is_overridden,
                    'type': signals.override_type,
                    'reason': signals.override_reason,
                    'reason_category': signals.override_reason_category,
                    'final_decision': signals.final_decision,
                },
            ),
        ]

        if signals.satisfaction_score is not None:
            sub_metrics.append(
                SubMetricResult(
                    name='satisfaction_score',
                    score=signals.satisfaction_score,
                    explanation=f'Score: {signals.satisfaction_score:.2f}',
                    metric_category=MetricCategory.SCORE,
                    group='subjective',
                    metadata={
                        'has_clear_reason': signals.has_clear_reason,
                        'has_supporting_evidence': signals.has_supporting_evidence,
                        'is_actionable': signals.is_actionable,
                        'suggestions': signals.improvement_suggestions,
                    },
                )
            )

        # Distribute cost across sub-metrics (normalized per metric)
        total_cost = getattr(self, 'cost_estimate', None)
        if total_cost and len(sub_metrics) > 0:
            cost_per_metric = total_cost / len(sub_metrics)
            for sub in sub_metrics:
                sub.metadata['cost_estimate'] = cost_per_metric

        return sub_metrics


__all__ = [
    'SlackSubjectiveAnalyzer',
    'SubjectiveAnalysisInput',
    'SubjectiveAnalysisOutput',
]
