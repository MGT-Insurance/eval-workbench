import asyncio
from typing import Any, Dict, List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion._handlers.llm.handler import LLMHandler
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SubMetricResult
from pydantic import Field

from eval_workbench.shared.metrics.slack.config import (
    AnalyzerConfig,
    resolve_analyzer_config,
)
from eval_workbench.shared.metrics.slack.utils import (
    build_transcript,
    get_human_messages,
)


class ProductSignalsInput(RichBaseModel):
    """Input for product signals analysis."""

    conversation_transcript: str = Field(description='Full conversation transcript')
    has_recommendation: bool = Field(
        description='Whether a recommendation was detected'
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

    # Context from other analyses
    has_intervention: bool = Field(
        default=False,
        description='Whether human intervention occurred',
    )
    intervention_type: str = Field(
        default='no_intervention',
        description='Type of intervention if present',
    )
    is_frustrated: bool = Field(
        default=False,
        description='Whether user showed frustration',
    )
    sentiment: str = Field(
        default='neutral',
        description='Overall user sentiment',
    )


class ProductSignalsOutput(RichBaseModel):
    """Output for product signals - actionable insights for daily report."""

    learnings: List[str] = Field(
        default_factory=list,
        description='Key learnings from this conversation',
    )
    learning_categories: List[
        Literal[
            'ux',
            'accuracy',
            'coverage',
            'speed',
            'workflow',  # New: Process friction (e.g. "Can't decline")
            'rules',  # New: Underwriting logic (e.g. "Don't ask X")
            'guardrails',  # New: Prevention (e.g. "Block agents from Y")
            'other',
        ]
    ] = Field(
        default_factory=list,
        description='Categories for each learning',
    )
    feature_requests: List[str] = Field(
        default_factory=list,
        description='Explicit or implicit feature requests',
    )
    has_actionable_feedback: bool = Field(
        default=False,
        description='Whether there is actionable feedback for product team',
    )
    priority_level: Literal['high', 'medium', 'low', 'none'] = Field(
        default='none',
        description='Priority level for product attention',
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description='Suggested action for product team',
    )
    reasoning_trace: str = Field(
        default='',
        description='Reasoning for extracted insights',
    )


@metric(
    name='Slack Product Analyzer',
    key='slack_product_analyzer',
    description='Extract actionable product insights from Slack conversations.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'multi_turn'],
)
class SlackProductAnalyzer(BaseMetric[ProductSignalsInput, ProductSignalsOutput]):
    """
    Product Signals Analyzer: Extract actionable product insights.

    Analyzes conversations to identify:
    - UX issues (confusing interface, unclear prompts)
    - Accuracy issues (wrong data, incorrect recommendations)
    - Workflow Friction (Platform limitations in Socotra/SFX)
    - Rule Configuration (Requests to change underwriting logic)
    - Guardrails (Requests to block specific behaviors)

    Used to generate daily product improvement reports.
    """

    instruction = """You are an Insurance Platform Product Manager extracting insights from Underwriter conversations.

## YOUR TASK
Analyze the conversation to identify improvements for the **AI Assistant ({bot_name})** and the **Underwriting Platform (SFX/Socotra)**.

---

## SIGNAL CATEGORIES

**workflow** - Platform Friction:
- User is blocked from performing an action in the UI.
- Issues with buttons, status transitions, or "Failed to..." errors.
- Example: "I can't decline an auto-approved quote without resetting it."

**rules** - Underwriting Logic Configuration:
- Feedback on the business rules, referrals, or questions asked.
- Requests to change *when* a referral triggers.
- Example: "Stop asking contractors if they are home-based."

**guardrails** - Safety/Validation:
- Requests to prevent agents from submitting invalid risks upfront.
- Example: "Block agents from quoting building coverage in coastal zones."

**accuracy** - Data Quality:
- Feedback on the correctness of 3rd party data (Magic Dust).
- Example: "Magic Dust square footage is always off."

**ux** - Interface/Clarity:
- Confusing messages or lack of visibility.
- Example: "Where can I see the Tier 1 county status?"

---

## PRIORITY LEVELS
- `high`: User cannot complete task, or requests a "Bug Fix" / "Ticket".
- `medium`: User suggests an improvement ("It would be nice if...").
- `low`: General observation or complaint without specific suggestion.

---

## OUTPUT FORMAT
Extract clear, actionable learnings. If a user explicitly asks for a ticket/fix, mark `has_actionable_feedback` as True."""

    input_model = ProductSignalsInput
    output_model = ProductSignalsOutput
    description = 'Extract actionable product insights'

    examples = [
        (
            ProductSignalsInput(
                conversation_transcript="""
[Turn 0] User: most contractors will be home based, and this isn't a question we should even be asking them.
[Turn 1] User: create a ticket in the Shadowfax project... remove home-based question for contractors.
""",
                has_recommendation=True,
                human_message_count=2,
                bot_name='Athena',
                domain_context='insurance underwriting',
                has_intervention=True,
                intervention_type='risk_appetite',
                is_frustrated=False,
                sentiment='neutral',
            ),
            ProductSignalsOutput(
                learnings=[
                    'Home-based business question is irrelevant for contractor class codes',
                ],
                learning_categories=['rules'],
                feature_requests=[
                    'Remove home-based referral rule for contractors',
                ],
                has_actionable_feedback=True,
                priority_level='high',
                suggested_action='Update referral logic to skip home-based check for contractor industries',
                reasoning_trace='User explicitly requested a ticket to remove a specific underwriting question/rule for contractors.',
            ),
        ),
        (
            ProductSignalsInput(
                conversation_transcript="""
[Turn 0] User: I would hope that SFX will block agents from quoting building coverage if in a coastal area that does not allow it.
[Turn 1] User: It doesn't right now.
""",
                has_recommendation=False,
                human_message_count=2,
                bot_name='Athena',
                domain_context='insurance underwriting',
                has_intervention=False,
                intervention_type='no_intervention',
                is_frustrated=True,
                sentiment='frustrated',
            ),
            ProductSignalsOutput(
                learnings=[
                    'Agents are currently able to quote building coverage in ineligible coastal zones',
                ],
                learning_categories=['guardrails'],
                feature_requests=[
                    'Implement frontend block for building coverage in coastal zones',
                ],
                has_actionable_feedback=True,
                priority_level='medium',
                suggested_action='Add validation rule in SFX to prevent building coverage selection based on coastal distance',
                reasoning_trace='User expressed desire for system to prevent invalid quotes upfront (guardrails).',
            ),
        ),
        (
            ProductSignalsInput(
                conversation_transcript="""
[Turn 0] User: what is the best way to decline an auto-approved quoted? I can't do it in SFX without editing something.
[Turn 1] User: I tried that and got 'failed to decline'.
""",
                has_recommendation=True,
                human_message_count=2,
                bot_name='Athena',
                domain_context='insurance underwriting',
                has_intervention=True,
                intervention_type='system_issue',
                is_frustrated=True,
                sentiment='frustrated',
            ),
            ProductSignalsOutput(
                learnings=[
                    'Users cannot easily decline quotes that were auto-approved by the system',
                    'SFX throws "failed to decline" error on valid decline attempts',
                ],
                learning_categories=['workflow', 'workflow'],
                feature_requests=[
                    'Enable "Decline" action for auto-approved quotes without requiring reset',
                ],
                has_actionable_feedback=True,
                priority_level='high',
                suggested_action='Fix bug preventing decline of approved quotes and improve workflow UI',
                reasoning_trace='User reported a direct blocker/bug in the workflow ("failed to decline") preventing them from doing their job.',
            ),
        ),
    ]

    is_multi_metric = True
    include_parent_score = False
    sub_metric_prefix = False

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the SlackProductAnalyzer.

        Args:
            config: Optional analyzer configuration
            analysis_context: Optional context from other analyses
            **kwargs: Additional arguments passed to BaseMetric
        """
        kwargs.setdefault('temperature', 0.3)
        super().__init__(**kwargs)
        self.analyzer_config = resolve_analyzer_config(config)
        self.analysis_context = analysis_context or {}
        self._instruction_lock = asyncio.Lock()

    @trace(name='SlackProductAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute product signals analysis."""
        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=ProductSignalsOutput(),
            )

        human_messages = get_human_messages(item.conversation)
        additional = item.additional_input or {}

        # Prepare transcript
        transcript = build_transcript(item.conversation)

        # Get analysis context from kwargs or stored context
        ctx = kwargs.get('analysis_context', self.analysis_context)

        # Build input for LLM
        input_data = ProductSignalsInput(
            conversation_transcript=transcript,
            has_recommendation=additional.get('has_recommendation', False),
            human_message_count=len(human_messages),
            truncation_summary=additional.get('truncation_summary', ''),
            bot_name=self.analyzer_config.bot_name,
            domain_context=self.analyzer_config.domain_context,
            has_intervention=ctx.get('has_intervention', False),
            intervention_type=ctx.get('intervention_type', 'no_intervention'),
            is_frustrated=ctx.get('frustration_score', 0) > 0.5,
            sentiment=ctx.get('sentiment', 'neutral'),
        )

        try:
            # LLMHandler does not auto-format instruction placeholders.
            # Temporarily render dynamic prompt variables for this call only.
            formatted_instruction = self.instruction.format(
                bot_name=self.analyzer_config.bot_name
            )
            async with self._instruction_lock:
                original_instruction = self.instruction
                self.instruction = formatted_instruction
                try:
                    # Call LLMHandler.execute() directly (not BaseMetric.execute() which expects DatasetItem)
                    # This handles LLM call, retries, tracing, and cost tracking
                    output = await LLMHandler.execute(self, input_data)
                finally:
                    self.instruction = original_instruction

            return MetricEvaluationResult(
                score=None,
                explanation=f'Found {len(output.learnings)} learnings, {len(output.feature_requests)} feature requests. Priority: {output.priority_level}',
                signals=output,
                metadata={'product_result': output.model_dump()},
            )

        except Exception as e:
            return MetricEvaluationResult(
                score=None,
                explanation=f'Analysis failed: {str(e)}',
                signals=ProductSignalsOutput(
                    reasoning_trace=f'Error during analysis: {str(e)}'
                ),
            )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """Extract sub-metrics from the analysis result."""
        signals: ProductSignalsOutput = result.signals
        if not signals:
            return []

        sub_metrics = [
            SubMetricResult(
                name='learnings_count',
                score=float(len(signals.learnings)),
                explanation=f'Count: {len(signals.learnings)}',
                metric_category=MetricCategory.SCORE,
                group='product',
                metadata={
                    'learnings': signals.learnings,
                    'categories': signals.learning_categories,
                },
            ),
            SubMetricResult(
                name='feature_requests_count',
                score=float(len(signals.feature_requests)),
                explanation=f'Count: {len(signals.feature_requests)}',
                metric_category=MetricCategory.SCORE,
                group='product',
                metadata={
                    'requests': signals.feature_requests,
                },
            ),
            SubMetricResult(
                name='has_actionable_feedback',
                score=None,
                explanation=str(signals.has_actionable_feedback).lower(),
                metric_category=MetricCategory.CLASSIFICATION,
                group='product',
                metadata={
                    'has_actionable_feedback': signals.has_actionable_feedback,
                    'suggested_action': signals.suggested_action,
                },
            ),
            SubMetricResult(
                name='priority_level',
                score=None,
                explanation=signals.priority_level,
                metric_category=MetricCategory.CLASSIFICATION,
                group='product',
                metadata={
                    'priority_level': signals.priority_level,
                },
            ),
        ]

        # Distribute cost across sub-metrics (normalized per metric)
        total_cost = getattr(self, 'cost_estimate', None)
        if total_cost and len(sub_metrics) > 0:
            cost_per_metric = total_cost / len(sub_metrics)
            for sub in sub_metrics:
                sub.metadata['cost_estimate'] = cost_per_metric

        return sub_metrics


__all__ = [
    'SlackProductAnalyzer',
    'ProductSignalsInput',
    'ProductSignalsOutput',
]
