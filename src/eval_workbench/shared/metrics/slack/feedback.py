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
from eval_workbench.shared.metrics.slack.utils import build_transcript


class FeedbackAttributionInput(RichBaseModel):
    """Input for feedback attribution analysis."""

    conversation_transcript: str = Field(description='Full conversation transcript')
    truncation_summary: str = Field(
        default='',
        description='Summary of any truncation applied',
    )
    bot_name: str = Field(default='Athena', description='Name of the AI assistant')
    domain_context: str = Field(
        default='insurance underwriting',
        description='Domain context for the conversation',
    )

    # Context triggering this analysis
    sentiment_score: float = Field(
        description='Sentiment score from subjective analysis',
    )
    frustration_score: float = Field(
        default=0.0,
        description='Frustration score from subjective analysis',
    )
    frustration_cause: str = Field(
        default='none',
        description='Frustration cause from subjective analysis',
    )
    has_intervention: bool = Field(
        default=False,
        description='Whether intervention occurred',
    )
    intervention_type: str = Field(
        default='no_intervention',
        description='Type of intervention if present',
    )


class FeedbackAttributionOutput(RichBaseModel):
    """Output for feedback attribution - which step failed."""

    has_negative_feedback: bool = Field(
        default=False,
        description='Whether negative feedback was identified',
    )
    # UPDATED: Failure categories specific to Underwriting/Socotra/Magic Dust
    failed_step: Optional[
        Literal[
            'classification_failure',  # Wrong Class Code / NAICS
            'data_integrity_failure',  # Magic Dust data was wrong (Year built, Sq Ft)
            'rule_engine_failure',  # Missed hard rule (Payroll cap, Coastal distance)
            'system_tooling_failure',  # Socotra/Swallow/SFX error
            'chat_interface',  # Hallucination or confusing prompt
            'unknown',
        ]
    ] = Field(
        default=None,
        description='The step in the pipeline that failed',
    )
    failure_evidence: str = Field(
        default='',
        description='Evidence from conversation supporting the attribution',
    )
    confidence: Literal['high', 'medium', 'low'] = Field(
        default='low',
        description='Confidence in the attribution',
    )
    remediation_hint: Optional[str] = Field(
        default=None,
        description='Suggested remediation for the failure',
    )
    reasoning_trace: str = Field(
        default='',
        description='Reasoning for the attribution',
    )


@metric(
    name='Slack Feedback Attribution Analyzer',
    key='slack_feedback_attribution_analyzer',
    description='Identify which pipeline step failed when negative feedback is detected.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'multi_turn'],
)
class SlackFeedbackAttributionAnalyzer(
    BaseMetric[FeedbackAttributionInput, FeedbackAttributionOutput]
):
    """
    Feedback Attribution Analyzer: Identify which step failed.

    Identifies the root cause of friction in the Underwriting process:
    - Classification Failure: Model/Data selected wrong Class Code (e.g. Retail vs Contractor)
    - Data Integrity Failure: Third-party data (Magic Dust) incorrect (Year Built, Sq Ft)
    - Rule Engine Failure: Model missed a hard eligibility rule (Payroll > $300k, State not live)
    - System Tooling Failure: Backend errors (Socotra, Swallow, SFX, Race Conditions)
    """

    instruction = """You are a Lead Underwriting Auditor diagnosing failures in an AI Assistant (Athena).

## YOUR TASK
The user (Underwriter) had friction with the AI. Identify the ROOT CAUSE of the failure based on the transcript.

Distinguish between the **AI Model**, the **Data Source (Magic Dust)**, and the **Platform (Socotra/Swallow)**.

---

## FAILURE CATEGORIES

**classification_failure** - Wrong Business Class:
- AI or Data selected the wrong Class Code/NAICS.
- Misunderstanding the business operations (e.g., calling a "Junk Hauler" an "Exterior Cleaner").
- Evidence: "This is misclassified", "Should be [Code X]", "Wrong industry group".

**data_integrity_failure** - Bad Third-Party Data (Magic Dust):
- The AI's logic was fine, but the *input data* was wrong.
- Issues with: Year Built, Square Footage, Employee Count, Payroll figures from 'Magic Dust'.
- Evidence: "Magic Dust shows 1978 but it's new construction", "Sq ft is actually 100k".

**rule_engine_failure** - Missed Hard Rule / Eligibility:
- The AI approved a risk that violated a hard eligibility rule.
- Issues with: Payroll caps ($300k), Coastal distance, Roof age, TIV limits.
- **Territory Issues**: Quoting in states where carrier is not live (CA, NY, FL).
- Evidence: "Payroll > $300k s/b ineligible", "Tier 1 county restriction", "We are not live in CA".

**system_tooling_failure** - Backend/UI/Sync Bugs:
- The Underwriter agrees with the decision but *cannot execute it* due to the tool.
- **Race Conditions**: Bot reports status before backend finishes calculation.
- **Service Failures**: External calls (AAL/Aon, Magic Dust) failing to return data.
- Evidence: "Failed to decline", "Athena posted before quote completed", "AAL is broken again".

**chat_interface** - UX/Hallucination:
- AI was confusing, verbose, or hallucinated a capability it doesn't have.
- Evidence: "You said X but meant Y", "Confusing response".

---

## ATTRIBUTION RULES
1. **Blame the Data, not the Bot**: If the user says "Magic Dust says X but reality is Y", this is `data_integrity_failure`.
2. **Blame the Rule, not the Bot**: If the user says "This should have auto-declined" or "Not live in this state", this is `rule_engine_failure`.
3. **Blame the System**: If the user mentions "Failed to decline", "AAL broken", or timing/sync issues, this is `system_tooling_failure`.

## OUTPUT FORMAT
Identify the most likely failed step and provide direct quote evidence.
"""

    input_model = FeedbackAttributionInput
    output_model = FeedbackAttributionOutput
    description = 'Identify which pipeline step failed'

    examples = [
        (
            FeedbackAttributionInput(
                conversation_transcript="""
[Turn 0] Athena: I recommend APPROVE. Payroll is $380,000.
[Turn 1] User: >$300k payroll s/b ineligible...did it price?
[Turn 2] Athena: You're right. Payroll of $380,000 exceeds the $300,000 limit. I missed that eligibility rule.
""",
                bot_name='Athena',
                domain_context='insurance underwriting',
                sentiment_score=0.2,
                frustration_score=0.6,
                frustration_cause='wrong_answer',
                has_intervention=True,
                intervention_type='correction_factual',
            ),
            FeedbackAttributionOutput(
                has_negative_feedback=True,
                failed_step='rule_engine_failure',
                failure_evidence='User noted "$300k payroll s/b ineligible" when actual was $380k. Bot admitted missing the rule.',
                confidence='high',
                remediation_hint='Update rule engine to hard-block Contractor payroll > $300k.',
                reasoning_trace='The data was correct ($380k), but the AI logic failed to trigger the decline rule.',
            ),
        ),
        (
            FeedbackAttributionInput(
                conversation_transcript="""
[Turn 0] Athena: Status: Approved.
[Turn 1] User: Taylor reached out because this condo is owner occupied. I will have to change to decline.
[Turn 2] User: I tried that and got "failed to decline" in SFX.
[Turn 3] User: I can't do it in SFX without editing something to make it not auto approve.
""",
                bot_name='Athena',
                domain_context='insurance underwriting',
                sentiment_score=0.2,
                frustration_score=0.8,
                frustration_cause='system_issue',
                has_intervention=True,
                intervention_type='system_issue',
            ),
            FeedbackAttributionOutput(
                has_negative_feedback=True,
                failed_step='system_tooling_failure',
                failure_evidence='User stated "failed to decline in SFX" and "I can\'t do it in SFX".',
                confidence='high',
                remediation_hint='Fix "Failed to Decline" bug in Socotra/SFX for auto-approved quotes.',
                reasoning_trace='User wanted to take action but the software tool (SFX) physically prevented them. Not an AI logic error.',
            ),
        ),
        (
            FeedbackAttributionInput(
                conversation_transcript="""
[Turn 0] Athena: Status: approved.
[Turn 1] User: Why is this showing as approved?
[Turn 2] Athena: This quote is showing as declined. The initial workflow message may have shown "approved" before the underwriting rules were fully evaluated.
[Turn 3] User: I think athena is posting before a quote is completed.
""",
                bot_name='Athena',
                domain_context='insurance underwriting',
                sentiment_score=0.2,
                frustration_score=0.6,
                frustration_cause='system_issue',
                has_intervention=True,
                intervention_type='system_issue',
            ),
            FeedbackAttributionOutput(
                has_negative_feedback=True,
                failed_step='system_tooling_failure',
                failure_evidence='User noted "Athena is posting before a quote is completed". Bot admitted display timing issue.',
                confidence='high',
                remediation_hint='Ensure Bot waits for Swallow "final" status event before posting to Slack.',
                reasoning_trace='Latency/Sync issue where the UI (Chat) reported state before the Backend (Swallow) finished processing.',
            ),
        ),
        (
            FeedbackAttributionInput(
                conversation_transcript="""
[Turn 0] Athena: Status: Approved.
[Turn 1] User: This risk should be declined. It's out of appetite.
[Turn 2] User: We're not yet approved in CA.
""",
                bot_name='Athena',
                domain_context='insurance underwriting',
                sentiment_score=0.3,
                frustration_score=0.4,
                frustration_cause='policy_violation',
                has_intervention=True,
                intervention_type='correction_factual',
            ),
            FeedbackAttributionOutput(
                has_negative_feedback=True,
                failed_step='rule_engine_failure',
                failure_evidence='User stated "We\'re not yet approved in CA" after bot approved a CA risk.',
                confidence='high',
                remediation_hint='Add hard state-based eligibility blocks for non-live territories (CA/NY/FL).',
                reasoning_trace='The AI/System failed to block a quote in a state where the carrier has no license/appetite.',
            ),
        ),
        (
            FeedbackAttributionInput(
                conversation_transcript="""
[Turn 0] Athena: Class: CLEANEXTC (Exterior Cleaning). Status: Blocked.
[Turn 1] User: This is United Junk Removal. Junk removal typically falls under waste management/hauling.
[Turn 2] User: Should have been ineligible due to these contractors not having class codes.
""",
                bot_name='Athena',
                domain_context='insurance underwriting',
                sentiment_score=0.4,
                frustration_score=0.5,
                frustration_cause='poor_understanding',
                has_intervention=True,
                intervention_type='correction_class_code',
            ),
            FeedbackAttributionOutput(
                has_negative_feedback=True,
                failed_step='classification_failure',
                failure_evidence='User corrected class from "Exterior Cleaning" to "Junk Removal/Waste Management".',
                confidence='high',
                remediation_hint='Tune embedding model to distinguish between "Cleaning" and "Hauling/Waste".',
                reasoning_trace='The AI/System mapped the business to a cleaning code, but it was actually a waste management risk.',
            ),
        ),
        (
            FeedbackAttributionInput(
                conversation_transcript="""
[Turn 0] Athena: Building Age: 1954 (72 years old).
[Turn 1] User: Customer states 2026 (new construction) but Magic Dust data shows 1954.
[Turn 2] User: I updated sq ft to reflect correct sq ft. Magic Dust said 2,600 but it's actually 1,776.
""",
                bot_name='Athena',
                domain_context='insurance underwriting',
                sentiment_score=0.5,
                frustration_score=0.3,
                frustration_cause='wrong_answer',
                has_intervention=True,
                intervention_type='correction_factual',
            ),
            FeedbackAttributionOutput(
                has_negative_feedback=True,
                failed_step='data_integrity_failure',
                failure_evidence='User noted "Magic Dust data shows 1954" vs actual "2026".',
                confidence='high',
                remediation_hint='Deprioritize Magic Dust for Year Built when "New Construction" flag is present.',
                reasoning_trace='The AI correctly read the data provided, but the third-party data source (Magic Dust) was factually incorrect.',
            ),
        ),
    ]

    is_multi_metric = False
    include_parent_score = False

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
        sentiment_threshold: float = 0.4,
        **kwargs,
    ):
        """
        Initialize the SlackFeedbackAttributionAnalyzer.

        Args:
            config: Optional analyzer configuration
            analysis_context: Optional context from other analyses
            sentiment_threshold: Sentiment score below which to trigger analysis
            **kwargs: Additional arguments passed to BaseMetric
        """
        kwargs.setdefault('temperature', 0.2)
        super().__init__(**kwargs)
        self.analyzer_config = config or AnalyzerConfig()
        self.analysis_context = analysis_context or {}
        self.sentiment_threshold = sentiment_threshold

    @trace(
        name='SlackFeedbackAttributionAnalyzer',
        capture_args=True,
        capture_response=True,
    )
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute feedback attribution analysis."""
        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=FeedbackAttributionOutput(),
            )

        additional = item.additional_input or {}

        # Get analysis context from kwargs or stored context
        ctx = kwargs.get('analysis_context', self.analysis_context)
        sentiment_score = ctx.get('sentiment_score', 0.5)

        # Early exit if sentiment is positive (AND no intervention occurred)
        # UPDATED LOGIC: Even if sentiment is neutral, if there was a major correction, we want to know why.
        has_intervention = ctx.get('has_intervention', False)

        if sentiment_score >= self.sentiment_threshold and not has_intervention:
            return MetricEvaluationResult(
                score=None,
                explanation=f'Sentiment {sentiment_score:.2f} high & no intervention. Skipping.',
                signals=FeedbackAttributionOutput(
                    has_negative_feedback=False,
                    reasoning_trace='No negative feedback or intervention detected.',
                ),
            )

        # Prepare transcript
        transcript = build_transcript(item.conversation)

        # Build input for LLM
        input_data = FeedbackAttributionInput(
            conversation_transcript=transcript,
            truncation_summary=additional.get('truncation_summary', ''),
            bot_name=self.analyzer_config.bot_name,
            domain_context=self.analyzer_config.domain_context,
            sentiment_score=sentiment_score,
            frustration_score=ctx.get('frustration_score', 0.0),
            frustration_cause=ctx.get('frustration_cause', 'none'),
            has_intervention=has_intervention,
            intervention_type=ctx.get('intervention_type', 'no_intervention'),
        )

        try:
            # Call LLMHandler.execute() directly
            output = await LLMHandler.execute(self, input_data)
            output.has_negative_feedback = True

            return MetricEvaluationResult(
                score=None,
                explanation=f'Failed step: {output.failed_step} (confidence: {output.confidence})',
                signals=output,
                metadata={'feedback_attribution_result': output.model_dump()},
            )

        except Exception as e:
            return MetricEvaluationResult(
                score=None,
                explanation=f'Analysis failed: {str(e)}',
                signals=FeedbackAttributionOutput(
                    has_negative_feedback=True,
                    failed_step='unknown',
                    confidence='low',
                    reasoning_trace=f'Error during analysis: {str(e)}',
                ),
            )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """Extract sub-metrics from the analysis result."""
        signals: FeedbackAttributionOutput = result.signals
        if not signals or not signals.has_negative_feedback:
            return []

        sub_metrics = [
            SubMetricResult(
                name='failed_step',
                score=None,
                explanation=signals.failed_step or 'unknown',
                metric_category=MetricCategory.CLASSIFICATION,
                group='feedback',
                metadata={
                    'failed_step': signals.failed_step,
                    'evidence': signals.failure_evidence,
                    'remediation_hint': signals.remediation_hint,
                },
            ),
            SubMetricResult(
                name='attribution_confidence',
                score=None,
                explanation=signals.confidence,
                metric_category=MetricCategory.CLASSIFICATION,
                group='feedback',
                metadata={
                    'confidence': signals.confidence,
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
    'SlackFeedbackAttributionAnalyzer',
    'FeedbackAttributionInput',
    'FeedbackAttributionOutput',
]
