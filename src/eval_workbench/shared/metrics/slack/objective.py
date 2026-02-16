import asyncio
from typing import Any, Dict, List, Literal, Optional

from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion._handlers.llm.handler import LLMHandler
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SubMetricResult
from pydantic import Field

from eval_workbench.shared.metrics.slack.config import (
    AnalyzerConfig,
    TruncationConfig,
    resolve_analyzer_config,
)
from eval_workbench.shared.metrics.slack.truncation import (
    format_truncated_transcript,
    truncate_conversation,
)
from eval_workbench.shared.metrics.slack.utils import (
    build_transcript,
    calculate_time_to_resolution,
    extract_mentions,
    extract_recommendation_type,
    find_recommendation_turn,
    get_human_messages,
    parse_slack_metadata,
)


class ObjectiveAnalysisInput(RichBaseModel):
    """Input for objective analysis (Pass 1)."""

    conversation_transcript: str = Field(description='Full conversation transcript')
    has_recommendation: bool = Field(
        description='Whether a recommendation was detected (from heuristics)'
    )
    recommendation_type: str = Field(description='Type of recommendation if found')
    recommendation_turn: Optional[int] = Field(
        description='Turn where recommendation was made'
    )
    detected_mentions: List[str] = Field(
        description='@mentions detected in conversation'
    )
    human_message_count: int = Field(description='Number of human messages')
    truncation_summary: str = Field(
        default='',
        description='Summary of any truncation applied to the conversation',
    )
    bot_name: str = Field(default='Athena', description='Name of the AI assistant')
    domain_context: str = Field(
        default='insurance underwriting',
        description='Domain context for the conversation',
    )


class ObjectiveAnalysisOutput(RichBaseModel):
    """Output for objective analysis - factual, deterministic classifications."""

    # Escalation
    is_escalated: bool = Field(
        default=False,
        description='Whether conversation was transferred to a human',
    )
    escalation_type: Literal[
        'no_escalation',
        'team_mention',
        'explicit_handoff',
        'error_escalation',
        'complexity_escalation',
    ] = Field(
        default='no_escalation',
        description='Type of escalation that occurred',
    )
    escalation_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where escalation occurred',
    )
    escalation_reason: str = Field(
        default='',
        description='Brief reason for escalation',
    )

    # Intervention
    has_intervention: bool = Field(
        default=False,
        description='Whether a human participated to help/correct the bot',
    )
    intervention_type: Literal[
        'no_intervention',
        'correction_factual',  # Fixing data errors (Year built, Sq Ft)
        'correction_classification',  # Fixing Class Codes (Church vs Retail)
        'missing_context',  # Providing info the bot didn't have
        'risk_appetite',  # Judgment call on risk
        'system_workaround',  # Bypassing a tool error (Force decline)
        'clarification',  # Explaining the "Why"
        'support',  # General help
        'approval',  # Signing off
        'unknown',
    ] = Field(
        default='no_intervention',
        description='Type of human intervention',
    )
    intervention_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where intervention occurred',
    )
    intervention_summary: str = Field(
        default='',
        description='Brief summary of the intervention',
    )
    friction_point: Optional[str] = Field(
        default=None,
        description='Specific concept causing friction (e.g., "Payroll", "Year Built")',
    )
    issue_details: Optional[str] = Field(
        default=None,
        description='Technical details of the issue',
    )
    rules_referenced: List[str] = Field(
        default_factory=list,
        description='Rule names or concepts referenced by the bot (e.g., "property_rate_low_Refer", "Payroll Cap $300k")',
    )

    # Resolution
    final_status: Literal[
        'approved',
        'declined',
        'blocked',
        'needs_info',
        'stalemate',
        'pending',
    ] = Field(
        default='pending',
        description='Final status of the conversation',
    )
    is_resolved: bool = Field(
        default=False,
        description='Whether the conversation reached a clear resolution',
    )
    resolution_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where resolution occurred',
    )

    # Chain-of-Thought
    reasoning_trace: str = Field(
        default='',
        description='Step-by-step reasoning for classifications',
    )


class EscalationSignals(RichBaseModel):
    """Escalation signals."""

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


class InterventionSignals(RichBaseModel):
    """Intervention signals."""

    has_intervention: bool = Field(default=False)
    intervention_type: Literal[
        'no_intervention',
        'correction_factual',
        'correction_classification',
        'missing_context',
        'risk_appetite',
        'system_workaround',
        'clarification',
        'support',
        'approval',
        'unknown',
    ] = Field(default='no_intervention')
    intervention_escalation: Literal['hard', 'soft', 'authority', 'none'] = Field(
        default='none'
    )
    is_stp: bool = Field(default=True, description='Straight-through processing')
    intervention_summary: str = Field(default='')
    friction_point: Optional[str] = Field(default=None)
    issue_details: Optional[str] = Field(default=None)
    rules_referenced: List[str] = Field(default_factory=list)


class ResolutionSignals(RichBaseModel):
    """Resolution signals."""

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
    time_to_resolution_seconds: Optional[int] = Field(default=None)


class ObjectiveAnalysisResult(RichBaseModel):
    """Result from objective analysis."""

    thread_id: Optional[str] = Field(default=None)
    channel_id: Optional[str] = Field(default=None)
    sender: Optional[str] = Field(default=None)
    case_id: Optional[str] = Field(default=None)

    escalation: EscalationSignals = Field(default_factory=EscalationSignals)
    intervention: InterventionSignals = Field(default_factory=InterventionSignals)
    resolution: ResolutionSignals = Field(default_factory=ResolutionSignals)

    reasoning_trace: str = Field(default='')
    truncation_applied: bool = Field(default=False)

    def to_kpi_summary(self) -> Dict[str, Any]:
        """Extract key KPI values."""
        return {
            'thread_id': self.thread_id,
            'case_id': self.case_id,
            'is_escalated': self.escalation.is_escalated,
            'escalation_type': self.escalation.escalation_type,
            'has_intervention': self.intervention.has_intervention,
            'intervention_type': self.intervention.intervention_type,
            'is_stp': self.intervention.is_stp,
            'rules_referenced': self.intervention.rules_referenced,
            'final_status': self.resolution.final_status,
            'is_resolved': self.resolution.is_resolved,
            'time_to_resolution_seconds': self.resolution.time_to_resolution_seconds,
        }


class _ObjectiveLLMAnalyzer(
    BaseMetric[ObjectiveAnalysisInput, ObjectiveAnalysisOutput]
):
    """Internal LLM analyzer for objective classification."""

    instruction = """You are an expert Underwriting Auditor for an AI Assistant ({bot_name}).

## YOUR TASK
Analyze the conversation objectively to classify **Escalation**, **Intervention**, and **Resolution**.

Focus specifically on differentiating between *helping* the bot (context) vs *correcting* the bot (data/logic errors).

{truncation_notice}

---

## INTERVENTION CATEGORIES
*Did a human participate to help/correct the bot?*

**correction_factual** - Fixing Bad Data:
- User corrects specific numbers or facts from "Magic Dust" or other sources.
- Examples: "Sq ft is 100k, not 10k", "Built in 2026, not 1978".

**correction_classification** - Fixing Class Codes:
- User changes the Industry or Class Code because the bot got it wrong.
- Examples: "It's a Church, not Retail", "Change from Office to Medical".

**system_workaround** - Tooling Failure:
- User is blocked by a software error or UI bug.
- Examples: "Failed to decline", "AAL is broken", "Force decline manually".

**risk_appetite** - Judgment Call:
- Bot followed rules, but human made a judgment call to override.
- Examples: "Approved based on inspection", "Declined due to crime score".

**missing_context** - Providing Info:
- User provides info the bot didn't have (not a correction, just an addition).
- Examples: "Agent confirmed sprinklers are present".

**clarification** - Asking "Why":
- User is asking for an explanation, not correcting anything.
- Examples: "Why did this get referred?", "What does property_rate_low mean?".

**support** - General System Help:
- User needs help navigating the platform, not about the recommendation itself.
- Examples: "How do I override this?", "Where is the AAL field?".

**approval** - Pure Sign-Off:
- User explicitly approves with zero new reasoning or information.
- Examples: "Looks good, approved", "Go ahead", "Accepted".

**unknown** - Unclassifiable Intervention:
- A human participated but the intervention doesn't fit any category above.
- Use sparingly — prefer a specific category whenever possible.

---

## RESOLUTION STATUS
- `approved`: Clear approval decision made.
- `declined`: Clear decline decision made.
- `blocked`: Waiting on external factor or system fix.
- `needs_info`: Explicitly waiting for agent/insured.
- `stalemate`: No progress being made.
- `pending`: Conversation still in progress, no clear outcome yet.

---

## ANALYSIS RULES
1. **Data vs. Opinion**: If the user says "The data is wrong", it is `correction_factual`. If they say "I don't like this risk", it is `risk_appetite`.
2. **System vs. Model**: If the user complains about "SFX", "Socotra", or "Swallow" errors, it is `system_workaround`.
3. **Escalation**: Only mark `is_escalated` if the conversation is explicitly handed off to another human/team.
4. **Approval vs. Risk Appetite**: "Approved" + own judgment/reasoning = `risk_appetite`. "Approved" with no additions = `approval`. True approval means zero new information or reasoning from the user.

---

## FRICTION CONTEXT
When an intervention is detected, also populate:
- `friction_point`: A concise noun phrase for the domain concept causing friction (e.g., "Payroll Cap", "Year Built", "Coastal Distance", "Class Code").
- `issue_details`: The specific technical details — incorrect values, rule names, system errors (e.g., "Magic Dust shows Year Built 1954, actual is 2026", "property_rate_low_Refer triggered incorrectly").

These fields are critical for downstream root-cause analysis. Populate them whenever `has_intervention` is True.

---

## RULES REFERENCED
Extract any specific underwriting rules, referral names, or business concepts that {bot_name} cited in its messages.
- Look for rule identifiers like "property_rate_low_Refer", "Payroll Cap $300k", "coastal_distance_Decline".
- Include referral reason names, hard-block rule names, and any quoted policy thresholds.
- If no rules are referenced, return an empty list.

---

## OUTPUT FORMAT
Provide your reasoning trace first, identifying the specific turn where intervention occurred."""

    input_model = ObjectiveAnalysisInput
    output_model = ObjectiveAnalysisOutput
    description = 'Objective escalation/intervention/resolution classification'

    examples = [
        (
            ObjectiveAnalysisInput(
                conversation_transcript="""
[Turn 0] Athena: Recommendation: Approved.
[Turn 1] User: Looks good, approved.
""",
                has_recommendation=True,
                recommendation_type='approve',
                recommendation_turn=0,
                detected_mentions=[],
                human_message_count=1,
                truncation_summary='',
                bot_name='Athena',
                domain_context='insurance underwriting',
            ),
            ObjectiveAnalysisOutput(
                is_escalated=False,
                escalation_type='no_escalation',
                escalation_turn_index=None,
                escalation_reason='',
                has_intervention=True,
                intervention_type='approval',
                intervention_turn_index=1,
                intervention_summary='User explicitly signed off with no additional reasoning.',
                friction_point=None,
                issue_details=None,
                rules_referenced=[],
                final_status='approved',
                is_resolved=True,
                resolution_turn_index=1,
                reasoning_trace='User gave pure approval in Turn 1 and did not add new data or judgment.',
            ),
        ),
        (
            ObjectiveAnalysisInput(
                conversation_transcript="""
[Turn 0] Athena: Referred due to year_built_pre_1980_Refer (Year Built = 1954 from Magic Dust).
[Turn 1] User: Year built is 2026, not 1954.
[Turn 2] User: Can you rerun this?
""",
                has_recommendation=True,
                recommendation_type='refer',
                recommendation_turn=0,
                detected_mentions=[],
                human_message_count=2,
                truncation_summary='',
                bot_name='Athena',
                domain_context='insurance underwriting',
            ),
            ObjectiveAnalysisOutput(
                is_escalated=False,
                escalation_type='no_escalation',
                escalation_turn_index=None,
                escalation_reason='',
                has_intervention=True,
                intervention_type='correction_factual',
                intervention_turn_index=1,
                intervention_summary='User corrected a factual data point used by the recommendation.',
                friction_point='Year Built',
                issue_details='Magic Dust reported 1954, but user states actual year built is 2026.',
                rules_referenced=['year_built_pre_1980_Refer'],
                final_status='pending',
                is_resolved=False,
                resolution_turn_index=None,
                reasoning_trace='This is a factual correction, not risk appetite. Conversation is awaiting rerun outcome.',
            ),
        ),
        (
            ObjectiveAnalysisInput(
                conversation_transcript="""
[Turn 0] Athena: Recommendation: Approved.
[Turn 1] User: This is owner occupied condo risk; I need to decline.
[Turn 2] User: SFX gives "failed to decline" when I try.
[Turn 3] User: @uw-support can someone force decline this?
""",
                has_recommendation=True,
                recommendation_type='approve',
                recommendation_turn=0,
                detected_mentions=['@uw-support'],
                human_message_count=3,
                truncation_summary='',
                bot_name='Athena',
                domain_context='insurance underwriting',
            ),
            ObjectiveAnalysisOutput(
                is_escalated=True,
                escalation_type='team_mention',
                escalation_turn_index=3,
                escalation_reason='User mentioned support team for manual intervention.',
                has_intervention=True,
                intervention_type='system_workaround',
                intervention_turn_index=2,
                intervention_summary='User cannot execute decline due to platform error.',
                friction_point='Decline Workflow',
                issue_details='SFX returns "failed to decline", requiring manual support path.',
                rules_referenced=[],
                final_status='blocked',
                is_resolved=False,
                resolution_turn_index=None,
                reasoning_trace='Intervention is driven by tooling failure and conversation escalates via team mention.',
            ),
        ),
    ]


@metric(
    name='Slack Objective Analyzer',
    key='slack_objective_analyzer',
    description='Objective factual classification of Slack conversations (escalation, intervention, resolution).',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.ANALYSIS,
    default_threshold=None,
    score_range=None,
    tags=['slack', 'multi_turn'],
)
class SlackObjectiveAnalyzer(BaseMetric):
    """
    Objective Analyzer for Slack conversations.

    Uses low temperature (0.0) for deterministic, consistent outputs.
    Focuses on objective, verifiable facts:
    - `escalation`: Was conversation transferred to a human?
    - `intervention`: Did human participate to help/correct?
    - `resolution`: Final outcome status

    This is Pass 1 of the split architecture - run before SubjectiveAnalyzer.
    """

    is_multi_metric = True
    include_parent_score = False
    sub_metric_prefix = False

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        truncation_config: Optional[TruncationConfig] = None,
        **kwargs,
    ):
        # Force low temperature for deterministic outputs
        kwargs['temperature'] = 0.0
        super().__init__(**kwargs)
        self.analyzer_config = resolve_analyzer_config(config)
        self.truncation_config = truncation_config or TruncationConfig()
        self._llm_analyzer: _ObjectiveLLMAnalyzer | None = None
        self._instruction_lock = asyncio.Lock()

    @property
    def llm_analyzer(self) -> _ObjectiveLLMAnalyzer:
        if self._llm_analyzer is None:
            self._llm_analyzer = _ObjectiveLLMAnalyzer(
                model_name=self.model_name,
                llm=self.llm,
                temperature=0.0,
            )
        return self._llm_analyzer

    @trace(name='SlackObjectiveAnalyzer', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute objective analysis."""
        if not item.conversation:
            return MetricEvaluationResult(
                score=None,
                explanation='No conversation provided.',
                signals=ObjectiveAnalysisResult(),
            )

        metadata = parse_slack_metadata(item.additional_input)
        additional = item.additional_input or {}
        all_messages = item.conversation.messages
        human_messages = get_human_messages(item.conversation)

        human_count = len(human_messages)

        # Early exit for non-interactive
        if human_count == 0:
            return MetricEvaluationResult(
                score=None,
                explanation='No human messages - skipping objective analysis.',
                signals=ObjectiveAnalysisResult(
                    thread_id=metadata.thread_ts,
                    channel_id=metadata.channel_id,
                ),
            )

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
        if has_recommendation:
            rec_text = all_messages[rec_turn].content or ''
            rec_type = extract_recommendation_type(rec_text) or 'none'

        # Truncate conversation
        truncated_messages, truncation_summary = truncate_conversation(
            all_messages,
            self.truncation_config,
            recommendation_turn=rec_turn,
        )

        if truncation_summary:
            transcript = format_truncated_transcript(
                truncated_messages, truncation_summary
            )
        else:
            transcript = build_transcript(item.conversation)

        llm_input = ObjectiveAnalysisInput(
            conversation_transcript=transcript,
            has_recommendation=has_recommendation,
            recommendation_type=rec_type,
            recommendation_turn=rec_turn,
            detected_mentions=unique_mentions,
            human_message_count=human_count,
            truncation_summary=truncation_summary,
            bot_name=self.analyzer_config.bot_name,
            domain_context=self.analyzer_config.domain_context,
        )

        try:
            truncation_notice = (
                f'IMPORTANT: Conversation was truncated. {truncation_summary}'
                if truncation_summary
                else ''
            )
            # LLMHandler does not auto-format instruction placeholders.
            # Temporarily render dynamic prompt variables for this call only.
            formatted_instruction = self.llm_analyzer.instruction.format(
                bot_name=self.analyzer_config.bot_name,
                truncation_notice=truncation_notice,
            )
            async with self._instruction_lock:
                original_instruction = self.llm_analyzer.instruction
                self.llm_analyzer.instruction = formatted_instruction
                try:
                    # Call LLMHandler.execute() directly (not BaseMetric.execute() which expects DatasetItem)
                    llm_result: ObjectiveAnalysisOutput = await LLMHandler.execute(
                        self.llm_analyzer, llm_input
                    )
                finally:
                    self.llm_analyzer.instruction = original_instruction

            # Copy cost from internal analyzer to self for sub-metric distribution
            self.cost_estimate = getattr(self.llm_analyzer, 'cost_estimate', None)

            # Map to signals
            escalation = EscalationSignals(
                is_escalated=llm_result.is_escalated,
                escalation_type=llm_result.escalation_type,
                escalation_turn_index=llm_result.escalation_turn_index,
                escalation_targets=unique_mentions if llm_result.is_escalated else [],
                escalation_reason=llm_result.escalation_reason,
            )

            int_type = llm_result.intervention_type
            esc_type, is_stp = self._classify_intervention_escalation(int_type)
            intervention = InterventionSignals(
                has_intervention=llm_result.has_intervention,
                intervention_type=int_type,
                intervention_escalation=esc_type,
                is_stp=is_stp,
                intervention_summary=llm_result.intervention_summary,
                friction_point=llm_result.friction_point,
                issue_details=llm_result.issue_details,
                rules_referenced=llm_result.rules_referenced,
            )

            status = llm_result.final_status
            is_resolved = status in ['approved', 'declined', 'stalemate']
            time_to_res = calculate_time_to_resolution(all_messages)
            resolution = ResolutionSignals(
                final_status=status,
                is_resolved=is_resolved,
                resolution_type=status if is_resolved else None,
                is_stalemate=status == 'stalemate',
                time_to_resolution_seconds=time_to_res,
            )

            result = ObjectiveAnalysisResult(
                thread_id=metadata.thread_ts,
                channel_id=metadata.channel_id,
                sender=metadata.sender,
                case_id=additional.get('case_id'),
                escalation=escalation,
                intervention=intervention,
                resolution=resolution,
                reasoning_trace=llm_result.reasoning_trace,
                truncation_applied=bool(truncation_summary),
            )

            parts = []
            if escalation.is_escalated:
                parts.append(f'Escalated: {escalation.escalation_type}')
            if intervention.has_intervention:
                parts.append(f'Intervention: {intervention.intervention_type}')
            parts.append(f'Status: {resolution.final_status}')

            return MetricEvaluationResult(
                score=None,
                explanation='. '.join(parts),
                signals=result,
                metadata={'objective_result': result.model_dump()},
            )

        except Exception as e:
            # Fallback to heuristic
            escalation = self._heuristic_escalation(all_messages, unique_mentions)
            return MetricEvaluationResult(
                score=None,
                explanation=f'LLM failed, using heuristic: {e}',
                signals=ObjectiveAnalysisResult(
                    thread_id=metadata.thread_ts,
                    escalation=escalation,
                    reasoning_trace=f'Heuristic fallback: {e}',
                ),
            )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """Extract sub-metrics."""
        signals: ObjectiveAnalysisResult = result.signals
        if not signals:
            return []

        sub_metrics = [
            SubMetricResult(
                name='escalation_type',
                score=None,
                explanation=signals.escalation.escalation_type,
                metric_category=MetricCategory.CLASSIFICATION,
                group='objective',
                metadata=signals.escalation.model_dump(),
            ),
            SubMetricResult(
                name='intervention_type',
                score=None,
                explanation=signals.intervention.intervention_type,
                metric_category=MetricCategory.CLASSIFICATION,
                group='objective',
                metadata=signals.intervention.model_dump(),
            ),
            SubMetricResult(
                name='resolution_status',
                score=None,
                explanation=signals.resolution.final_status,
                metric_category=MetricCategory.CLASSIFICATION,
                group='objective',
                metadata=signals.resolution.model_dump(),
            ),
        ]

        # Distribute cost across sub-metrics (normalized per metric)
        total_cost = getattr(self, 'cost_estimate', None)
        if total_cost and len(sub_metrics) > 0:
            cost_per_metric = total_cost / len(sub_metrics)
            for sub in sub_metrics:
                sub.metadata['cost_estimate'] = cost_per_metric

        return sub_metrics

    def _classify_intervention_escalation(self, intervention_type: str) -> tuple:
        HARD = {
            'correction_factual',
            'correction_classification',
            'system_workaround',
        }
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

    def _heuristic_escalation(
        self, messages: list, mentions: List[str]
    ) -> EscalationSignals:
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


__all__ = [
    'SlackObjectiveAnalyzer',
    'ObjectiveAnalysisResult',
    'ObjectiveAnalysisInput',
    'ObjectiveAnalysisOutput',
    'EscalationSignals',
    'InterventionSignals',
    'ResolutionSignals',
]
