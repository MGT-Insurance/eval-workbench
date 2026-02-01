from typing import List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import build_transcript, get_human_messages

InterventionCategory = Literal[
    'no_intervention',
    'correction_factual',
    'missing_context',
    'risk_appetite',
    'tech_issue',
    'data_quality',
    'clarification',
    'support',
    'approval',
]

EscalationCategory = Literal['hard', 'soft', 'authority', 'none']


class InterventionInput(RichBaseModel):
    """Input model for intervention detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    human_message_count: int = Field(
        description='Number of human messages in the conversation'
    )


class InterventionOutput(RichBaseModel):
    """Output model for intervention detection."""

    has_human_intervention: bool = Field(
        default=False, description='Whether a human intervened'
    )
    intervention_category: InterventionCategory = Field(
        default='no_intervention', description='Type of intervention'
    )
    summary_of_human_input: str = Field(
        default='', description='Summary of human input'
    )
    friction_point: Optional[str] = Field(
        default=None, description='Specific concept causing friction'
    )
    issue_details: Optional[str] = Field(
        default=None, description='Technical details of the issue'
    )
    reasoning: str = Field(default='', description='Brief analysis explanation')


class InterventionResult(RichBaseModel):
    """Result model for intervention analysis."""

    has_intervention: bool = Field(
        default=False, description='Whether a human intervened'
    )
    intervention_type: InterventionCategory = Field(
        default='no_intervention', description='Type of intervention'
    )
    escalation_type: EscalationCategory = Field(
        default='none', description='Escalation classification'
    )
    is_hard_escalation: bool = Field(default=False)
    is_soft_escalation: bool = Field(default=False)
    is_authority_escalation: bool = Field(default=False)
    is_stp: bool = Field(default=True, description='Straight-through processing')
    intervention_summary: str = Field(default='', description='Summary of human input')
    friction_point: Optional[str] = Field(default=None)
    issue_details: Optional[str] = Field(default=None)
    reasoning: str = Field(default='')
    human_message_count: int = Field(default=0)


def classify_escalation(
    intervention_type: InterventionCategory,
) -> tuple[EscalationCategory, bool, bool, bool, bool]:
    """
    Map intervention type to escalation classification.

    Returns: (escalation_type, is_hard, is_soft, is_authority, is_stp)
    """
    HARD = {'correction_factual', 'tech_issue', 'data_quality'}
    SOFT = {'missing_context', 'risk_appetite', 'clarification', 'support'}
    AUTHORITY = {'approval'}

    if intervention_type in HARD:
        return 'hard', True, False, False, False
    elif intervention_type in SOFT:
        return 'soft', False, True, False, False
    elif intervention_type in AUTHORITY:
        return 'authority', False, False, True, False
    else:
        return 'none', False, False, False, True


class InterventionAnalyzer(BaseMetric[InterventionInput, InterventionOutput]):
    """Internal LLM-based analyzer for intervention detection."""

    instruction = """You are a Senior Underwriting Auditor analyzing AI-Human conversations.

**CONTEXT**:
- The AI assistant is named 'Athena' (appears as AI/bot messages)
- Human messages are from Underwriters or Agents

**TASK**: Determine if a human intervened and why.

**INTERVENTION CATEGORIES**:
- `no_intervention`: No human involvement (STP - Straight Through Processing)
- `correction_factual`: Human correcting AI's factual errors (wrong data, calculations)
- `missing_context`: Human providing context AI didn't have
- `risk_appetite`: Human applying risk judgment AI couldn't make
- `tech_issue`: Human reporting a bug or technical problem
- `data_quality`: Human fixing bad/missing source data
- `clarification`: Human asking for clarification from AI
- `support`: Human requesting help or assistance
- `approval`: Human providing required approval/authorization

**EXTRACT**:
- `friction_point`: The specific rule, data point, or concept causing discussion (e.g., "Roof Age", "Premium Calculation", "Industry Code")
- `issue_details`: Technical details including exact values, discrepancies, root causes (e.g., "Premium showing $4,303 instead of $430K - divided by 100")

**OUTPUT**: Provide all fields as specified. Be precise with friction_point and issue_details."""

    input_model = InterventionInput
    output_model = InterventionOutput


@metric(
    name='Intervention Detector',
    key='intervention_detector',
    description='Detects human intervention in AI conversations and classifies the type.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.CLASSIFICATION,
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['slack', 'multi_turn', 'classification', 'feedback'],
)
class InterventionDetector(BaseMetric):
    """
    LLM-based metric that detects human intervention in conversations.

    Used for computing:
    - intervention_rate: Conversations with intervention / Total conversations
    - stp_rate: Straight-through processing rate (no intervention)
    - escalation breakdown: hard / soft / authority

    Intervention taxonomy:
    - HARD: correction_factual, tech_issue, data_quality
    - SOFT: missing_context, risk_appetite, clarification, support
    - AUTHORITY: approval
    - NONE: no_intervention (STP)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intervention_analyzer = InterventionAnalyzer(**kwargs)

    @trace(name='InterventionDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect intervention in conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=InterventionResult(),
            )

        human_messages = get_human_messages(item.conversation)
        transcript = build_transcript(item.conversation)

        # If no human messages, it's STP
        if not human_messages:
            result = InterventionResult(
                has_intervention=False,
                intervention_type='no_intervention',
                escalation_type='none',
                is_stp=True,
                human_message_count=0,
            )
            return MetricEvaluationResult(
                score=0.0,
                explanation='No human messages - STP (straight-through processing).',
                signals=result,
            )

        # Run LLM analysis
        try:
            analysis_input = InterventionInput(
                conversation_transcript=transcript,
                human_message_count=len(human_messages),
            )

            llm_result = await self.intervention_analyzer.execute(analysis_input)

            # Classify escalation
            esc_type, is_hard, is_soft, is_auth, is_stp = classify_escalation(
                llm_result.intervention_category
            )

            result = InterventionResult(
                has_intervention=llm_result.has_human_intervention,
                intervention_type=llm_result.intervention_category,
                escalation_type=esc_type,
                is_hard_escalation=is_hard,
                is_soft_escalation=is_soft,
                is_authority_escalation=is_auth,
                is_stp=is_stp,
                intervention_summary=llm_result.summary_of_human_input,
                friction_point=llm_result.friction_point,
                issue_details=llm_result.issue_details,
                reasoning=llm_result.reasoning,
                human_message_count=len(human_messages),
            )

            score = 1.0 if result.has_intervention else 0.0
            explanation = (
                f'Intervention: {result.intervention_type}, '
                f'Escalation: {result.escalation_type}. '
                f'Friction: {result.friction_point or "N/A"}'
            )

        except Exception as e:
            result = self._heuristic_fallback(human_messages)
            score = 1.0 if result.has_intervention else 0.0
            explanation = f'Heuristic analysis (LLM failed: {e})'

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=result,
        )

    def _heuristic_fallback(self, human_messages: list) -> InterventionResult:
        """Fallback heuristic when LLM fails."""
        # Simple heuristic: if there are human messages, there was intervention
        has_intervention = len(human_messages) > 0

        return InterventionResult(
            has_intervention=has_intervention,
            intervention_type='clarification'
            if has_intervention
            else 'no_intervention',
            escalation_type='soft' if has_intervention else 'none',
            is_soft_escalation=has_intervention,
            is_stp=not has_intervention,
            reasoning='Heuristic: human messages detected',
            human_message_count=len(human_messages),
        )

    def get_signals(
        self, result: InterventionResult
    ) -> List[SignalDescriptor[InterventionResult]]:
        """Generate signal descriptors."""

        intervention_scores = {
            'no_intervention': 0.0,
            'correction_factual': 1.0,
            'missing_context': 1.0,
            'risk_appetite': 1.0,
            'tech_issue': 1.0,
            'data_quality': 1.0,
            'clarification': 1.0,
            'support': 1.0,
            'approval': 1.0,
        }

        return [
            SignalDescriptor(
                name='has_intervention',
                extractor=lambda r: r.has_intervention,
                headline_display=True,
                description='Human intervened in conversation',
            ),
            SignalDescriptor(
                name='intervention_type',
                extractor=lambda r: r.intervention_type,
                headline_display=True,
                score_mapping=intervention_scores,
                description='Type of intervention',
            ),
            SignalDescriptor(
                name='escalation_type',
                extractor=lambda r: r.escalation_type,
                headline_display=True,
                description='Escalation classification (hard/soft/authority/none)',
            ),
            SignalDescriptor(
                name='is_stp',
                extractor=lambda r: r.is_stp,
                headline_display=True,
                description='Straight-through processing (no escalation)',
            ),
            SignalDescriptor(
                name='friction_point',
                extractor=lambda r: r.friction_point,
                description='Concept causing friction',
            ),
            SignalDescriptor(
                name='issue_details',
                extractor=lambda r: r.issue_details[:100] + '...'
                if r.issue_details and len(r.issue_details) > 100
                else r.issue_details,
                description='Technical details of the issue',
            ),
            SignalDescriptor(
                name='intervention_summary',
                extractor=lambda r: r.intervention_summary,
                description='Summary of human input',
            ),
        ]
