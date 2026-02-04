from typing import List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from eval_workbench.shared.metrics.slack.utils import build_transcript, extract_mentions

EscalationType = Literal[
    'no_escalation',
    'team_mention',
    'explicit_handoff',
    'error_escalation',
    'complexity_escalation',
]


class EscalationInput(RichBaseModel):
    """Input model for escalation detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    detected_mentions: List[str] = Field(
        default_factory=list,
        description='@mentions detected in the conversation',
    )


class EscalationOutput(RichBaseModel):
    """Output model for escalation detection."""

    is_escalated: bool = Field(
        description='Whether the conversation was escalated to a human'
    )
    escalation_type: EscalationType = Field(description='Type of escalation detected')
    escalation_turn_index: Optional[int] = Field(
        default=None,
        description='Turn index where escalation occurred (0-based)',
    )
    escalation_targets: List[str] = Field(
        default_factory=list,
        description='List of @mentioned users during escalation',
    )
    escalation_reason: str = Field(
        default='',
        description='Brief explanation of why escalation occurred',
    )


class EscalationResult(RichBaseModel):
    """Result model combining input analysis and LLM output."""

    is_escalated: bool = Field(
        default=False,
        description='Whether conversation was escalated',
    )
    escalation_type: EscalationType = Field(
        default='no_escalation',
        description='Type of escalation',
    )
    escalation_turn_index: Optional[int] = Field(
        default=None,
        description='Turn where escalation occurred',
    )
    escalation_targets: List[str] = Field(
        default_factory=list,
        description='@mentioned users during escalation',
    )
    escalation_reason: str = Field(
        default='',
        description='Reason for escalation',
    )
    total_mentions_in_thread: int = Field(
        default=0,
        description='Total @mentions detected in thread',
    )


class EscalationAnalyzer(BaseMetric[EscalationInput, EscalationOutput]):
    """Internal LLM-based analyzer for escalation detection."""

    instruction = """You are an expert analyzer detecting escalation patterns in Slack conversations between users and an AI assistant (Athena).

**TASK**: Analyze the conversation to determine if it was escalated to human team members.

**ESCALATION TYPES**:
1. **no_escalation**: Normal AI-handled conversation, no human intervention needed
2. **team_mention**: User @mentioned team members for help or review
3. **explicit_handoff**: AI explicitly handed off the conversation to a human
4. **error_escalation**: Escalation occurred due to an AI error or inability to help
5. **complexity_escalation**: Escalation due to case complexity requiring human judgment

**ESCALATION INDICATORS**:
- @mentions to non-AI users after AI interaction began
- Phrases like "I apologize, but I encountered an error" from AI
- AI saying it needs to "escalate", "transfer", or "hand off"
- Users asking for human review or assistance
- Discussion of issues AI cannot resolve
- Team members joining to provide guidance

**OUTPUT**:
- is_escalated: True if any escalation occurred
- escalation_type: The primary type of escalation (use the most specific type)
- escalation_turn_index: The turn number (0-indexed) where escalation first occurred
- escalation_targets: List of @mentioned users (from the detected_mentions provided)
- escalation_reason: A brief (1-2 sentence) explanation of why escalation occurred

If no escalation occurred, set is_escalated=False, escalation_type="no_escalation", and leave other fields empty/null."""

    input_model = EscalationInput
    output_model = EscalationOutput


@metric(
    name='Escalation Detector',
    key='escalation_detector',
    description='Detects when Slack conversations escalate to human team members.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.CLASSIFICATION,
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['slack', 'multi_turn', 'classification'],
)
class EscalationDetector(BaseMetric):
    """
    LLM-based metric that detects escalation in Slack conversations.

    Used for computing:
    - escalation_rate: Cases escalated / Total AI cases

    Classification categories:
    - no_escalation: Normal AI-handled conversation
    - team_mention: User @mentioned team members
    - explicit_handoff: AI explicitly handed off to human
    - error_escalation: Escalation due to AI error
    - complexity_escalation: Escalation due to case complexity
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.escalation_analyzer = EscalationAnalyzer(**kwargs)

    @trace(name='EscalationDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect escalation in Slack conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=EscalationResult(),
            )

        # Build transcript and extract mentions
        transcript = build_transcript(item.conversation)
        all_mentions = []
        for msg in item.conversation.messages:
            if msg.content:
                all_mentions.extend(extract_mentions(msg.content))
        unique_mentions = list(set(all_mentions))

        # Prepare input for LLM
        analysis_input = EscalationInput(
            conversation_transcript=transcript,
            detected_mentions=unique_mentions,
        )

        # Run LLM analysis
        try:
            llm_result = await self.escalation_analyzer.execute(analysis_input)

            result = EscalationResult(
                is_escalated=llm_result.is_escalated,
                escalation_type=llm_result.escalation_type,
                escalation_turn_index=llm_result.escalation_turn_index,
                escalation_targets=llm_result.escalation_targets,
                escalation_reason=llm_result.escalation_reason,
                total_mentions_in_thread=len(unique_mentions),
            )

            # Score: 1.0 if escalated, 0.0 if not
            score = 1.0 if result.is_escalated else 0.0

            explanation = (
                f'Escalation: {result.escalation_type}. {result.escalation_reason}'
                if result.is_escalated
                else 'No escalation detected.'
            )

        except Exception as e:
            # Fallback to heuristic if LLM fails
            result = self._heuristic_fallback(item, unique_mentions)
            score = 1.0 if result.is_escalated else 0.0
            explanation = (
                f'Heuristic analysis (LLM failed: {e}): {result.escalation_type}'
            )

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=result,
        )

    def _heuristic_fallback(
        self, item: DatasetItem, mentions: List[str]
    ) -> EscalationResult:
        """
        Fallback heuristic when LLM analysis fails.

        Uses simple pattern matching for escalation detection.
        """
        import re

        is_escalated = False
        escalation_type: EscalationType = 'no_escalation'
        escalation_turn = None
        escalation_reason = ''

        # Check for mentions after AI messages (simple heuristic)
        ai_seen = False
        for idx, msg in enumerate(item.conversation.messages):
            from axion._core.schema import AIMessage

            if isinstance(msg, AIMessage):
                ai_seen = True
            elif ai_seen and msg.content:
                # Check for mentions in human messages after AI
                msg_mentions = extract_mentions(msg.content)
                if msg_mentions:
                    is_escalated = True
                    escalation_type = 'team_mention'
                    escalation_turn = idx
                    escalation_reason = (
                        'User @mentioned team members after AI interaction'
                    )
                    break

        # Check for error patterns in AI messages
        if not is_escalated:
            for idx, msg in enumerate(item.conversation.messages):
                from axion._core.schema import AIMessage

                if isinstance(msg, AIMessage) and msg.content:
                    error_patterns = [
                        r'apologize.*error',
                        r'encountered.*(?:error|issue|problem)',
                        r'unable\s+to\s+(?:process|complete|help)',
                    ]
                    for pattern in error_patterns:
                        if re.search(pattern, msg.content, re.IGNORECASE):
                            is_escalated = True
                            escalation_type = 'error_escalation'
                            escalation_turn = idx
                            escalation_reason = 'AI encountered an error'
                            break
                    if is_escalated:
                        break

        return EscalationResult(
            is_escalated=is_escalated,
            escalation_type=escalation_type,
            escalation_turn_index=escalation_turn,
            escalation_targets=mentions if is_escalated else [],
            escalation_reason=escalation_reason,
            total_mentions_in_thread=len(mentions),
        )

    def get_signals(
        self, result: EscalationResult
    ) -> List[SignalDescriptor[EscalationResult]]:
        """Generate signal descriptors for escalation detection."""

        escalation_scores = {
            'no_escalation': 0.0,
            'team_mention': 1.0,
            'explicit_handoff': 1.0,
            'error_escalation': 1.0,
            'complexity_escalation': 1.0,
        }

        return [
            # Headline signals
            SignalDescriptor(
                name='is_escalated',
                extractor=lambda r: r.is_escalated,
                headline_display=True,
                description='Whether conversation was escalated',
            ),
            SignalDescriptor(
                name='escalation_type',
                extractor=lambda r: r.escalation_type,
                headline_display=True,
                score_mapping=escalation_scores,
                description='Type of escalation',
            ),
            # Detail signals
            SignalDescriptor(
                name='escalation_turn_index',
                extractor=lambda r: r.escalation_turn_index,
                description='Turn where escalation occurred',
            ),
            SignalDescriptor(
                name='escalation_targets',
                extractor=lambda r: ', '.join(r.escalation_targets)
                if r.escalation_targets
                else None,
                description='@mentioned users during escalation',
            ),
            SignalDescriptor(
                name='escalation_reason',
                extractor=lambda r: r.escalation_reason,
                description='Reason for escalation',
            ),
            SignalDescriptor(
                name='total_mentions_in_thread',
                extractor=lambda r: r.total_mentions_in_thread,
                description='Total @mentions in thread',
            ),
        ]
