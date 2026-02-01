from datetime import datetime, timezone
from typing import List, Literal, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import build_transcript

ResolutionStatus = Literal[
    'approved',
    'declined',
    'blocked',
    'needs_info',
    'stalemate',
    'pending',
]

# Stalemate threshold: 72 hours of inactivity
STALEMATE_THRESHOLD_SECONDS = 72 * 60 * 60  # 259200 seconds


class ResolutionInput(RichBaseModel):
    """Input model for resolution detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    message_count: int = Field(description='Total number of messages')


class ResolutionOutput(RichBaseModel):
    """Output model for resolution detection."""

    final_status: ResolutionStatus = Field(
        default='pending', description='Final status of the conversation'
    )
    is_resolved: bool = Field(default=False, description='Whether resolved')
    resolution_type: Optional[str] = Field(
        default=None, description='Type: approved/declined/stalemate'
    )
    reasoning: str = Field(default='', description='Brief analysis explanation')


class ResolutionResult(RichBaseModel):
    """Result model for resolution analysis."""

    final_status: ResolutionStatus = Field(default='pending')
    is_resolved: bool = Field(default=False, description='Conversation is resolved')
    resolution_type: Optional[str] = Field(
        default=None, description='approved/declined/stalemate'
    )
    is_stalemate: bool = Field(
        default=False, description='Inactive >72 hours without resolution'
    )
    time_to_resolution_seconds: Optional[float] = Field(default=None)
    reasoning: str = Field(default='')
    message_count: int = Field(default=0)


class ResolutionAnalyzer(BaseMetric[ResolutionInput, ResolutionOutput]):
    """Internal LLM-based analyzer for resolution detection."""

    instruction = """You are an expert at determining conversation outcomes in insurance underwriting workflows.

**TASK**: Analyze the conversation to determine its final resolution status.

**RESOLUTION STATUSES**:
- `approved`: Quote/case was approved, bound, or accepted
- `declined`: Quote/case was declined, rejected, or turned down
- `blocked`: Quote is blocked pending additional information or review
- `needs_info`: More information was requested but not yet provided
- `stalemate`: Conversation ended without clear resolution
- `pending`: No clear resolution yet, conversation may be ongoing

**INDICATORS**:

*Approved*:
- "Approved", "Bound", "Accepted"
- "Looks good, proceed"
- "I'm approving this"
- Confirmation of binding

*Declined*:
- "Declined", "Rejected", "Not approved"
- "Outside risk appetite"
- "Cannot bind this"

*Blocked*:
- "Blocked pending..."
- "Cannot proceed until..."
- Waiting for specific documentation

*Needs Info*:
- "Need more information"
- Questions asked but not answered
- Missing documentation mentioned

*Pending*:
- Conversation seems incomplete
- No clear outcome mentioned

**OUTPUT**:
- final_status: The resolution status
- is_resolved: True only for approved, declined, or stalemate
- resolution_type: "approved", "declined", or "stalemate" if resolved, else null
- reasoning: Brief explanation (1-2 sentences)"""

    input_model = ResolutionInput
    output_model = ResolutionOutput


@metric(
    name='Resolution Detector',
    key='resolution_detector',
    description='Detects how conversations are resolved (approved/declined/blocked/stalemate).',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.CLASSIFICATION,
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['slack', 'multi_turn', 'classification', 'feedback'],
)
class ResolutionDetector(BaseMetric):
    """
    LLM-based metric that detects conversation resolution.

    Used for computing:
    - resolution_rate: Resolved / Total
    - approval_rate: Approved / Resolved
    - decline_rate: Declined / Resolved
    - stalemate_rate: Stalemate / Total

    Resolution types:
    - approved: Case was approved/bound
    - declined: Case was declined/rejected
    - blocked: Pending additional info/review
    - needs_info: Waiting for information
    - stalemate: Inactive >72h without resolution
    - pending: No clear outcome yet
    """

    def __init__(self, stalemate_hours: float = 72.0, **kwargs):
        """
        Initialize the resolution detector.

        Args:
            stalemate_hours: Hours of inactivity to consider stalemate (default: 72)
        """
        super().__init__(**kwargs)
        self.stalemate_seconds = stalemate_hours * 3600
        self.resolution_analyzer = ResolutionAnalyzer(**kwargs)

    @trace(name='ResolutionDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect resolution in conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=ResolutionResult(),
            )

        transcript = build_transcript(item.conversation)
        message_count = len(item.conversation.messages)

        try:
            analysis_input = ResolutionInput(
                conversation_transcript=transcript,
                message_count=message_count,
            )

            llm_result = await self.resolution_analyzer.execute(analysis_input)

            # Check for stalemate based on last activity
            is_stalemate = self._check_stalemate(item, llm_result.is_resolved)

            final_status = llm_result.final_status
            is_resolved = llm_result.is_resolved
            resolution_type = llm_result.resolution_type

            # Override if stalemate detected
            if is_stalemate and not is_resolved:
                final_status = 'stalemate'
                is_resolved = True
                resolution_type = 'stalemate'

            # Calculate time to resolution
            ttr = self._calculate_ttr(item) if is_resolved else None

            result = ResolutionResult(
                final_status=final_status,
                is_resolved=is_resolved,
                resolution_type=resolution_type,
                is_stalemate=is_stalemate,
                time_to_resolution_seconds=ttr,
                reasoning=llm_result.reasoning,
                message_count=message_count,
            )

            score = 1.0 if is_resolved else 0.0
            explanation = (
                f'Status: {result.final_status}, '
                f'Resolved: {result.is_resolved}. '
                f'{result.reasoning}'
            )

        except Exception as e:
            result = self._heuristic_fallback(item)
            score = 1.0 if result.is_resolved else 0.0
            explanation = f'Heuristic analysis (LLM failed: {e}): {result.final_status}'

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=result,
        )

    def _check_stalemate(self, item: DatasetItem, is_resolved: bool) -> bool:
        """Check if conversation is stalemate based on last activity."""
        if is_resolved:
            return False

        additional = item.additional_input or {}
        last_activity = additional.get('last_activity') or additional.get(
            'thread_last_activity_at'
        )

        if not last_activity:
            return False

        # Parse last activity timestamp
        if isinstance(last_activity, datetime):
            last_dt = last_activity
        elif isinstance(last_activity, str):
            try:
                ts_float = float(last_activity.split('.')[0])
                last_dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)
            except (ValueError, TypeError):
                return False
        elif isinstance(last_activity, (int, float)):
            last_dt = datetime.fromtimestamp(last_activity, tz=timezone.utc)
        else:
            return False

        # Ensure timezone aware
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        current_time = datetime.now(timezone.utc)
        inactive_seconds = (current_time - last_dt).total_seconds()

        return inactive_seconds > self.stalemate_seconds

    def _calculate_ttr(self, item: DatasetItem) -> Optional[float]:
        """Calculate time to resolution from timestamps."""
        additional = item.additional_input or {}
        timestamps = additional.get('timestamps') or additional.get(
            'message_timestamps'
        )

        if not timestamps or len(timestamps) < 2:
            return None

        try:
            # Convert to datetimes
            dts = []
            for ts in timestamps:
                if isinstance(ts, datetime):
                    dts.append(ts)
                elif isinstance(ts, (int, float)):
                    dts.append(datetime.fromtimestamp(ts, tz=timezone.utc))
                elif isinstance(ts, str):
                    ts_float = float(ts.split('.')[0])
                    dts.append(datetime.fromtimestamp(ts_float, tz=timezone.utc))

            if len(dts) >= 2:
                return (max(dts) - min(dts)).total_seconds()
        except (ValueError, TypeError):
            pass

        return None

    def _heuristic_fallback(self, item: DatasetItem) -> ResolutionResult:
        """Fallback heuristic when LLM fails."""
        import re

        transcript = build_transcript(item.conversation) if item.conversation else ''
        transcript_lower = transcript.lower()

        final_status: ResolutionStatus = 'pending'
        is_resolved = False
        resolution_type = None

        # Check for resolution patterns
        if re.search(r'\b(approved|bound|accepted)\b', transcript_lower):
            final_status = 'approved'
            is_resolved = True
            resolution_type = 'approved'
        elif re.search(r'\b(declined|rejected|denied)\b', transcript_lower):
            final_status = 'declined'
            is_resolved = True
            resolution_type = 'declined'
        elif re.search(r'\b(blocked|pending)\b', transcript_lower):
            final_status = 'blocked'
        elif re.search(r'\b(need|require).*info', transcript_lower):
            final_status = 'needs_info'

        # Check stalemate
        is_stalemate = self._check_stalemate(item, is_resolved)
        if is_stalemate and not is_resolved:
            final_status = 'stalemate'
            is_resolved = True
            resolution_type = 'stalemate'

        return ResolutionResult(
            final_status=final_status,
            is_resolved=is_resolved,
            resolution_type=resolution_type,
            is_stalemate=is_stalemate,
            reasoning='Heuristic analysis based on keyword patterns',
            message_count=len(item.conversation.messages) if item.conversation else 0,
        )

    def get_signals(
        self, result: ResolutionResult
    ) -> List[SignalDescriptor[ResolutionResult]]:
        """Generate signal descriptors."""

        status_scores = {
            'approved': 1.0,
            'declined': 1.0,
            'stalemate': 0.5,
            'blocked': 0.3,
            'needs_info': 0.2,
            'pending': 0.0,
        }

        return [
            SignalDescriptor(
                name='final_status',
                extractor=lambda r: r.final_status,
                headline_display=True,
                score_mapping=status_scores,
                description='Final conversation status',
            ),
            SignalDescriptor(
                name='is_resolved',
                extractor=lambda r: r.is_resolved,
                headline_display=True,
                description='Conversation reached resolution',
            ),
            SignalDescriptor(
                name='resolution_type',
                extractor=lambda r: r.resolution_type,
                headline_display=True,
                description='approved/declined/stalemate',
            ),
            SignalDescriptor(
                name='is_stalemate',
                extractor=lambda r: r.is_stalemate,
                headline_display=True,
                description=f'Inactive >{self.stalemate_seconds / 3600:.0f}h',
            ),
            SignalDescriptor(
                name='time_to_resolution_seconds',
                extractor=lambda r: r.time_to_resolution_seconds,
                description='Time from first to last message',
            ),
            SignalDescriptor(
                name='message_count',
                extractor=lambda r: r.message_count,
                description='Total messages in conversation',
            ),
        ]
