from typing import List, Optional
from datetime import datetime, timezone
from pydantic import Field


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


from axion._core.schema import RichBaseModel, RichEnum


class Message(RichBaseModel):
    """Represents a single message in a conversation thread."""

    ts: str = Field(..., description="Timestamp of the message (Slack ts format)")
    timestamp_utc: Optional[datetime] = Field(
        None, description="Parsed UTC datetime from ts", exclude=True
    )
    sender: str = Field(..., description="Display name of sender")
    user_id: Optional[str] = Field(
        None, description="Slack user ID (U...)", exclude=True
    )
    is_bot: bool = Field(
        default=False, description="True if the message is from a system bot"
    )
    content: str = Field(..., description="The text content of the message")
    messageUrl: Optional[str] = Field(
        None, description="Direct URL to this specific message"
    )
    reply_count: int = Field(
        default=0, description="Number of replies to this message", exclude=True
    )
    is_thread_reply: bool = Field(
        default=False,
        description="True if this is a reply within a thread",
        exclude=True,
    )


class ConversationContext(RichBaseModel):
    """Represents a single underwriting thread/case to be analyzed."""

    thread_id: str = Field(
        ..., description="The unique ID of the Slack thread (from data.id)"
    )
    channel_id: Optional[str] = Field(
        None, description="Slack channel ID for utilization tracking", exclude=True
    )
    case_id: Optional[str] = Field(
        None, description="Extracted Case ID (e.g., MGT-BOP-...)"
    )
    slack_url: Optional[str] = Field(None, description="URL to the start of the thread")
    title: Optional[str] = Field(None, description="Name of the business or subject")
    messages: List[Message] = Field(
        default_factory=list, description="Chronological list of messages in the thread"
    )
    thread_created_at: Optional[datetime] = Field(
        None, description="When the thread was created", exclude=True
    )
    thread_last_activity_at: Optional[datetime] = Field(
        None, description="When the last message was posted", exclude=True
    )
    human_participants: List[str] = Field(
        default_factory=list,
        description="List of human user IDs for MAU tracking",
        exclude=True,
    )


class InterventionType(str, RichEnum):
    NONE = "no_intervention"
    CORRECTION_FACTUAL = "correction_factual"
    MISSING_CONTEXT = "missing_context"
    RISK_APPETITE = "risk_appetite"
    TECH_ISSUE = "tech_issue"
    DATA_QUALITY = "data_quality"
    CLARIFICATION = "clarification"
    SUPPORT = "support"
    APPROVAL = "approval"


class EscalationType(str, RichEnum):
    """Escalation taxonomy for dashboard categorization."""

    HARD = "hard"  # CORRECTION_FACTUAL, TECH_ISSUE, DATA_QUALITY
    SOFT = "soft"  # MISSING_CONTEXT, RISK_APPETITE, CLARIFICATION, SUPPORT
    AUTHORITY = "authority"  # APPROVAL
    NONE = "none"  # No intervention (STP)


class Sentiment(str, RichEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"


class AuditResult(RichBaseModel):
    """The structured analysis extracted from the conversation."""

    model_config = {"extra": "forbid"}

    has_human_intervention: bool = Field(
        ...,
        description="True if a human (non-bot) actually wrote a message in the thread.",
    )

    intervention_category: InterventionType

    escalation_type: Optional[EscalationType] = Field(
        None,
        description="Escalation classification (hard/soft/authority/none) computed from intervention_category.",
    )

    summary_of_human_input: Optional[str] = Field(
        None, description="A 1-sentence summary of what the human(s) said or asked."
    )

    friction_point: Optional[str] = Field(
        None,
        description="The specific concept causing friction (e.g., 'Roof Age', 'Inverted Logic', 'Default Square Footage').",
    )

    issue_details: Optional[str] = Field(
        None,
        description="Detailed technical description of the issue, including specific values, calculations, or root cause analysis (e.g., 'Premium showing $4,303 instead of $430K - appears to be divided by 100', 'Wine Bar with >75% alcohol sales has no valid industry code mapping').",
    )

    human_sentiment: Sentiment

    final_status: str = Field(
        ...,
        description="The final state of the quote (e.g., Approved, Declined, Blocked, Needs Info).",
    )


class ConversationMetrics(RichBaseModel):
    """Computed metrics for a single conversation."""

    thread_id: str = Field(..., description="Reference to the conversation")
    computed_at: datetime = Field(
        default_factory=_utcnow, description="When metrics were computed"
    )

    # Message counts
    total_messages: int = Field(0, description="Total number of messages in the thread")
    human_messages: int = Field(0, description="Number of messages from humans")
    bot_messages: int = Field(0, description="Number of messages from bots")

    # Time metrics
    duration_seconds: float = Field(
        0, description="Total duration of conversation in seconds"
    )
    first_response_time_seconds: Optional[float] = Field(
        None, description="Time to first human response"
    )
    avg_response_time_seconds: Optional[float] = Field(
        None, description="Average response time"
    )

    # Ping-pong
    exchange_count: int = Field(0, description="Number of bot<->human transitions")
    ping_pong_ratio: float = Field(
        0, description="Ratio of exchanges to total messages"
    )

    # Resolution
    is_resolved: bool = Field(
        False, description="Whether the conversation reached a resolution"
    )
    resolution_type: Optional[str] = Field(
        None, description="approved, declined, stalemate, or None"
    )
    time_to_resolution_seconds: Optional[float] = Field(
        None, description="Time from start to resolution"
    )

    # Sentiment
    has_frustrated_message: bool = Field(
        False, description="Whether any message indicated frustration"
    )

    # Stalemate detection (72 hours = 259200 seconds)
    is_stalemate: bool = Field(
        False,
        description="True if conversation inactive for >72 hours without resolution",
    )


class AggregatedMetrics(RichBaseModel):
    """Aggregated metrics for a time period (dashboard data)."""

    period_start: datetime = Field(..., description="Start of the aggregation period")
    period_end: datetime = Field(..., description="End of the aggregation period")
    period_type: str = Field(..., description="daily, weekly, or monthly")

    # Trust Metrics
    mau: int = Field(0, description="Monthly Active Users (human participants)")
    total_conversations: int = Field(0, description="Total conversations in period")
    shift_left_rate: float = Field(0, description="Rate of STP (no human intervention)")

    # Escalation Spectrum
    hard_count: int = Field(0, description="Count of hard escalations")
    soft_count: int = Field(0, description="Count of soft escalations")
    authority_count: int = Field(0, description="Count of authority escalations")
    stp_count: int = Field(
        0, description="Count of straight-through processing (no intervention)"
    )

    # Operational Efficiency
    stp_rate: float = Field(0, description="STP rate (stp_count / total)")
    avg_messages: float = Field(0, description="Average messages per conversation")
    avg_ping_pong: float = Field(0, description="Average ping-pong ratio")
    avg_turnaround_seconds: float = Field(0, description="Average time to resolution")

    # Conversation Hygiene
    clarification_rate: float = Field(
        0, description="Rate of clarification interventions"
    )
    mttr_seconds: float = Field(0, description="Mean Time to Resolution")
    stalemate_rate: float = Field(0, description="Rate of stalemate conversations")
    frustrated_rate: float = Field(
        0, description="Rate of conversations with frustrated messages"
    )

    # Business Impact
    approval_rate: float = Field(0, description="Rate of approved quotes")
    decline_rate: float = Field(0, description="Rate of declined quotes")
