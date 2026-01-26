"""
Per-conversation metric computation for feedback analytics.

This module provides utilities for computing quantitative metrics from
conversation contexts and audit results at ingestion time.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from implementations.athena.models.feedback_analysis.schema import (
    ConversationContext,
    ConversationMetrics,
    AuditResult,
    Message,
    InterventionType,
    EscalationType,
    Sentiment,
)


# Stalemate threshold: 72 hours of inactivity
STALEMATE_THRESHOLD_SECONDS = 72 * 60 * 60  # 259200 seconds


def classify_escalation(intervention: InterventionType) -> EscalationType:
    """
    Map InterventionType to EscalationType for dashboard categorization.

    Escalation taxonomy:
    - HARD: Factual corrections, technical issues, data quality problems
    - SOFT: Missing context, risk appetite, clarification, support requests
    - AUTHORITY: Approval requests (requires human authorization)
    - NONE: No intervention (straight-through processing)
    """
    HARD_INTERVENTIONS = {
        InterventionType.CORRECTION_FACTUAL,
        InterventionType.TECH_ISSUE,
        InterventionType.DATA_QUALITY,
    }
    SOFT_INTERVENTIONS = {
        InterventionType.MISSING_CONTEXT,
        InterventionType.RISK_APPETITE,
        InterventionType.CLARIFICATION,
        InterventionType.SUPPORT,
    }
    AUTHORITY_INTERVENTIONS = {
        InterventionType.APPROVAL,
    }

    if intervention in HARD_INTERVENTIONS:
        return EscalationType.HARD
    if intervention in SOFT_INTERVENTIONS:
        return EscalationType.SOFT
    if intervention in AUTHORITY_INTERVENTIONS:
        return EscalationType.AUTHORITY
    return EscalationType.NONE


class ConversationMetricCalculator:
    """
    Computes per-conversation metrics at ingestion time.

    Metrics computed:
    - Message counts (total, human, bot)
    - Time metrics (duration, response times)
    - Ping-pong analysis (exchange count, ratio)
    - Resolution detection (resolved, type, time to resolution)
    - Sentiment analysis (frustrated message detection)
    - Stalemate detection (>72 hours inactive)
    """

    def compute(
        self,
        context: ConversationContext,
        audit_result: AuditResult,
        current_time: Optional[datetime] = None,
    ) -> ConversationMetrics:
        """
        Compute all metrics for a single conversation.

        Args:
            context: The conversation context with messages
            audit_result: The LLM analysis result
            current_time: Optional current time for stalemate detection (defaults to utcnow)

        Returns:
            ConversationMetrics with all computed values
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        messages = context.messages

        if not messages:
            return ConversationMetrics(
                thread_id=context.thread_id,
                computed_at=current_time,
            )

        # Message counts
        human_msgs = [m for m in messages if not m.is_bot]
        bot_msgs = [m for m in messages if m.is_bot]

        total_messages = len(messages)
        human_messages = len(human_msgs)
        bot_messages = len(bot_msgs)

        # Time calculations
        timestamps = self._get_timestamps(messages)
        duration_seconds = self._compute_duration(timestamps)
        first_response_time = self._compute_first_response_time(messages)
        avg_response_time = self._compute_avg_response_time(messages)

        # Ping-pong analysis
        exchange_count = self._count_speaker_switches(messages)
        ping_pong_ratio = exchange_count / total_messages if total_messages > 0 else 0.0

        # Resolution detection
        is_resolved, resolution_type, ttr = self._detect_resolution(
            audit_result, timestamps, context.thread_last_activity_at, current_time
        )

        # Sentiment analysis
        has_frustrated = audit_result.human_sentiment == Sentiment.FRUSTRATED

        # Stalemate detection
        is_stalemate = self._detect_stalemate(
            is_resolved, context.thread_last_activity_at, current_time
        )

        # Update resolution_type if stalemate
        if is_stalemate and not is_resolved:
            resolution_type = "stalemate"
            is_resolved = True

        return ConversationMetrics(
            thread_id=context.thread_id,
            computed_at=current_time,
            total_messages=total_messages,
            human_messages=human_messages,
            bot_messages=bot_messages,
            duration_seconds=duration_seconds,
            first_response_time_seconds=first_response_time,
            avg_response_time_seconds=avg_response_time,
            exchange_count=exchange_count,
            ping_pong_ratio=ping_pong_ratio,
            is_resolved=is_resolved,
            resolution_type=resolution_type,
            time_to_resolution_seconds=ttr,
            has_frustrated_message=has_frustrated,
            is_stalemate=is_stalemate,
        )

    def _get_timestamps(self, messages: List[Message]) -> List[datetime]:
        """Extract parsed timestamps from messages."""
        timestamps = []
        for m in messages:
            if m.timestamp_utc:
                timestamps.append(m.timestamp_utc)
            elif m.ts:
                # Fallback: parse from ts string
                try:
                    ts_float = float(m.ts.split(".")[0])
                    timestamps.append(datetime.utcfromtimestamp(ts_float))
                except (ValueError, TypeError):
                    pass
        return timestamps

    def _compute_duration(self, timestamps: List[datetime]) -> float:
        """Compute total conversation duration in seconds."""
        if len(timestamps) < 2:
            return 0.0
        return (max(timestamps) - min(timestamps)).total_seconds()

    def _compute_first_response_time(self, messages: List[Message]) -> Optional[float]:
        """
        Compute time to first human response after bot message.

        Returns None if no bot->human response pair exists.
        """
        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]

            # Look for bot followed by human
            if prev_msg.is_bot and not curr_msg.is_bot:
                prev_ts = prev_msg.timestamp_utc
                curr_ts = curr_msg.timestamp_utc

                if prev_ts and curr_ts:
                    return (curr_ts - prev_ts).total_seconds()

                # Fallback to ts parsing
                try:
                    prev_float = float(prev_msg.ts.split(".")[0])
                    curr_float = float(curr_msg.ts.split(".")[0])
                    return curr_float - prev_float
                except (ValueError, TypeError):
                    pass

        return None

    def _compute_avg_response_time(self, messages: List[Message]) -> Optional[float]:
        """
        Compute average response time for all responses.

        A response is any message following a message from a different sender type.
        """
        response_times = []

        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]

            # Only count when sender type changes (bot->human or human->bot)
            if prev_msg.is_bot != curr_msg.is_bot:
                prev_ts = prev_msg.timestamp_utc
                curr_ts = curr_msg.timestamp_utc

                if prev_ts and curr_ts:
                    response_times.append((curr_ts - prev_ts).total_seconds())
                else:
                    # Fallback to ts parsing
                    try:
                        prev_float = float(prev_msg.ts.split(".")[0])
                        curr_float = float(curr_msg.ts.split(".")[0])
                        response_times.append(curr_float - prev_float)
                    except (ValueError, TypeError):
                        pass

        if not response_times:
            return None

        return sum(response_times) / len(response_times)

    def _count_speaker_switches(self, messages: List[Message]) -> int:
        """
        Count the number of bot<->human transitions.

        This measures the "ping-pong" nature of the conversation.
        """
        switches = 0
        for i in range(1, len(messages)):
            if messages[i].is_bot != messages[i - 1].is_bot:
                switches += 1
        return switches

    def _detect_resolution(
        self,
        audit_result: AuditResult,
        timestamps: List[datetime],
        last_activity: Optional[datetime],
        current_time: datetime,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Detect if conversation is resolved and compute time to resolution.

        Returns:
            Tuple of (is_resolved, resolution_type, time_to_resolution_seconds)
        """
        final_status = (
            audit_result.final_status.lower() if audit_result.final_status else ""
        )

        # Map final_status to resolution type
        resolution_type = None
        is_resolved = False

        if "approved" in final_status or "bound" in final_status:
            resolution_type = "approved"
            is_resolved = True
        elif "declined" in final_status or "rejected" in final_status:
            resolution_type = "declined"
            is_resolved = True
        elif "blocked" in final_status or "needs info" in final_status:
            # Not fully resolved yet
            is_resolved = False
            resolution_type = None

        # Compute time to resolution
        ttr = None
        if is_resolved and len(timestamps) >= 2:
            ttr = (max(timestamps) - min(timestamps)).total_seconds()

        return is_resolved, resolution_type, ttr

    def _detect_stalemate(
        self,
        is_resolved: bool,
        last_activity: Optional[datetime],
        current_time: datetime,
    ) -> bool:
        """
        Detect if conversation is stalemate (>72 hours inactive without resolution).
        """
        if is_resolved:
            return False

        if not last_activity:
            return False

        inactive_seconds = (current_time - last_activity).total_seconds()
        return inactive_seconds > STALEMATE_THRESHOLD_SECONDS


def parse_slack_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse Slack timestamp string to datetime.

    Slack timestamps are Unix timestamps with microseconds: "1234567890.123456"
    """
    if not ts:
        return None
    try:
        ts_float = float(ts)
        return datetime.utcfromtimestamp(ts_float)
    except (ValueError, TypeError):
        return None
