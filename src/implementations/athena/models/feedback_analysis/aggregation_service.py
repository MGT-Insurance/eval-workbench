from datetime import datetime, timedelta
from typing import List, Set

from implementations.athena.models.feedback_analysis.schema import (
    AggregatedMetrics,
    ConversationMetrics,
    AuditResult,
    EscalationType,
    InterventionType,
)


class MetricAggregationService:
    """
    Service for aggregating per-conversation metrics into period-based dashboard data.

    Aggregation periods:
    - daily: Single day aggregation
    - weekly: 7-day aggregation
    - monthly: Calendar month aggregation
    """

    def aggregate_period(
        self,
        metrics: List[ConversationMetrics],
        audit_results: List[AuditResult],
        user_ids: Set[str],
        period_start: datetime,
        period_end: datetime,
        period_type: str = "daily"
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for a time period.

        Args:
            metrics: List of per-conversation metrics
            audit_results: List of audit results (must be same length as metrics)
            user_ids: Set of human user IDs for MAU calculation
            period_start: Start of aggregation period
            period_end: End of aggregation period
            period_type: One of "daily", "weekly", "monthly"

        Returns:
            AggregatedMetrics for the period
        """
        total = len(metrics)

        if total == 0:
            return self._empty_aggregation(period_start, period_end, period_type)

        # Validate alignment
        if len(audit_results) != total:
            raise ValueError(f"Metrics ({total}) and audit_results ({len(audit_results)}) must have same length")

        # Trust Metrics
        mau = len(user_ids)
        stp_count = sum(1 for r in audit_results if not r.has_human_intervention)
        shift_left_rate = stp_count / total

        # Escalation Spectrum counts
        hard_count = sum(1 for r in audit_results if r.escalation_type == EscalationType.HARD)
        soft_count = sum(1 for r in audit_results if r.escalation_type == EscalationType.SOFT)
        authority_count = sum(1 for r in audit_results if r.escalation_type == EscalationType.AUTHORITY)

        # Operational Efficiency
        avg_messages = sum(m.total_messages for m in metrics) / total
        avg_ping_pong = sum(m.ping_pong_ratio for m in metrics) / total

        # Time to resolution (only for resolved conversations)
        turnarounds = [
            m.time_to_resolution_seconds
            for m in metrics
            if m.time_to_resolution_seconds is not None
        ]
        avg_turnaround = sum(turnarounds) / len(turnarounds) if turnarounds else 0.0

        # STP rate
        stp_rate = stp_count / total

        # Conversation Hygiene
        stalemate_count = sum(1 for m in metrics if m.is_stalemate)
        frustrated_count = sum(1 for m in metrics if m.has_frustrated_message)
        clarification_count = sum(
            1 for r in audit_results
            if r.intervention_category == InterventionType.CLARIFICATION
        )

        clarification_rate = clarification_count / total
        stalemate_rate = stalemate_count / total
        frustrated_rate = frustrated_count / total

        # MTTR (same as avg_turnaround for resolved)
        mttr_seconds = avg_turnaround

        # Business Impact
        approved_count = sum(1 for m in metrics if m.resolution_type == 'approved')
        declined_count = sum(1 for m in metrics if m.resolution_type == 'declined')

        # Rates based on resolved conversations
        resolved_count = sum(1 for m in metrics if m.is_resolved)
        approval_rate = approved_count / resolved_count if resolved_count > 0 else 0.0
        decline_rate = declined_count / resolved_count if resolved_count > 0 else 0.0

        return AggregatedMetrics(
            period_start=period_start,
            period_end=period_end,
            period_type=period_type,
            # Trust Metrics
            mau=mau,
            total_conversations=total,
            shift_left_rate=shift_left_rate,
            # Escalation Spectrum
            hard_count=hard_count,
            soft_count=soft_count,
            authority_count=authority_count,
            stp_count=stp_count,
            # Operational Efficiency
            stp_rate=stp_rate,
            avg_messages=avg_messages,
            avg_ping_pong=avg_ping_pong,
            avg_turnaround_seconds=avg_turnaround,
            # Conversation Hygiene
            clarification_rate=clarification_rate,
            mttr_seconds=mttr_seconds,
            stalemate_rate=stalemate_rate,
            frustrated_rate=frustrated_rate,
            # Business Impact
            approval_rate=approval_rate,
            decline_rate=decline_rate,
        )

    def _empty_aggregation(
        self,
        period_start: datetime,
        period_end: datetime,
        period_type: str
    ) -> AggregatedMetrics:
        """Return an empty aggregation for periods with no data."""
        return AggregatedMetrics(
            period_start=period_start,
            period_end=period_end,
            period_type=period_type,
            mau=0,
            total_conversations=0,
            shift_left_rate=0.0,
            hard_count=0,
            soft_count=0,
            authority_count=0,
            stp_count=0,
            stp_rate=0.0,
            avg_messages=0.0,
            avg_ping_pong=0.0,
            avg_turnaround_seconds=0.0,
            clarification_rate=0.0,
            mttr_seconds=0.0,
            stalemate_rate=0.0,
            frustrated_rate=0.0,
            approval_rate=0.0,
            decline_rate=0.0,
        )

    def aggregate_daily(
        self,
        metrics: List[ConversationMetrics],
        audit_results: List[AuditResult],
        user_ids: Set[str],
        date: datetime
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for a single day.

        Args:
            metrics: List of per-conversation metrics for the day
            audit_results: Corresponding audit results
            user_ids: Set of human user IDs
            date: The date to aggregate (time component ignored)

        Returns:
            AggregatedMetrics for the day
        """
        period_start = datetime(date.year, date.month, date.day)
        period_end = period_start + timedelta(days=1)
        return self.aggregate_period(
            metrics, audit_results, user_ids, period_start, period_end, "daily"
        )

    def aggregate_weekly(
        self,
        metrics: List[ConversationMetrics],
        audit_results: List[AuditResult],
        user_ids: Set[str],
        week_start: datetime
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for a week (7 days starting from week_start).

        Args:
            metrics: List of per-conversation metrics for the week
            audit_results: Corresponding audit results
            user_ids: Set of human user IDs
            week_start: The starting date of the week

        Returns:
            AggregatedMetrics for the week
        """
        period_start = datetime(week_start.year, week_start.month, week_start.day)
        period_end = period_start + timedelta(days=7)
        return self.aggregate_period(
            metrics, audit_results, user_ids, period_start, period_end, "weekly"
        )

    def aggregate_monthly(
        self,
        metrics: List[ConversationMetrics],
        audit_results: List[AuditResult],
        user_ids: Set[str],
        year: int,
        month: int
    ) -> AggregatedMetrics:
        """
        Aggregate metrics for a calendar month.

        Args:
            metrics: List of per-conversation metrics for the month
            audit_results: Corresponding audit results
            user_ids: Set of human user IDs
            year: The year
            month: The month (1-12)

        Returns:
            AggregatedMetrics for the month
        """
        period_start = datetime(year, month, 1)

        # Calculate end of month
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        return self.aggregate_period(
            metrics, audit_results, user_ids, period_start, period_end, "monthly"
        )
