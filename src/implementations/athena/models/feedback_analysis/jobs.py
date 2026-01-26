"""
Background Jobs for Feedback Analytics

Provides scheduled job functions for computing and storing aggregated metrics.
These jobs can be triggered by cron, task queues, or manual invocation.

Job Types:
- Daily aggregation: Run after midnight for previous day's metrics
- Weekly aggregation: Run on Monday for previous week's metrics
- Monthly aggregation: Run on 1st of month for previous month's metrics
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set, Dict

from implementations.athena.models.feedback_analysis.schema import (
    AggregatedMetrics,
    ConversationMetrics,
    AuditResult,
)
from implementations.athena.models.feedback_analysis.aggregation_service import (
    MetricAggregationService,
)


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


logger = logging.getLogger(__name__)


class AggregationJobRunner:
    """
    Runner for aggregation jobs.

    Coordinates data fetching, aggregation, and storage.
    """

    def __init__(self, db_manager=None):
        """
        Initialize the job runner.

        Args:
            db_manager: Database manager for fetching and storing data
        """
        self.db = db_manager
        self.aggregation_service = MetricAggregationService()

    async def run_daily_aggregation(
        self,
        date: Optional[datetime] = None,
        store_result: bool = True
    ) -> AggregatedMetrics:
        """
        Run daily aggregation for a specific date.

        Args:
            date: The date to aggregate (defaults to yesterday)
            store_result: Whether to store the result in database

        Returns:
            AggregatedMetrics for the day
        """
        if date is None:
            date = _utcnow() - timedelta(days=1)

        period_start = datetime(date.year, date.month, date.day)
        period_end = period_start + timedelta(days=1)

        logger.info(f"Running daily aggregation for {period_start.date()}")

        # Fetch data for the period
        metrics, audit_results, user_ids = await self._fetch_period_data(
            period_start, period_end
        )

        # Compute aggregation
        aggregation = self.aggregation_service.aggregate_daily(
            metrics, audit_results, user_ids, date
        )

        # Store if configured
        if store_result and self.db:
            await self._store_aggregation(aggregation)

        logger.info(
            f"Daily aggregation complete: {aggregation.total_conversations} conversations, "
            f"STP rate: {aggregation.stp_rate:.1%}"
        )

        return aggregation

    async def run_weekly_aggregation(
        self,
        week_start: Optional[datetime] = None,
        store_result: bool = True
    ) -> AggregatedMetrics:
        """
        Run weekly aggregation for a specific week.

        Args:
            week_start: The starting date of the week (defaults to last week's Monday)
            store_result: Whether to store the result in database

        Returns:
            AggregatedMetrics for the week
        """
        if week_start is None:
            # Default to last week's Monday
            today = _utcnow()
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday + 7)
            week_start = datetime(last_monday.year, last_monday.month, last_monday.day)

        period_end = week_start + timedelta(days=7)

        logger.info(f"Running weekly aggregation for week of {week_start.date()}")

        # Fetch data for the period
        metrics, audit_results, user_ids = await self._fetch_period_data(
            week_start, period_end
        )

        # Compute aggregation
        aggregation = self.aggregation_service.aggregate_weekly(
            metrics, audit_results, user_ids, week_start
        )

        # Store if configured
        if store_result and self.db:
            await self._store_aggregation(aggregation)

        logger.info(
            f"Weekly aggregation complete: {aggregation.total_conversations} conversations, "
            f"STP rate: {aggregation.stp_rate:.1%}"
        )

        return aggregation

    async def run_monthly_aggregation(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        store_result: bool = True
    ) -> AggregatedMetrics:
        """
        Run monthly aggregation for a specific month.

        Args:
            year: The year (defaults to last month's year)
            month: The month (defaults to last month)
            store_result: Whether to store the result in database

        Returns:
            AggregatedMetrics for the month
        """
        if year is None or month is None:
            # Default to last month
            today = _utcnow()
            if today.month == 1:
                year = today.year - 1
                month = 12
            else:
                year = today.year
                month = today.month - 1

        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        logger.info(f"Running monthly aggregation for {year}-{month:02d}")

        # Fetch data for the period
        metrics, audit_results, user_ids = await self._fetch_period_data(
            period_start, period_end
        )

        # Compute aggregation
        aggregation = self.aggregation_service.aggregate_monthly(
            metrics, audit_results, user_ids, year, month
        )

        # Store if configured
        if store_result and self.db:
            await self._store_aggregation(aggregation)

        logger.info(
            f"Monthly aggregation complete: {aggregation.total_conversations} conversations, "
            f"STP rate: {aggregation.stp_rate:.1%}"
        )

        return aggregation

    async def _fetch_period_data(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> tuple:
        """
        Fetch metrics and audit results for a time period.

        Returns:
            Tuple of (metrics, audit_results, user_ids)
        """
        if not self.db:
            logger.warning("No database configured, returning empty data")
            return [], [], set()

        # These queries should be customized based on your database schema
        metrics = await self._query_metrics(period_start, period_end)
        audit_results = await self._query_audit_results(period_start, period_end)
        user_ids = await self._query_user_ids(period_start, period_end)

        return metrics, audit_results, user_ids

    async def _query_metrics(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> List[ConversationMetrics]:
        """Query conversation metrics for period. Override for actual implementation."""
        # Reference SQL:
        # SELECT * FROM athena_conversation_metrics
        # WHERE computed_at >= period_start AND computed_at < period_end
        return []

    async def _query_audit_results(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> List[AuditResult]:
        """Query audit results for period. Override for actual implementation."""
        # Reference SQL:
        # SELECT * FROM athena_audit_results
        # WHERE created_at >= period_start AND created_at < period_end
        return []

    async def _query_user_ids(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> Set[str]:
        """Query unique user IDs for period. Override for actual implementation."""
        # Reference SQL:
        # SELECT DISTINCT user_id FROM athena_user_activity
        # WHERE last_active >= period_start AND last_active < period_end
        return set()

    async def _store_aggregation(self, aggregation: AggregatedMetrics) -> None:
        """Store aggregation in database. Override for actual implementation."""
        # Reference SQL:
        # INSERT INTO athena_aggregated_metrics (...)
        # VALUES (...) ON CONFLICT (period_start, period_type) DO UPDATE SET ...
        logger.info(f"Would store aggregation for {aggregation.period_start.date()}")


async def run_all_pending_aggregations(
    runner: AggregationJobRunner,
    since: Optional[datetime] = None
) -> Dict[str, List[AggregatedMetrics]]:
    """
    Run all pending aggregations since a given date.

    Useful for backfilling or catching up after downtime.

    Args:
        runner: The job runner instance
        since: Start date for backfill (defaults to 30 days ago)

    Returns:
        Dict with 'daily', 'weekly', 'monthly' keys containing aggregations
    """
    if since is None:
        since = _utcnow() - timedelta(days=30)

    results = {
        'daily': [],
        'weekly': [],
        'monthly': [],
    }

    now = _utcnow()

    # Run daily aggregations
    current = since
    while current < now:
        aggregation = await runner.run_daily_aggregation(current)
        results['daily'].append(aggregation)
        current += timedelta(days=1)

    # Run weekly aggregations (for complete weeks)
    days_since_monday = since.weekday()
    week_start = since - timedelta(days=days_since_monday)
    week_start = datetime(week_start.year, week_start.month, week_start.day)

    while week_start + timedelta(days=7) <= now:
        aggregation = await runner.run_weekly_aggregation(week_start)
        results['weekly'].append(aggregation)
        week_start += timedelta(days=7)

    # Run monthly aggregations (for complete months)
    year = since.year
    month = since.month

    while True:
        if month == 12:
            month_end = datetime(year + 1, 1, 1)
        else:
            month_end = datetime(year, month + 1, 1)

        if month_end > now:
            break

        aggregation = await runner.run_monthly_aggregation(year, month)
        results['monthly'].append(aggregation)

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return results
