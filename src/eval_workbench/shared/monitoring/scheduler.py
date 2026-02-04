"""
APScheduler-based monitoring scheduler for local/dev environments.

For production, use GitHub Actions cron instead.

Example:
    from eval_workbench.shared.monitoring.scheduler import MonitoringScheduler
    from eval_workbench.shared.monitoring.scored_items import ScoredItemsStore

    store = ScoredItemsStore("data/scored_items.csv")
    scheduler = MonitoringScheduler(scored_store=store)

    # Add monitors with different intervals
    scheduler.add_monitor(
        config_path="config/monitoring_langfuse.yaml",
        interval_minutes=60,  # hourly
        job_id="langfuse_hourly",
    )

    scheduler.add_monitor(
        config_path="config/monitoring_slack.yaml",
        interval_minutes=10,  # every 10 minutes
        job_id="slack_frequent",
    )

    # Start and run until interrupted
    scheduler.start()
    try:
        scheduler.wait()  # Blocks until shutdown
    except KeyboardInterrupt:
        scheduler.stop()
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from eval_workbench.shared import config
from eval_workbench.shared.monitoring.monitor import OnlineMonitor
from eval_workbench.shared.monitoring.scored_items import ScoredItemsStore

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """APScheduler-based scheduler for running monitors on intervals.

    Supports both interval-based and cron-based scheduling.

    Args:
        scored_store: Optional ScoredItemsStore for deduplication
        timezone: Timezone for cron schedules (default: "UTC")
    """

    def __init__(
        self,
        scored_store: ScoredItemsStore | None = None,
        timezone: str = 'UTC',
    ):
        self._scored_store = scored_store
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._monitors: dict[str, OnlineMonitor] = {}
        self._running = False

    def add_monitor(
        self,
        config_path: str | Path,
        interval_minutes: int | None = None,
        cron: str | None = None,
        job_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        run_immediately: bool = False,
    ) -> str:
        """Add a monitor to the scheduler.

        Specify either interval_minutes OR cron, not both.

        Args:
            config_path: Path to monitor YAML config
            interval_minutes: Run every N minutes
            cron: Cron expression (e.g., "0 * * * *" for hourly)
            job_id: Optional job ID (auto-generated if not provided)
            overrides: Config overrides to apply
            run_immediately: Run once immediately when added

        Returns:
            Job ID that can be used to remove the monitor

        Example:
            # Every 10 minutes
            scheduler.add_monitor("config.yaml", interval_minutes=10)

            # Hourly via cron
            scheduler.add_monitor("config.yaml", cron="0 * * * *")

            # Daily at 2am
            scheduler.add_monitor("config.yaml", cron="0 2 * * *")
        """
        if interval_minutes is None and cron is None:
            cfg = config.load_config(config_path, overrides)
            schedule_cfg = config.get('schedule', cfg=cfg) or {}
            interval_minutes = schedule_cfg.get('interval_minutes')
            cron = schedule_cfg.get('cron')

        if interval_minutes is None and cron is None:
            raise ValueError(
                'Must specify interval_minutes or cron (or set schedule in YAML)'
            )
        if interval_minutes and cron:
            raise ValueError('Cannot specify both interval_minutes and cron')

        # Load monitor from config
        monitor = OnlineMonitor.from_yaml(
            config_path,
            scored_store=self._scored_store,
            overrides=overrides,
        )

        # Generate job ID if not provided
        if not job_id:
            job_id = f'{monitor.name}_{Path(config_path).stem}'

        self._monitors[job_id] = monitor

        # Create trigger
        if interval_minutes:
            trigger = IntervalTrigger(minutes=interval_minutes)
            schedule_desc = f'every {interval_minutes} minutes'
        else:
            trigger = CronTrigger.from_crontab(cron)
            schedule_desc = f'cron: {cron}'

        # Add job to scheduler
        self._scheduler.add_job(
            self._run_monitor,
            trigger=trigger,
            id=job_id,
            args=[job_id],
            replace_existing=True,
            name=f'Monitor: {monitor.name}',
        )

        logger.info(f"Added monitor '{job_id}' ({schedule_desc})")

        # Run immediately if requested
        if run_immediately:
            asyncio.create_task(self._run_monitor(job_id))

        return job_id

    def remove_monitor(self, job_id: str) -> bool:
        """Remove a monitor from the scheduler.

        Args:
            job_id: Job ID returned from add_monitor()

        Returns:
            True if removed, False if not found
        """
        if job_id not in self._monitors:
            return False

        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass

        del self._monitors[job_id]
        logger.info(f"Removed monitor '{job_id}'")
        return True

    async def _run_monitor(self, job_id: str) -> Any:
        """Execute a monitor job.

        Args:
            job_id: Job ID to run

        Returns:
            Results from monitor.run_async(publish=True)
        """
        monitor = self._monitors.get(job_id)
        if not monitor:
            logger.error(f'Monitor not found: {job_id}')
            return None

        logger.info(f'Running monitor: {job_id}')
        try:
            results = await monitor.run_async(publish=True)
            if results:
                # Get count from results if available
                count = len(results) if hasattr(results, '__len__') else '?'
                logger.info(f"Monitor '{job_id}' completed: {count} items processed")
            else:
                logger.info(f"Monitor '{job_id}' completed: no new items")
            return results
        except Exception as e:
            logger.error(f"Monitor '{job_id}' failed: {e}", exc_info=True)
            return None

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._scheduler.start()
        self._running = True
        logger.info('Monitoring scheduler started')

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info('Monitoring scheduler stopped')

    def wait(self) -> None:
        """Block until the scheduler is stopped.

        Use this in scripts that should run continuously.
        Press Ctrl+C to stop.
        """
        if not self._running:
            raise RuntimeError('Scheduler not started')

        # Create event loop and run until interrupted
        loop = asyncio.get_event_loop()
        try:
            loop.run_forever()
        except (KeyboardInterrupt, SystemExit):
            pass

    @property
    def jobs(self) -> list[dict[str, Any]]:
        """Get list of scheduled jobs.

        Returns:
            List of job info dicts with id, name, next_run_time
        """
        return [
            {
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time,
            }
            for job in self._scheduler.get_jobs()
        ]

    async def run_all_once(self) -> dict[str, Any]:
        """Run all monitors once (useful for testing).

        Returns:
            Dict mapping job_id to results
        """
        results = {}
        for job_id in self._monitors:
            results[job_id] = await self._run_monitor(job_id)
        return results
