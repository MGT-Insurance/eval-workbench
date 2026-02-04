"""
Data source abstractions for online monitoring.

Provides pluggable data sources that fetch items for evaluation:
- LangfuseDataSource: Fetch traces from Langfuse
- SlackDataSource: Fetch messages from Slack channels

Example:
    from eval_workbench.shared.monitoring.sources import LangfuseDataSource, SlackDataSource

    # Langfuse source
    source = LangfuseDataSource(
        name="athena",
        extractor=extract_recommendation,
        limit=100,
        hours_back=2,
    )
    items = await source.fetch_items()

    # Slack source
    source = SlackDataSource(
        name="support_channels",
        channel_ids=["C09MAP9HR9D"],
        limit=10,
    )
    items = await source.fetch_items()
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from axion.dataset import DatasetItem
from axion.tracing import LangfuseTraceLoader

from eval_workbench.shared.langfuse.trace import (
    PromptPatternsBase,
    Trace,
    TraceCollection,
)

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base for data sources that provide items to evaluate."""

    @property
    @abstractmethod
    def source_key(self) -> str:
        """Unique identifier for this source (used in deduplication)."""
        pass

    @abstractmethod
    async def fetch_items(self) -> list[DatasetItem]:
        """Fetch items to evaluate. Each item must have a unique `id`."""
        pass


class LangfuseDataSource(DataSource):
    """Fetch traces from Langfuse and extract DatasetItems.

    Args:
        name: Name identifier for this source (e.g., "athena")
        extractor: Function to convert Trace -> DatasetItem
        prompt_patterns: Optional prompt patterns class for variable extraction
        limit: Max traces to fetch (default: 100)
        days_back: Fetch traces from last N days
        hours_back: Fetch traces from last N hours
        tags: Filter traces by tags
        trace_ids: Specific trace IDs to fetch (bypasses time-based fetch)
        timeout: API timeout in seconds (default: 60)
        fetch_full_traces: Whether to fetch full trace data (default: True)
        show_progress: Show progress bar during fetch (default: True)
    """

    def __init__(
        self,
        name: str,
        extractor: Callable[[Trace], DatasetItem],
        prompt_patterns: type[PromptPatternsBase] | None = None,
        limit: int = 100,
        days_back: int | None = None,
        hours_back: int | None = None,
        minutes_back: int | None = None,
        tags: list[str] | None = None,
        trace_ids: list[str] | None = None,
        timeout: int = 60,
        fetch_full_traces: bool = True,
        show_progress: bool = True,
    ):
        self._name = name
        self._extractor = extractor
        self._prompt_patterns = prompt_patterns
        self._limit = limit
        self._days_back = days_back
        self._hours_back = hours_back
        self._minutes_back = minutes_back
        self._tags = tags
        self._trace_ids = trace_ids
        self._timeout = timeout
        self._fetch_full_traces = fetch_full_traces
        self._show_progress = show_progress

    @property
    def source_key(self) -> str:
        """Return source key in format 'langfuse:{name}'."""
        return f'langfuse:{self._name}'

    def _fetch_traces_by_time(self) -> TraceCollection:
        """Fetch traces using time-based filtering."""
        loader = LangfuseTraceLoader(timeout=self._timeout)

        hours_back = self._hours_back
        if (
            self._days_back is None
            and hours_back is None
            and self._minutes_back is not None
        ):
            hours_back = self._minutes_back / 60.0

        logger.info(
            f'Fetching traces: name={self._name}, limit={self._limit}, '
            f'days_back={self._days_back}, hours_back={hours_back}, tags={self._tags}'
        )

        fetch_kwargs = {
            'limit': self._limit,
            'name': self._name,
            'tags': self._tags,
            'fetch_full_traces': self._fetch_full_traces,
            'show_progress': self._show_progress,
        }
        if self._days_back is not None:
            fetch_kwargs['days_back'] = self._days_back
            fetch_kwargs['mode'] = 'days_back'
        elif hours_back is not None:
            fetch_kwargs['hours_back'] = hours_back
            fetch_kwargs['mode'] = 'hours_back'

        trace_data = loader.fetch_traces(**fetch_kwargs)
        return TraceCollection(trace_data, prompt_patterns=self._prompt_patterns)

    def _fetch_traces_by_ids(self, trace_ids: list[str]) -> TraceCollection:
        """Fetch specific traces by their IDs."""
        logger.info(f'Fetching {len(trace_ids)} traces by ID')

        loader = LangfuseTraceLoader(timeout=self._timeout)
        trace_data = loader.fetch_traces(
            trace_ids=trace_ids,
            show_progress=self._show_progress,
        )

        return TraceCollection(trace_data, prompt_patterns=self._prompt_patterns)

    async def fetch_items(self) -> list[DatasetItem]:
        """Fetch traces from Langfuse and extract DatasetItems."""
        # If specific trace_ids provided, fetch those directly
        if self._trace_ids:
            collection = self._fetch_traces_by_ids(self._trace_ids)
        else:
            collection = self._fetch_traces_by_time()

        logger.info(f'Fetched {len(collection)} traces')

        items = []
        for trace in collection:
            try:
                item = self._extractor(trace)
                # Ensure item has trace_id for deduplication
                if not item.id:
                    item.id = str(getattr(trace, 'id', ''))
                items.append(item)
            except Exception as e:
                trace_id = getattr(trace, 'id', 'unknown')
                logger.warning(f'Failed to extract trace {trace_id}: {e}')

        logger.info(f'Extracted {len(items)} items from {len(collection)} traces')
        return items


class SlackDataSource(DataSource):
    """Fetch messages from Slack channels and convert to DatasetItems.

    Args:
        name: Name identifier for this source (e.g., "support_channels")
        channel_ids: List of Slack channel IDs to scrape
        limit: Max messages per channel (default: 10)
        scrape_threads: Include thread replies (default: True)
        filter_sender: Only keep messages from this sender
        bot_name: Name used to identify AI vs human messages (default: "Athena")
        workspace_domain: Slack workspace subdomain (default: "mgtinsurance")
        drop_if_first_is_user: Drop if first turn is from user (default: False)
        drop_if_all_ai: Drop if all turns are AI (default: False)
        max_concurrent: Max concurrent channel scrapes (default: 2)
    """

    def __init__(
        self,
        name: str,
        channel_ids: list[str],
        limit: int = 10,
        scrape_threads: bool = True,
        filter_sender: str | None = None,
        bot_name: str = 'Athena',
        workspace_domain: str = 'mgtinsurance',
        drop_if_first_is_user: bool = False,
        drop_if_all_ai: bool = False,
        max_concurrent: int = 2,
    ):
        self._name = name
        self._channel_ids = channel_ids
        self._limit = limit
        self._scrape_threads = scrape_threads
        self._filter_sender = filter_sender
        self._bot_name = bot_name
        self._workspace_domain = workspace_domain
        self._drop_if_first_is_user = drop_if_first_is_user
        self._drop_if_all_ai = drop_if_all_ai
        self._max_concurrent = max_concurrent

    @property
    def source_key(self) -> str:
        """Return source key in format 'slack:{name}'."""
        return f'slack:{self._name}'

    async def fetch_items(self) -> list[DatasetItem]:
        """Fetch messages from Slack and convert to DatasetItems."""
        from eval_workbench.shared.slack.exporter import SlackExporter

        logger.info(
            f'Fetching Slack messages: channels={self._channel_ids}, '
            f'limit={self._limit}, scrape_threads={self._scrape_threads}'
        )

        exporter = SlackExporter(
            channel_ids=self._channel_ids,
            limit=self._limit,
            scrape_threads=self._scrape_threads,
            filter_sender=self._filter_sender,
            bot_name=self._bot_name,
            workspace_domain=self._workspace_domain,
            drop_if_first_is_user=self._drop_if_first_is_user,
            drop_if_all_ai=self._drop_if_all_ai,
            max_concurrent=self._max_concurrent,
        )

        items = await exporter.execute()
        logger.info(f'Fetched {len(items)} items from Slack')
        return items


class NeonDataSource(DataSource):
    """Fetch rows from Neon (PostgreSQL) database and extract DatasetItems.

    Args:
        name: Name identifier for this source (e.g., "athena_cases")
        query: SQL query to execute
        extractor: Function to convert row dict -> DatasetItem
        connection_string: Optional database URL (falls back to DATABASE_URL env var)
        params: Optional query parameters (tuple or dict)
        limit: Optional limit to append if query lacks LIMIT clause
    """

    def __init__(
        self,
        name: str,
        query: str,
        extractor: Callable[[dict[str, Any]], DatasetItem],
        connection_string: str | None = None,
        params: tuple | dict | None = None,
        limit: int | None = None,
        has_field: str | None = None,
    ):
        self._name = name
        self._query = query
        self._extractor = extractor
        self._connection_string = connection_string
        self._params = params
        self._limit = limit
        self._has_field = has_field

    @property
    def source_key(self) -> str:
        return f'neon:{self._name}'

    async def fetch_items(self) -> list[DatasetItem]:
        from eval_workbench.shared.database.neon import AsyncNeonConnection

        query = self._query
        if self._limit is not None and 'LIMIT' not in query.upper():
            query = f'{query} LIMIT {self._limit}'

        logger.info(f'Executing query for neon source {self._name}')

        async with AsyncNeonConnection(self._connection_string) as db:
            rows = await db.fetch_all(query, self._params)

        logger.info(f'Fetched {len(rows)} rows from database')

        items = []
        for row in rows:
            try:
                if self._has_field and len(row.get(self._has_field, [])) == 0:
                    continue
                item = self._extractor(row)
                if not item.id:
                    item.id = str(row.get('id', ''))
                items.append(item)
            except Exception as e:
                row_id = row.get('id', 'unknown')
                logger.warning(f'Failed to extract row {row_id}: {e}')

        logger.info(f'Extracted {len(items)} items from {len(rows)} rows')
        return items
