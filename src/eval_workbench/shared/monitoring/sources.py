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
    dataset = await source.fetch_items()

    # Slack source
    source = SlackDataSource(
        name="support_channels",
        channel_ids=["C09MAP9HR9D"],
        limit=10,
    )
    dataset = await source.fetch_items()
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

import pandas as pd
from axion.dataset import Dataset, DatasetItem
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
    async def fetch_items(self) -> Dataset:
        """Fetch items to evaluate as a dataset. Each item must have a unique `id`."""
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

    async def fetch_items(self) -> Dataset:
        """Fetch traces from Langfuse and extract DatasetItems into a Dataset."""
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
        return Dataset.create(name=self._name, items=items)


class SlackDataSource(DataSource):
    """Fetch messages from Slack channels and convert to DatasetItems.

    Args:
        name: Name identifier for this source (e.g., "support_channels")
        channel_ids: List of Slack channel IDs to scrape
        limit: Max messages per channel (default: 10)
        scrape_threads: Include thread replies (default: True)
        filter_sender: Only keep messages from this sender
        bot_name: Name used to identify AI vs human messages (default: "Athena")
        bot_names: Optional list of names to identify AI vs human messages
        workspace_domain: Slack workspace subdomain (default: "mgtinsurance")
        drop_if_first_is_user: Drop if first turn is from user (default: False)
        drop_if_all_ai: Drop if all turns are AI (default: False)
        max_concurrent: Max concurrent channel scrapes (default: 2)
        exclude_senders: Optional list of sender names to omit from conversations
        drop_message_regexes: Optional list of regex patterns to drop messages
        strip_citation_block: Remove trailing citation blocks like "[1] ..." lines
        oldest_ts: Optional inclusive lower timestamp bound for channel history pulls
        latest_ts: Optional inclusive upper timestamp bound for channel history pulls
        window_days: Relative lookback window in days (if oldest_ts is unset)
        window_hours: Relative lookback window in hours (if oldest_ts is unset)
        window_minutes: Relative lookback window in minutes (if oldest_ts is unset)
    """

    def __init__(
        self,
        name: str,
        channel_ids: list[str],
        limit: int = 10,
        scrape_threads: bool = True,
        filter_sender: str | None = None,
        bot_name: str = 'Athena',
        bot_names: list[str] | None = None,
        workspace_domain: str = 'mgtinsurance',
        drop_if_first_is_user: bool = False,
        drop_if_all_ai: bool = False,
        max_concurrent: int = 2,
        exclude_senders: list[str] | None = None,
        drop_message_regexes: list[str] | None = None,
        strip_citation_block: bool = False,
        oldest_ts: float | None = None,
        latest_ts: float | None = None,
        window_days: float | None = None,
        window_hours: float | None = None,
        window_minutes: float | None = None,
    ):
        self._name = name
        self._channel_ids = channel_ids
        self._limit = limit
        self._scrape_threads = scrape_threads
        self._filter_sender = filter_sender
        self._bot_name = bot_name
        self._bot_names = bot_names
        self._workspace_domain = workspace_domain
        self._drop_if_first_is_user = drop_if_first_is_user
        self._drop_if_all_ai = drop_if_all_ai
        self._max_concurrent = max_concurrent
        self._exclude_senders = exclude_senders
        self._drop_message_regexes = drop_message_regexes
        self._strip_citation_block = strip_citation_block
        self._oldest_ts = oldest_ts
        self._latest_ts = latest_ts
        self._window_days = window_days
        self._window_hours = window_hours
        self._window_minutes = window_minutes

    @property
    def source_key(self) -> str:
        """Return source key in format 'slack:{name}'."""
        return f'slack:{self._name}'

    async def fetch_items(self) -> Dataset:
        """Fetch messages from Slack and convert to DatasetItems into a Dataset."""
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
            bot_names=self._bot_names,
            workspace_domain=self._workspace_domain,
            drop_if_first_is_user=self._drop_if_first_is_user,
            drop_if_all_ai=self._drop_if_all_ai,
            max_concurrent=self._max_concurrent,
            exclude_senders=self._exclude_senders,
            drop_message_regexes=self._drop_message_regexes,
            strip_citation_block=self._strip_citation_block,
            oldest_ts=self._oldest_ts,
            latest_ts=self._latest_ts,
            window_days=self._window_days,
            window_hours=self._window_hours,
            window_minutes=self._window_minutes,
        )

        items = await exporter.execute()
        logger.info(f'Fetched {len(items)} items from Slack')
        return Dataset.create(name=self._name, items=items)


class SlackNeonJoinDataSource(DataSource):
    """Fetch Slack items, join with Neon rows, and return merged Dataset."""

    _SAFE_SQL_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_\.]*$')

    def __init__(
        self,
        *,
        name: str,
        channel_ids: list[str],
        neon_query: str,
        slack_join_columns: list[str],
        neon_join_columns: list[str],
        dataset_id_column: str | None = None,
        use_slack_thread_dataset_id: bool = False,
        neon_time_column: str = 'created_at',
        buffer_minutes: float = 0.0,
        neon_connection_string: str | None = None,
        neon_chunk_size: int = 1000,
        neon_timeout_seconds: float | None = None,
        limit: int = 10,
        scrape_threads: bool = True,
        filter_sender: str | None = None,
        bot_name: str = 'Athena',
        bot_names: list[str] | None = None,
        workspace_domain: str = 'mgtinsurance',
        drop_if_first_is_user: bool = False,
        drop_if_all_ai: bool = False,
        max_concurrent: int = 2,
        exclude_senders: list[str] | None = None,
        drop_message_regexes: list[str] | None = None,
        strip_citation_block: bool = False,
        oldest_ts: float | None = None,
        latest_ts: float | None = None,
        window_days: float | None = None,
        window_hours: float | None = None,
        window_minutes: float | None = None,
    ):
        if not slack_join_columns or not neon_join_columns:
            raise ValueError('slack_join_columns and neon_join_columns are required')
        if len(slack_join_columns) != len(neon_join_columns):
            raise ValueError(
                'slack_join_columns and neon_join_columns must have equal lengths'
            )
        if buffer_minutes < 0:
            raise ValueError('buffer_minutes must be >= 0')
        if not self._SAFE_SQL_IDENTIFIER.match(neon_time_column):
            raise ValueError(f'Invalid neon_time_column: {neon_time_column}')

        self._name = name
        self._channel_ids = channel_ids
        self._neon_query = neon_query
        self._slack_join_columns = slack_join_columns
        self._neon_join_columns = neon_join_columns
        self._dataset_id_column = dataset_id_column
        self._use_slack_thread_dataset_id = use_slack_thread_dataset_id
        self._neon_time_column = neon_time_column
        self._buffer_minutes = buffer_minutes
        self._neon_connection_string = neon_connection_string
        self._neon_chunk_size = neon_chunk_size
        self._neon_timeout_seconds = neon_timeout_seconds

        self._slack_exporter_kwargs = {
            'channel_ids': channel_ids,
            'limit': limit,
            'scrape_threads': scrape_threads,
            'filter_sender': filter_sender,
            'bot_name': bot_name,
            'bot_names': bot_names,
            'workspace_domain': workspace_domain,
            'drop_if_first_is_user': drop_if_first_is_user,
            'drop_if_all_ai': drop_if_all_ai,
            'max_concurrent': max_concurrent,
            'exclude_senders': exclude_senders,
            'drop_message_regexes': drop_message_regexes,
            'strip_citation_block': strip_citation_block,
            'oldest_ts': oldest_ts,
            'latest_ts': latest_ts,
            'window_days': window_days,
            'window_hours': window_hours,
            'window_minutes': window_minutes,
        }

    @property
    def source_key(self) -> str:
        return f'slack_neon_join:{self._name}'

    @staticmethod
    def _with_time_bounds(
        query: str,
        time_column: str,
        oldest_ts: float | None,
        latest_ts: float | None,
    ) -> tuple[str, tuple[Any, ...] | None]:
        """Append parameterized time predicates to SQL query."""
        if oldest_ts is None and latest_ts is None:
            return query, None

        base_query = query.strip().rstrip(';')
        clauses: list[str] = []
        params: list[Any] = []
        if oldest_ts is not None:
            clauses.append(f'{time_column} >= to_timestamp(%s)')
            params.append(oldest_ts)
        if latest_ts is not None:
            clauses.append(f'{time_column} <= to_timestamp(%s)')
            params.append(latest_ts)

        connector = (
            ' AND ' if re.search(r'\bWHERE\b', base_query, re.IGNORECASE) else ' WHERE '
        )
        return f'{base_query}{connector}{" AND ".join(clauses)}', tuple(params)

    @staticmethod
    def _parse_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _expand_dataset_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'dataset_metadata' not in df.columns:
            return df
        parsed_metadata = df['dataset_metadata'].apply(self._parse_metadata)
        metadata_df = parsed_metadata.apply(pd.Series)
        if metadata_df.empty:
            return df
        return pd.concat([df, metadata_df], axis=1)

    @staticmethod
    def _serialize_metadata(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value)
        return json.dumps({})

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f'Missing {label} columns: {missing}')

    async def fetch_items(self) -> Dataset:
        from eval_workbench.shared.database.neon import AsyncNeonConnection
        from eval_workbench.shared.slack.exporter import SlackExporter

        exporter = SlackExporter(**self._slack_exporter_kwargs)
        slack_items = await exporter.execute()
        if not slack_items:
            return Dataset.create(name=self._name, items=[])

        slack_dataset = Dataset.create(name=self._name, items=slack_items)
        slack_df = self._expand_dataset_metadata(slack_dataset.to_dataframe())

        self._require_columns(slack_df, self._slack_join_columns, 'slack join')

        slack_oldest = exporter.default_oldest_ts
        slack_latest = exporter.default_latest_ts
        buffer_seconds = self._buffer_minutes * 60.0
        neon_oldest = (
            slack_oldest - buffer_seconds if slack_oldest is not None else None
        )
        neon_latest = (
            slack_latest + buffer_seconds if slack_latest is not None else None
        )
        neon_query, neon_params = self._with_time_bounds(
            query=self._neon_query,
            time_column=self._neon_time_column,
            oldest_ts=neon_oldest,
            latest_ts=neon_latest,
        )

        async with AsyncNeonConnection(self._neon_connection_string) as db:
            neon_df = await db.fetch_dataframe_chunked(
                neon_query,
                neon_params,
                chunk_size=self._neon_chunk_size,
                timeout_seconds=self._neon_timeout_seconds,
            )

        if neon_df.empty:
            return Dataset.create(name=self._name, items=[])
        self._require_columns(neon_df, self._neon_join_columns, 'neon join')

        # Keep Slack join keys, but drop any overlapping payload columns so Neon values win.
        # This avoids brittle one-off drops (e.g. trace_id) and prevents *_x/*_y suffix noise.
        slack_join_set = set(self._slack_join_columns)
        neon_columns = set(neon_df.columns.tolist())
        slack_drop_columns = [
            column
            for column in slack_df.columns.tolist()
            if column in neon_columns and column not in slack_join_set
        ]
        slack_df_for_join = slack_df.drop(columns=slack_drop_columns, errors='ignore')

        merged = slack_df_for_join.merge(
            neon_df,
            left_on=self._slack_join_columns,
            right_on=self._neon_join_columns,
            how='inner',
        )

        if merged.empty:
            return Dataset.create(name=self._name, items=[])

        if self._dataset_id_column:
            if self._dataset_id_column not in merged.columns:
                raise ValueError(
                    f'Missing dataset_id_column in merged rows: {self._dataset_id_column}'
                )
            merged['id'] = merged[self._dataset_id_column].astype(str)
        elif self._use_slack_thread_dataset_id:
            required = ['channel_id', 'thread_ts']
            missing = [col for col in required if col not in merged.columns]
            if missing:
                raise ValueError(
                    'Missing required columns for Slack thread dataset id: '
                    f'{missing}. Present columns: {merged.columns.tolist()}'
                )
            merged['id'] = (
                'slack:'
                + merged['channel_id'].astype(str)
                + ':'
                + merged['thread_ts'].astype(str)
            )

        if 'dataset_metadata' in merged.columns:
            merged['dataset_metadata'] = merged['dataset_metadata'].apply(
                self._serialize_metadata
            )

        return Dataset.read_dataframe(merged, ignore_extra_keys=True)


class NeonDataSource(DataSource):
    """Fetch rows from Neon (PostgreSQL) database and extract DatasetItems.

    Args:
        name: Name identifier for this source (e.g., "athena_cases")
        query: SQL query to execute
        extractor: Function to convert row dict -> DatasetItem
        connection_string: Optional database URL (falls back to DATABASE_URL env var)
        params: Optional query parameters (tuple or dict)
        limit: Optional limit to append if query lacks LIMIT clause
        chunk_size: Rows per batch when streaming results from Neon
        timeout_seconds: Optional per-query statement timeout override
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
        chunk_size: int = 1000,
        timeout_seconds: float | None = None,
    ):
        self._name = name
        self._query = query
        self._extractor = extractor
        self._connection_string = connection_string
        self._params = params
        self._limit = limit
        self._has_field = has_field
        self._chunk_size = chunk_size
        self._timeout_seconds = timeout_seconds

    @property
    def source_key(self) -> str:
        return f'neon:{self._name}'

    async def fetch_items(self) -> Dataset:
        from eval_workbench.shared.database.neon import AsyncNeonConnection

        query = self._query
        if self._limit is not None and 'LIMIT' not in query.upper():
            query = f'{query} LIMIT {self._limit}'

        logger.info(f'Executing query for neon source {self._name}')

        rows_fetched = 0
        async with AsyncNeonConnection(self._connection_string) as db:
            items = []
            async for rows in db.fetch_chunks(
                query,
                self._params,
                chunk_size=self._chunk_size,
                timeout_seconds=self._timeout_seconds,
            ):
                rows_fetched += len(rows)
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

        logger.info(f'Extracted {len(items)} items from {rows_fetched} rows')
        return Dataset.create(name=self._name, items=items)
