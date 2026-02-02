"""
OnlineMonitor that supports pluggable data sources and deduplication.

Supports multiple data sources (Langfuse, Slack, etc.) and tracks scored
items to avoid re-processing.

Example usage:
    from shared.monitoring import OnlineMonitor
    from shared.monitoring.sources import LangfuseDataSource
    from shared.monitoring.scored_items import ScoredItemsStore

    # From YAML config
    store = ScoredItemsStore("data/scored_items.csv")
    monitor = OnlineMonitor.from_yaml("config/monitoring.yaml", scored_store=store)
    results = monitor.run()

    # Programmatic setup
    source = LangfuseDataSource(name="athena", extractor=extract_fn, limit=100)
    monitor = OnlineMonitor(
        name="athena_monitor",
        source=source,
        metrics_config={"Metric": {"class": "metric_class"}},
        scored_store=store,
    )
    results = monitor.run()
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from axion._core.asyncio import run_async_function
from axion.dataset import DatasetItem
from axion.runners import evaluation_runner
from axion.tracing import configure_tracing
from pandas.api import types as ptypes

# -- need this file to be auto loaded for metrics access
from implementations.athena.metrics.recommendation import metric_registry
from shared import config
from shared.config import ConfigurationError
from shared.database.evaluation_upload import EvaluationUploader
from shared.database.neon import NeonConnection
from shared.langfuse.trace import Trace
from shared.monitoring.sampling import (
    AllSampling,
    SamplingStrategy,
    SamplingStrategyType,
)
from shared.monitoring.scored_items import (
    CSVScoredItemsStore,
    DBScoredItemsStore,
    ScoredItemsStore,
)
from shared.monitoring.sources import DataSource, LangfuseDataSource, SlackDataSource

metric_registry.finalize_initial_state()

logger = logging.getLogger(__name__)


# Type alias for extractor functions
ExtractorFn = Callable[[Trace], DatasetItem]


def _load_function(path: str) -> Callable:
    """Load a function from dotted path like 'module.submodule.function'."""
    module_path, func_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _load_class(path: str) -> type:
    """Load a class from dotted path."""
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class OnlineMonitor:
    """
    Generic online monitoring that:
    1. Fetches items from a pluggable data source (Langfuse, Slack, etc.)
    2. Filters out already-scored items (deduplication)
    3. Runs evaluation via axion's evaluation_runner
    4. Records scored items
    5. Publishes results

    Supports both programmatic construction and YAML-based configuration.
    """

    def __init__(
        self,
        name: str,
        source: DataSource,
        metrics_config: dict[str, Any],
        scored_store: ScoredItemsStore | None = None,
        publishing_config: dict[str, Any] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        trace_experiment: bool = False,
        raw_config: dict[str, Any] | None = None,
    ):
        """Initialize monitor with data source and configuration.

        Args:
            name: Monitor name (used for deduplication key)
            source: DataSource instance to fetch items from
            metrics_config: Metrics configuration for evaluation_runner
            scored_store: Optional store for tracking scored items
            publishing_config: Optional publishing configuration
            sampling_strategy: Strategy for sampling items before evaluation
            trace_experiment: Whether to trace evaluations in Langfuse
            raw_config: Raw configuration dictionary
        """
        self.name = name
        self._source = source
        self._metrics_config = metrics_config
        self._scored_store = scored_store
        self._publishing_config = publishing_config or {}
        self._sampling_strategy = sampling_strategy or AllSampling()
        self._trace_experiment = trace_experiment
        self._raw_config = raw_config or {}

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        scored_store: ScoredItemsStore | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> 'OnlineMonitor':
        """Create monitor from YAML config file.

        The config file should specify a source type (langfuse or slack)
        along with the source-specific configuration.

        Args:
            path: Path to YAML config file
            scored_store: Optional store for tracking scored items
            overrides: Dict with dot-notation keys to override config values

        Returns:
            Configured OnlineMonitor instance

        Example config (langfuse):
            name: "athena_monitor"
            source:
              type: langfuse
              name: "athena"
              extractor: "implementations.athena.extractors.extract_recommendation"
              limit: 100
              hours_back: 2
            metrics_config:
              UnderwritingFaithfulness:
                class: "underwriting_faithfulness"

        Example config (slack):
            name: "slack_monitor"
            source:
              type: slack
              name: "support_channels"
              channel_ids: ["C09MAP9HR9D"]
              limit: 10
            metrics_config:
              SlackAnalyzer:
                class: "slack_analyzer"
        """
        cfg = config.load_config(path, overrides)
        if scored_store is None:
            scored_store = cls._build_scored_store(cfg)

        # Get source configuration
        source_cfg = config.get('source', cfg=cfg) or {}
        source_type = source_cfg.get('type', 'langfuse')

        # Build source based on type
        if source_type == 'langfuse':
            source = cls._build_langfuse_source(source_cfg)
        elif source_type == 'slack':
            source = cls._build_slack_source(source_cfg)
        else:
            raise ConfigurationError(f'Unknown source type: {source_type}')

        # Build sampling strategy from config
        sampling_cfg = config.get('sampling', cfg=cfg) or {}
        sampling_strategy = cls._build_sampling_strategy(sampling_cfg)

        # Get trace_experiment from publishing config
        pub_cfg = config.get('publishing', cfg=cfg) or {}
        trace_experiment = config.get('trace_experiment', default=False, cfg=pub_cfg)

        return cls(
            name=config.get('name', cfg=cfg) or 'online_monitoring',
            source=source,
            metrics_config=config.get('metrics_config', cfg=cfg) or {},
            scored_store=scored_store,
            publishing_config=pub_cfg,
            sampling_strategy=sampling_strategy,
            trace_experiment=trace_experiment,
            raw_config=cfg,
        )

    @classmethod
    def _build_scored_store(cls, cfg: dict[str, Any]) -> ScoredItemsStore | None:
        store_cfg = config.get('scored_store', cfg=cfg) or {}
        if not store_cfg:
            return None

        store_type = store_cfg.get('type', 'none')
        if store_type in (None, 'none'):
            return None
        if store_type == 'csv':
            file_path = store_cfg.get('file_path', 'data/scored_items.csv')
            return CSVScoredItemsStore(file_path)
        if store_type in ('db', 'database'):
            connection_string = store_cfg.get('connection_string')
            return DBScoredItemsStore(connection_string=connection_string)

        raise ConfigurationError(
            f'Unknown scored_store.type: {store_type!r}. Expected csv, db, or none.'
        )

    @classmethod
    def _build_langfuse_source(cls, source_cfg: dict) -> LangfuseDataSource:
        """Build LangfuseDataSource from config."""
        # Load extractor function
        extractor_path = source_cfg.get('extractor')
        if not extractor_path:
            raise ConfigurationError("Langfuse source requires 'extractor' path")
        extractor = _load_function(extractor_path)

        # Load prompt patterns (optional)
        patterns_path = source_cfg.get('prompt_patterns')
        patterns = _load_class(patterns_path) if patterns_path else None

        return LangfuseDataSource(
            name=source_cfg.get('name', 'default'),
            extractor=extractor,
            prompt_patterns=patterns,
            limit=source_cfg.get('limit', 100),
            days_back=source_cfg.get('days_back'),
            hours_back=source_cfg.get('hours_back'),
            tags=source_cfg.get('tags'),
            timeout=source_cfg.get('timeout', 60),
            fetch_full_traces=source_cfg.get('fetch_full_traces', True),
            show_progress=source_cfg.get('show_progress', True),
        )

    @classmethod
    def _build_slack_source(cls, source_cfg: dict) -> SlackDataSource:
        """Build SlackDataSource from config."""
        channel_ids = source_cfg.get('channel_ids', [])
        if not channel_ids:
            raise ConfigurationError("Slack source requires 'channel_ids'")

        return SlackDataSource(
            name=source_cfg.get('name', 'default'),
            channel_ids=channel_ids,
            limit=source_cfg.get('limit', 10),
            scrape_threads=source_cfg.get('scrape_threads', True),
            filter_sender=source_cfg.get('filter_sender'),
            bot_name=source_cfg.get('bot_name', 'Athena'),
            workspace_domain=source_cfg.get('workspace_domain', 'mgtinsurance'),
            drop_if_first_is_user=source_cfg.get('drop_if_first_is_user', False),
            drop_if_all_ai=source_cfg.get('drop_if_all_ai', False),
            max_concurrent=source_cfg.get('max_concurrent', 2),
        )

    @classmethod
    def _build_sampling_strategy(cls, sampling_cfg: dict) -> SamplingStrategy:
        """Build SamplingStrategy from config.

        Args:
            sampling_cfg: Sampling configuration dict with keys:
                - strategy: Strategy name (all, random, most_recent, oldest)
                - n: Number of items to sample (for non-all strategies)
                - seed: Random seed (for random strategy only)

        Returns:
            Configured SamplingStrategy instance
        """
        strategy_name = sampling_cfg.get('strategy', 'all')
        try:
            strategy_type = SamplingStrategyType(strategy_name)
        except ValueError:
            raise ConfigurationError(f'Unknown sampling strategy: {strategy_name}')

        return strategy_type.create(
            n=sampling_cfg.get('n'),
            seed=sampling_cfg.get('seed'),
        )

    def sample_items(self, items: list[DatasetItem]) -> list[DatasetItem]:
        """Apply sampling strategy to items.

        Args:
            items: List of items to sample from

        Returns:
            Sampled subset of items
        """
        sampled = self._sampling_strategy.sample(items)
        logger.info(f'Sampling: {len(items)} available, {len(sampled)} sampled')
        return sampled

    def filter_unscored_items(self, items: list[DatasetItem]) -> list[DatasetItem]:
        """Filter out items that have already been scored.

        Args:
            items: List of items to filter

        Returns:
            List of items that haven't been scored yet
        """
        if not self._scored_store:
            return items

        scored_ids = self._scored_store.get_scored_item_ids(self._source.source_key)
        unscored = [item for item in items if item.id not in scored_ids]

        logger.info(
            f'Dedup: {len(items)} fetched, {len(scored_ids)} already scored, '
            f'{len(unscored)} to process'
        )
        return unscored

    def record_scored_items(self, items: list[DatasetItem]) -> None:
        """Record items as scored in the store.

        Args:
            items: List of items that were successfully scored
        """
        if not self._scored_store:
            return

        item_ids = [str(item.id) for item in items if item.id]
        self._scored_store.record_scored_items(self._source.source_key, item_ids)

    def _infer_sql_type(self, series: pd.Series) -> str:
        if ptypes.is_bool_dtype(series):
            return 'BOOLEAN'
        if ptypes.is_integer_dtype(series):
            return 'BIGINT'
        if ptypes.is_float_dtype(series):
            return 'DOUBLE PRECISION'
        if ptypes.is_datetime64_any_dtype(series):
            return 'TIMESTAMP'
        if ptypes.is_timedelta64_dtype(series):
            return 'INTERVAL'
        return 'TEXT'

    def _prepare_dataframe_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        for column in prepared.columns:
            series = prepared[column]
            if ptypes.is_object_dtype(series):
                if series.map(lambda value: isinstance(value, (dict, list))).any():
                    prepared[column] = series.map(
                        lambda value: json.dumps(value, default=str)
                        if value is not None
                        else None
                    )
        return prepared

    def _push_to_db(self, results: Any) -> None:
        dataset_df, metrics_df = results.to_normalized_dataframes()
        dataset_df = dataset_df.copy() if dataset_df is not None else pd.DataFrame()
        metrics_df = metrics_df.copy() if metrics_df is not None else pd.DataFrame()

        cfg = self._raw_config or {}
        source_cfg = config.get('source', cfg=cfg) or {}
        source_name = source_cfg.get('name')
        source_type = source_cfg.get('type')
        source_component = source_cfg.get('component', 'agent')
        environment = source_cfg.get('environment', 'preview')
        eval_mode = source_cfg.get('eval_mode')

        for df in (dataset_df, metrics_df):
            if not df.empty:
                df['source_name'] = source_name
                df['source_type'] = source_type
                # Match the evaluation table schema name.
                df['source_component'] = source_component
                df['environment'] = environment

        if not metrics_df.empty and eval_mode is not None:
            metrics_df['eval_mode'] = eval_mode

        if dataset_df.empty and metrics_df.empty:
            logger.info('No rows to push to DB')
            return

        # Get database config from publishing section
        pub_cfg = self._publishing_config or {}
        db_cfg = config.get('database', cfg=pub_cfg) or {}
        connection_string = db_cfg.get('connection_string')
        on_conflict = db_cfg.get('on_conflict', 'do_nothing')
        chunk_size = db_cfg.get('chunk_size', 1000)
        print(f'Pushing to DB: {connection_string}, {on_conflict}, {chunk_size}')
        with NeonConnection(connection_string=connection_string) as db:
            uploader = EvaluationUploader(
                db=db, on_conflict=on_conflict, chunk_size=chunk_size
            )
            if not dataset_df.empty:
                uploader.upload_dataset(dataset_df)
            if not metrics_df.empty:
                uploader.upload_results(metrics_df, dataset_id_source='id')

    def run(
        self,
        deduplicate: bool = True,
        publish: bool = False,
        dataset_name: str | None = None,
        run_name: str | None = None,
        link_to_traces: bool = True,
        metric_names: list[str] | None = None,
        trace_experiment: bool | None = None,
    ) -> Any:
        """Run the monitoring pipeline synchronously."""
        return run_async_function(
            self.run_async,
            deduplicate=deduplicate,
            publish=publish,
            dataset_name=dataset_name,
            run_name=run_name,
            link_to_traces=link_to_traces,
            metric_names=metric_names,
            trace_experiment=trace_experiment,
        )

    async def run_async(
        self,
        deduplicate: bool = True,
        publish: bool = False,
        dataset_name: str | None = None,
        run_name: str | None = None,
        link_to_traces: bool = True,
        metric_names: list[str] | None = None,
        trace_experiment: bool | None = None,
    ) -> Any:
        """Run the monitoring pipeline asynchronously.

        1. Fetch items from data source
        2. Filter already-scored items (if deduplicate=True)
        3. Apply sampling strategy
        4. Run evaluation
        5. Record scored items
        6. Publish results (optional)

        Args:
            deduplicate: Whether to skip already-scored items (default: True)
            publish: Whether to publish results after evaluation
            dataset_name: Override config dataset_name (publishing only)
            run_name: Override config run_name (publishing only)
            link_to_traces: Link scores to source traces (publishing only)
            metric_names: Filter which metrics to publish (publishing only)
            trace_experiment: Whether to trace the experiment (None uses config default)
        Returns:
            EvaluationResults from axion's evaluation_runner, or None if no items
        """
        # Fetch items from source
        items = await self._source.fetch_items()
        if not items:
            logger.warning('No items found from source')
            return None

        # Filter already-scored items
        if deduplicate:
            items = self.filter_unscored_items(items)
            if not items:
                logger.info('All items already scored, nothing to process')
                return None

        # Apply sampling strategy
        items = self.sample_items(items)
        if not items:
            logger.info('No items to process after sampling')
            return None

        # Run evaluation
        if not self._metrics_config:
            raise ConfigurationError('metrics_config required')

        logger.info(
            f'Evaluating {len(items)} items with metrics: {list(self._metrics_config.keys())}'
        )

        # Use config default if not overridden at runtime
        should_trace = (
            trace_experiment if trace_experiment is not None else self._trace_experiment
        )

        if should_trace:
            configure_tracing('langfuse')
        else:
            configure_tracing('noop')

        results = evaluation_runner(
            evaluation_inputs=items,
            scoring_strategy='flat',
            scoring_config={'metric': self._metrics_config},
            evaluation_name=self.name,
            trace_granularity='single',
        )

        if not should_trace:
            configure_tracing('langfuse')

        # Record scored items
        self.record_scored_items(items)

        if publish:
            pub_cfg = self._publishing_config

            # Publish as experiment if enabled
            if config.get('experiment.enabled', default=True, cfg=pub_cfg):
                results.publish_as_experiment(
                    dataset_name=dataset_name
                    or config.get('experiment.dataset_name', cfg=pub_cfg),
                    run_name=run_name or config.get('experiment.run_name', cfg=pub_cfg),
                    link_to_traces=link_to_traces,
                    metric_names=metric_names
                    or config.get('experiment.metrics', cfg=pub_cfg),
                )
                logger.info('Published as experiment')
            elif config.get('push_to_langfuse', cfg=pub_cfg):
                results.publish_to_observability()
                logger.info('Published to Langfuse')
            else:
                logger.info('No publishing configuration found')

            if config.get('push_to_db', cfg=pub_cfg):
                self._push_to_db(results)

        logger.info('Evaluation complete')
        return results
