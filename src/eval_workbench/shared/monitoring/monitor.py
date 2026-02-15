import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from axion._core.asyncio import run_async_function
from axion.dataset import Dataset, DatasetItem
from axion.metrics import metric_registry
from axion.runners import evaluation_runner
from axion.tracing import configure_tracing
from pandas.api import types as ptypes

# -- need this file to be auto loaded for metrics access
from eval_workbench.implementations.athena.metrics import (
    recommendation as _recommendation_metrics,  # noqa: F401
)
from eval_workbench.implementations.athena.metrics import slack as _slack_metrics  # noqa: F401
from eval_workbench.shared import config
from eval_workbench.shared.config import ConfigurationError
from eval_workbench.shared.database.evaluation_upload import EvaluationUploader
from eval_workbench.shared.database.neon import NeonConnection
from eval_workbench.shared.langfuse.trace import Trace
from eval_workbench.shared.monitoring.sampling import (
    AllSampling,
    SamplingStrategy,
    SamplingStrategyType,
)
from eval_workbench.shared.monitoring.scored_items import (
    CSVScoredItemsStore,
    DBScoredItemsStore,
    ScoredItemsStore,
)
from eval_workbench.shared.monitoring.sources import (
    DataSource,
    LangfuseDataSource,
    NeonDataSource,
    SlackDataSource,
    SlackNeonJoinDataSource,
)

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
        evaluation_config: dict[str, Any] | None = None,
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
            evaluation_config: Configuration for evaluation_runner parameters
        """
        self.name = name
        self._source = source
        self._metrics_config = metrics_config
        self._scored_store = scored_store
        self._publishing_config = publishing_config or {}
        self._sampling_strategy = sampling_strategy or AllSampling()
        self._trace_experiment = trace_experiment
        self._raw_config = raw_config or {}
        self._evaluation_config = evaluation_config or {}

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
              UWFaithfulness:
                class: "uw_faithfulness"

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
        elif source_type == 'slack_neon_join':
            source = cls._build_slack_neon_join_source(source_cfg)
        elif source_type == 'neon':
            source = cls._build_neon_source(source_cfg)
        else:
            raise ConfigurationError(f'Unknown source type: {source_type}')

        # Build sampling strategy from config
        sampling_cfg = config.get('sampling', cfg=cfg) or {}
        sampling_strategy = cls._build_sampling_strategy(sampling_cfg)

        # Get trace_experiment from publishing config
        pub_cfg = config.get('publishing', cfg=cfg) or {}
        trace_experiment = config.get('trace_experiment', default=False, cfg=pub_cfg)

        # Build evaluation config
        evaluation_cfg = config.get('evaluation', cfg=cfg) or {}
        evaluation_config = cls._build_evaluation_config(evaluation_cfg)

        return cls(
            name=config.get('name', cfg=cfg) or 'online_monitoring',
            source=source,
            metrics_config=config.get('metrics_config', cfg=cfg) or {},
            scored_store=scored_store,
            publishing_config=pub_cfg,
            sampling_strategy=sampling_strategy,
            trace_experiment=trace_experiment,
            raw_config=cfg,
            evaluation_config=evaluation_config,
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
            minutes_back=source_cfg.get('minutes_back'),
            tags=source_cfg.get('tags'),
            trace_ids=source_cfg.get('trace_ids'),
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
            bot_names=source_cfg.get('bot_names'),
            workspace_domain=source_cfg.get('workspace_domain', 'mgtinsurance'),
            drop_if_first_is_user=source_cfg.get('drop_if_first_is_user', False),
            drop_if_all_ai=source_cfg.get('drop_if_all_ai', False),
            max_concurrent=source_cfg.get('max_concurrent', 2),
            exclude_senders=source_cfg.get('exclude_senders'),
            drop_message_regexes=source_cfg.get('drop_message_regexes'),
            strip_citation_block=source_cfg.get('strip_citation_block', False),
            oldest_ts=source_cfg.get('oldest_ts'),
            latest_ts=source_cfg.get('latest_ts'),
            window_days=source_cfg.get('window_days'),
            window_hours=source_cfg.get('window_hours'),
            window_minutes=source_cfg.get('window_minutes'),
        )

    @classmethod
    def _build_neon_source(cls, source_cfg: dict) -> NeonDataSource:
        """Build NeonDataSource from config."""
        extractor_path = source_cfg.get('extractor')
        if not extractor_path:
            raise ConfigurationError("Neon source requires 'extractor' path")
        extractor = _load_function(extractor_path)

        query = source_cfg.get('query')
        if not query:
            raise ConfigurationError("Neon source requires 'query'")

        return NeonDataSource(
            name=source_cfg.get('name', 'default'),
            query=query,
            extractor=extractor,
            connection_string=source_cfg.get('connection_string'),
            params=source_cfg.get('params'),
            limit=source_cfg.get('limit'),
            chunk_size=source_cfg.get('chunk_size', 1000),
            timeout_seconds=source_cfg.get('timeout_seconds'),
        )

    @classmethod
    def _build_slack_neon_join_source(cls, source_cfg: dict) -> SlackNeonJoinDataSource:
        """Build SlackNeonJoinDataSource from config."""
        channel_ids = source_cfg.get('channel_ids', [])
        if not channel_ids:
            raise ConfigurationError("Slack-neon join source requires 'channel_ids'")

        neon_query = source_cfg.get('neon_query')
        if not neon_query:
            raise ConfigurationError("Slack-neon join source requires 'neon_query'")

        slack_join_columns = source_cfg.get('slack_join_columns')
        neon_join_columns = source_cfg.get('neon_join_columns')
        if not slack_join_columns or not neon_join_columns:
            raise ConfigurationError(
                "Slack-neon join source requires 'slack_join_columns' and 'neon_join_columns'"
            )

        return SlackNeonJoinDataSource(
            name=source_cfg.get('name', 'default'),
            channel_ids=channel_ids,
            neon_query=neon_query,
            slack_join_columns=slack_join_columns,
            neon_join_columns=neon_join_columns,
            dataset_id_column=source_cfg.get('dataset_id_column'),
            use_slack_thread_dataset_id=source_cfg.get(
                'use_slack_thread_dataset_id', False
            ),
            neon_time_column=source_cfg.get('neon_time_column', 'created_at'),
            buffer_minutes=source_cfg.get('buffer_minutes', 0),
            neon_connection_string=source_cfg.get('connection_string'),
            neon_chunk_size=source_cfg.get('chunk_size', 1000),
            neon_timeout_seconds=source_cfg.get('timeout_seconds'),
            limit=source_cfg.get('limit', 10),
            scrape_threads=source_cfg.get('scrape_threads', True),
            filter_sender=source_cfg.get('filter_sender'),
            bot_name=source_cfg.get('bot_name', 'Athena'),
            bot_names=source_cfg.get('bot_names'),
            workspace_domain=source_cfg.get('workspace_domain', 'mgtinsurance'),
            drop_if_first_is_user=source_cfg.get('drop_if_first_is_user', False),
            drop_if_all_ai=source_cfg.get('drop_if_all_ai', False),
            max_concurrent=source_cfg.get('max_concurrent', 2),
            exclude_senders=source_cfg.get('exclude_senders'),
            drop_message_regexes=source_cfg.get('drop_message_regexes'),
            strip_citation_block=source_cfg.get('strip_citation_block', False),
            oldest_ts=source_cfg.get('oldest_ts'),
            latest_ts=source_cfg.get('latest_ts'),
            window_days=source_cfg.get('window_days'),
            window_hours=source_cfg.get('window_hours'),
            window_minutes=source_cfg.get('window_minutes'),
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

    @classmethod
    def _build_evaluation_config(cls, eval_cfg: dict) -> dict[str, Any]:
        """Build evaluation_runner kwargs from config.

        Args:
            eval_cfg: Evaluation configuration dict from YAML

        Returns:
            Dict of kwargs to pass to evaluation_runner
        """
        from axion._core.cache.schema import CacheConfig
        from axion.schema import ErrorConfig

        result: dict[str, Any] = {}

        # Simple params that map directly
        for key in (
            'max_concurrent',
            'throttle_delay',
            'show_progress',
            'scoring_strategy',
            'scoring_key_mapping',
            'trace_granularity',
            'flush_per_metric',
            'thresholds',
            'model',  # For hierarchical scoring
            'weights',  # For hierarchical scoring
        ):
            if key in eval_cfg:
                result[key] = eval_cfg[key]

        # Rename mappings
        if 'description' in eval_cfg:
            result['evaluation_description'] = eval_cfg['description']
        if 'metadata' in eval_cfg:
            result['evaluation_metadata'] = eval_cfg['metadata']

        # Cache config
        cache_cfg = eval_cfg.get('cache', {})
        if cache_cfg:
            result['enable_internal_caching'] = cache_cfg.get('enabled', True)
            result['cache_config'] = CacheConfig(
                use_cache=cache_cfg.get('use_cache', True),
                write_cache=cache_cfg.get('write_cache', True),
                cache_type=cache_cfg.get('cache_type', 'memory'),
                cache_dir=cache_cfg.get('cache_dir', '.cache'),
                cache_task=cache_cfg.get('cache_task', True),
            )

        # Error config
        errors_cfg = eval_cfg.get('errors', {})
        if errors_cfg:
            result['error_config'] = ErrorConfig(
                ignore_errors=errors_cfg.get('ignore_errors', True),
                skip_on_missing_params=errors_cfg.get('skip_on_missing_params', False),
            )

        return result

    @staticmethod
    def _clone_dataset(dataset: Dataset, items: list[DatasetItem]) -> Dataset:
        return Dataset(
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            created_at=dataset.created_at,
            metadata=dataset.metadata,
            items=items,
        )

    def sample_items(self, dataset: Dataset) -> Dataset:
        """Apply sampling strategy to dataset items.

        Args:
            dataset: Dataset to sample from

        Returns:
            Dataset containing the sampled subset of items
        """
        sampled = self._sampling_strategy.sample(dataset.items)
        logger.info(f'Sampling: {len(dataset.items)} available, {len(sampled)} sampled')
        return self._clone_dataset(dataset, sampled)

    @staticmethod
    def _run_timestamp() -> str:
        return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    def filter_unscored_items(self, dataset: Dataset) -> Dataset:
        """Filter out items that have already been scored and deduplicate within batch.

        Args:
            dataset: Dataset to filter

        Returns:
            Dataset containing unique items that haven't been scored yet
        """
        scored_id_set: set[str] = set()
        if self._scored_store:
            scored_ids = self._scored_store.get_scored_item_ids(self._source.source_key)
            scored_id_set = set(scored_ids)

        pulled_ids = [item.id for item in dataset.items if item.id]
        pulled_unique_ids = set(pulled_ids)
        pulled_dups = len(pulled_ids) - len(pulled_unique_ids)

        already_scored_matches = sum(
            1 for item_id in pulled_ids if item_id in scored_id_set
        )

        # Deduplicate within batch AND filter out already-scored items
        seen_ids: set[str] = set()
        unscored: list[DatasetItem] = []
        for item in dataset.items:
            if item.id and item.id not in scored_id_set and item.id not in seen_ids:
                unscored.append(item)
                seen_ids.add(item.id)

        logger.info(
            f'Dedup: fetched={len(dataset.items)} (unique_ids={len(pulled_unique_ids)}, '
            f'pulled_dups={pulled_dups}), already_scored_matches={already_scored_matches} '
            f'(scored_store_size={len(scored_id_set)}), to_process={len(unscored)}'
        )
        return self._clone_dataset(dataset, unscored)

    def record_scored_items(self, dataset: Dataset) -> None:
        """Record items as scored in the store.

        Args:
            dataset: Dataset whose items were successfully scored
        """
        if not self._scored_store:
            return

        item_ids = [str(item.id) for item in dataset.items if item.id]
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
                has_json = (
                    series.map(lambda value: isinstance(value, (dict, list)))
                    .to_numpy(dtype=bool, na_value=False)
                    .any()
                )
                if has_json:
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
        source_type = source_cfg.get('db_source_type') or source_cfg.get('type')
        source_component = source_cfg.get('component', 'agent')
        environment = source_cfg.get('environment')
        eval_mode = source_cfg.get('eval_mode')
        monitor_version = config.get('version', cfg=cfg)

        for df in (dataset_df, metrics_df):
            if not df.empty:
                df['source_name'] = source_name
                df['source_type'] = source_type
                # Match the evaluation table schema name.
                df['source_component'] = source_component
                df['environment'] = environment

        if not metrics_df.empty and eval_mode is not None:
            metrics_df['eval_mode'] = eval_mode
        if not metrics_df.empty and monitor_version is not None:
            metrics_df['version'] = monitor_version

        if dataset_df.empty and metrics_df.empty:
            logger.info('No rows to push to DB')
            return

        # Get database config from publishing section
        pub_cfg = self._publishing_config or {}
        db_cfg = config.get('database', cfg=pub_cfg) or {}
        connection_string = db_cfg.get('connection_string')
        on_conflict = db_cfg.get('on_conflict', 'do_nothing')
        chunk_size = db_cfg.get('chunk_size', 1000)
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
        dataset = await self._source.fetch_items()
        if not dataset:
            logger.warning('No items found from source')
            return None

        # Filter already-scored items
        if deduplicate:
            dataset = self.filter_unscored_items(dataset)
            if not dataset:
                logger.info('All items already scored, nothing to process')
                return None

        # Apply sampling strategy
        dataset = self.sample_items(dataset)
        if not dataset:
            logger.info('No items to process after sampling')
            return None

        # Run evaluation
        if not self._metrics_config:
            raise ConfigurationError('metrics_config required')

        logger.info(
            f'Evaluating {len(dataset)} items with metrics: {list(self._metrics_config.keys())}'
        )

        # Use config default if not overridden at runtime
        should_trace = (
            trace_experiment if trace_experiment is not None else self._trace_experiment
        )

        if should_trace:
            configure_tracing('langfuse')
        else:
            configure_tracing('noop')

        # Build scoring_config (start with metrics, add model/weights if present)
        scoring_config: dict[str, Any] = {'metric': self._metrics_config}
        if 'model' in self._evaluation_config:
            scoring_config['model'] = self._evaluation_config['model']
        if 'weights' in self._evaluation_config:
            scoring_config['weights'] = self._evaluation_config['weights']

        # Build base kwargs with defaults
        eval_kwargs: dict[str, Any] = {
            'evaluation_inputs': dataset,
            'scoring_config': scoring_config,
            'evaluation_name': self.name,
            'scoring_strategy': 'flat',
            'trace_granularity': 'single',
            'run_id': f'{self.name}-{self._run_timestamp()}',
        }

        # Merge config-based settings (exclude model/weights as they're in scoring_config)
        for key, value in self._evaluation_config.items():
            if key not in ('model', 'weights'):
                eval_kwargs[key] = value

        logger.info(
            f'Evaluation config: max_concurrent={eval_kwargs.get("max_concurrent")}, '
            f'throttle_delay={eval_kwargs.get("throttle_delay")}'
        )

        results = evaluation_runner(**eval_kwargs)
        if results is None:
            logger.warning('Evaluation runner returned no results')
            return None

        if not should_trace:
            configure_tracing('langfuse')

        # Record scored items
        self.record_scored_items(dataset)

        if publish:
            pub_cfg = self._publishing_config

            # Publish as experiment if enabled
            publish_metric_names = config.get(
                'metric_names', cfg=pub_cfg
            ) or config.get('experiment.metrics', cfg=pub_cfg)
            if config.get('experiment.enabled', default=True, cfg=pub_cfg):
                results.publish_as_experiment(
                    dataset_name=dataset_name
                    or config.get('experiment.dataset_name', cfg=pub_cfg),
                    run_name=run_name or config.get('experiment.run_name', cfg=pub_cfg),
                    run_metadata=config.get('experiment.run_metadata', cfg=pub_cfg),
                    flush=config.get('experiment.flush', default=True, cfg=pub_cfg),
                    tags=config.get('experiment.tags', cfg=pub_cfg),
                    score_on_runtime_traces=config.get(
                        'experiment.score_on_runtime_traces',
                        default=False,
                        cfg=pub_cfg,
                    ),
                    link_to_traces=link_to_traces,
                    metric_names=metric_names or publish_metric_names,
                )
                logger.info('Published as experiment')
            elif config.get('push_to_langfuse', cfg=pub_cfg):
                try:
                    results.publish_to_observability(metric_names=publish_metric_names)
                except TypeError:
                    results.publish_to_observability()
            else:
                logger.info('No publishing configuration found')

            if config.get('push_to_db', cfg=pub_cfg):
                self._push_to_db(results)

        logger.info('Evaluation complete')
        return results
