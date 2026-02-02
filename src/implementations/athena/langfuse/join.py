from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
from axion.tracing import LangfuseTraceLoader 

from implementations.athena.langfuse.prompt_patterns import WorkflowPromptPatterns
from shared.database.neon import NeonConnection
from shared.langfuse.trace import PromptPatternsBase, TraceCollection


@dataclass(frozen=True)
class JoinSettings:
    case_table: str = 'athena_cases'
    case_columns: tuple[str, ...] = (
        'id',
        'workflow_id',
        'quote_locator',
        'slack_thread_ts',
        'slack_channel_id',
        'langfuse_trace_id',
    )
    trace_name: str = 'athena'
    trace_tags: tuple[str, ...] = ('production',)


class AthenaNeonLangfuseJoiner:
    """
    Helper to join Neon case rows with Langfuse trace data.

    Usage:
        joiner = AthenaNeonLangfuseJoiner(neon_db, langfuse_loader)
        cases = joiner.fetch_cases(limit=50)
        traces = joiner.fetch_traces(limit=200)
        joined = joiner.join_cases_with_traces(cases, traces)
    """

    def __init__(
        self,
        neon_db: NeonConnection,
        trace_loader: LangfuseTraceLoader,
        *,
        settings: JoinSettings | None = None,
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    ) -> None:
        self._db = neon_db
        self._loader = trace_loader
        self._settings = settings or JoinSettings()
        self._prompt_patterns = prompt_patterns or WorkflowPromptPatterns

    def fetch_cases(
        self,
        *,
        limit: Optional[int] = None,
        where: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        cols = ', '.join(columns or self._settings.case_columns)
        query = f'SELECT {cols} FROM {self._settings.case_table}'
        if where:
            query = f'{query} WHERE {where}'
        if limit is not None:
            query = f'{query} LIMIT {int(limit)}'
        return self._db.fetch_dataframe(query)

    def fetch_traces(
        self,
        *,
        limit: int = 200,
        name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        fetch_full_traces: bool = True,
        show_progress: bool = False,
    ) -> TraceCollection:
        trace_data_list = self._loader.fetch_traces(
            limit=limit,
            name=name or self._settings.trace_name,
            fetch_full_traces=fetch_full_traces,
            tags=list(tags or self._settings.trace_tags),
            show_progress=show_progress,
        )
        return TraceCollection(trace_data_list, prompt_patterns=self._prompt_patterns)

    def fetch_traces_by_ids(
        self,
        trace_ids: Sequence[str],
        *,
        fetch_full_traces: bool = True,
        show_progress: bool = False,
        trace_fetcher: Callable[[str], object] | None = None,
    ) -> TraceCollection:
        ids = [str(trace_id) for trace_id in trace_ids if trace_id]
        if not ids:
            return TraceCollection([], prompt_patterns=self._prompt_patterns)

        if trace_fetcher is not None:
            trace_data_list = [trace_fetcher(trace_id) for trace_id in ids]
            return TraceCollection(
                trace_data_list, prompt_patterns=self._prompt_patterns
            )

        loader = self._loader
        fetch_by_id = (
            getattr(loader, 'fetch_trace_by_id', None)
            or getattr(loader, 'fetch_trace', None)
            or getattr(loader, 'get_trace', None)
        )
        if not callable(fetch_by_id):
            raise ValueError(
                'LangfuseTraceLoader does not expose a trace-id fetch method. '
                'Pass trace_fetcher=callable to fetch traces by id.'
            )

        trace_data_list = []
        for trace_id in ids:
            try:
                trace_data = fetch_by_id(trace_id, fetch_full_traces=fetch_full_traces)
            except TypeError:
                trace_data = fetch_by_id(trace_id)
            trace_data_list.append(trace_data)

        if show_progress and len(ids) > 0:
            pass

        return TraceCollection(trace_data_list, prompt_patterns=self._prompt_patterns)

    def join_cases_with_traces(
        self,
        cases: pd.DataFrame,
        traces: TraceCollection,
        *,
        trace_id_column: str = 'langfuse_trace_id',
        trace_output_column: str = 'langfuse_trace',
    ) -> pd.DataFrame:
        if trace_id_column not in cases.columns:
            raise KeyError(f"Missing '{trace_id_column}' in cases dataframe.")

        traces_by_id = self._index_traces_by_id(traces)
        joined = cases.copy()
        joined[trace_output_column] = joined[trace_id_column].map(traces_by_id.get)
        return joined

    def _index_traces_by_id(self, traces: Iterable) -> dict[str, object]:
        indexed: dict[str, object] = {}
        for trace in traces:
            try:
                trace_id = getattr(trace, 'id', None)
            except Exception:
                trace_id = None
            if trace_id:
                indexed[str(trace_id)] = trace
        return indexed
