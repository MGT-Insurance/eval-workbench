from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest


def test_with_time_bounds_appends_where_or_and() -> None:
    from eval_workbench.shared.monitoring.sources import SlackNeonJoinDataSource

    q1, p1 = SlackNeonJoinDataSource._with_time_bounds(
        'SELECT * FROM athena_cases',
        'created_at',
        oldest_ts=100.0,
        latest_ts=200.0,
    )
    assert (
        'WHERE created_at >= to_timestamp(%s) AND created_at <= to_timestamp(%s)' in q1
    )
    assert p1 == (100.0, 200.0)

    q2, p2 = SlackNeonJoinDataSource._with_time_bounds(
        'SELECT * FROM athena_cases WHERE status = %s',
        'created_at',
        oldest_ts=50.0,
        latest_ts=None,
    )
    assert 'WHERE status = %s AND created_at >= to_timestamp(%s)' in q2
    assert p2 == (50.0,)


def test_init_validates_join_column_lengths() -> None:
    from eval_workbench.shared.monitoring.sources import SlackNeonJoinDataSource

    with pytest.raises(ValueError, match='must have equal lengths'):
        SlackNeonJoinDataSource(
            name='x',
            channel_ids=['C1'],
            neon_query='SELECT 1',
            slack_join_columns=['a', 'b'],
            neon_join_columns=['a'],
        )


def test_fetch_items_merges_and_sets_dataset_id(monkeypatch) -> None:
    from eval_workbench.shared.monitoring import sources

    class _FakeExporter:
        default_oldest_ts = 1000.0
        default_latest_ts = 2000.0

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def execute(self):
            return ['item']

    class _FakeDatasetObj:
        def __init__(self, df: pd.DataFrame):
            self._df = df

        def to_dataframe(self) -> pd.DataFrame:
            return self._df.copy()

    class _FakeDataset:
        slack_df = pd.DataFrame(
            {
                'dataset_id': ['orig-1'],
                'dataset_metadata': [
                    json.dumps({'channel_id': 'C09', 'thread_ts': '171234.000100'})
                ],
            }
        )
        read_dataframe_input: pd.DataFrame | None = None

        @staticmethod
        def create(name: str, items):
            return _FakeDatasetObj(_FakeDataset.slack_df)

        @staticmethod
        def read_dataframe(df: pd.DataFrame, ignore_extra_keys: bool = False):
            _FakeDataset.read_dataframe_input = df.copy()
            return df

    class _FakeAsyncNeonConnection:
        last_query: str | None = None
        last_params: tuple[float, ...] | None = None

        def __init__(self, connection_string=None) -> None:
            self.connection_string = connection_string

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def fetch_dataframe_chunked(
            self, query, params=None, *, chunk_size=1000, timeout_seconds=None
        ):
            _FakeAsyncNeonConnection.last_query = query
            _FakeAsyncNeonConnection.last_params = params
            return pd.DataFrame(
                {
                    'slack_channel_id': ['C09'],
                    'slack_thread_ts': ['171234.000100'],
                    'quote_locator': ['MGT-BOP-2007582'],
                    'created_at': ['2026-02-01T00:00:00Z'],
                }
            )

    monkeypatch.setattr(sources, 'Dataset', _FakeDataset)
    monkeypatch.setitem(
        __import__('sys').modules,
        'eval_workbench.shared.slack.exporter',
        type('m', (), {'SlackExporter': _FakeExporter}),
    )
    monkeypatch.setitem(
        __import__('sys').modules,
        'eval_workbench.shared.database.neon',
        type('m', (), {'AsyncNeonConnection': _FakeAsyncNeonConnection}),
    )

    source = sources.SlackNeonJoinDataSource(
        name='athena',
        channel_ids=['C09'],
        neon_query='SELECT slack_thread_ts, slack_channel_id, quote_locator, created_at FROM athena_cases',
        slack_join_columns=['channel_id', 'thread_ts'],
        neon_join_columns=['slack_channel_id', 'slack_thread_ts'],
        dataset_id_column='quote_locator',
        window_minutes=300,
        buffer_minutes=5,
    )
    result = asyncio.run(source.fetch_items())

    assert isinstance(result, pd.DataFrame)
    assert _FakeAsyncNeonConnection.last_params == (700.0, 2300.0)
    assert _FakeAsyncNeonConnection.last_query is not None
    assert 'created_at >= to_timestamp(%s)' in _FakeAsyncNeonConnection.last_query
    assert 'created_at <= to_timestamp(%s)' in _FakeAsyncNeonConnection.last_query

    merged = _FakeDataset.read_dataframe_input
    assert merged is not None
    assert merged['dataset_id'].tolist() == ['MGT-BOP-2007582']
    assert isinstance(merged.loc[0, 'dataset_metadata'], str)


def test_fetch_items_raises_on_missing_join_columns(monkeypatch) -> None:
    from eval_workbench.shared.monitoring import sources

    class _FakeExporter:
        default_oldest_ts = None
        default_latest_ts = None

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def execute(self):
            return ['item']

    class _FakeDatasetObj:
        def to_dataframe(self) -> pd.DataFrame:
            return pd.DataFrame({'dataset_metadata': [json.dumps({'thread_ts': '1'})]})

    class _FakeDataset:
        @staticmethod
        def create(name: str, items):
            return _FakeDatasetObj()

    monkeypatch.setattr(sources, 'Dataset', _FakeDataset)
    monkeypatch.setitem(
        __import__('sys').modules,
        'eval_workbench.shared.slack.exporter',
        type('m', (), {'SlackExporter': _FakeExporter}),
    )

    source = sources.SlackNeonJoinDataSource(
        name='athena',
        channel_ids=['C09'],
        neon_query='SELECT 1',
        slack_join_columns=['channel_id', 'thread_ts'],
        neon_join_columns=['slack_channel_id', 'slack_thread_ts'],
    )

    with pytest.raises(ValueError, match='Missing slack join columns'):
        asyncio.run(source.fetch_items())
