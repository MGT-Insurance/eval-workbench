from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pytest
from psycopg.types.json import Jsonb

from shared.database.evaluation_upload import (
    EVALUATION_DATASET_COLUMNS,
    EVALUATION_RESULTS_COLUMNS,
    EvaluationUploader,
    EvaluationUploadError,
    subset_evaluation_dataset_df_for_upload,
    subset_evaluation_results_df_for_upload,
    upload_evaluation_results_df,
)


class _FakeCursor:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple[object, list[tuple[object, ...]]]] = []

    def executemany(self, query, rows) -> None:
        # Store a concrete list copy for assertions
        self.executemany_calls.append((query, list(rows)))

    def __enter__(self) -> '_FakeCursor':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1

    def __enter__(self) -> '_FakeConn':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeNeonConnection:
    def __init__(self) -> None:
        self.conn = _FakeConn()

    @contextmanager
    def _connection(self):
        yield self.conn


def test_subset_dataset_renames_metadata_to_data_metadata() -> None:
    df = pd.DataFrame(
        [
            {
                'id': 'x',
                'query': 'q',
                'metadata': '{"hello": "world"}',  # should become data_metadata
                'extra': 123,  # should be dropped
            }
        ]
    )
    out = subset_evaluation_dataset_df_for_upload(df)
    assert 'metadata' not in out.columns
    assert 'data_metadata' in out.columns
    assert 'extra' not in out.columns
    assert out.loc[0, 'data_metadata'] == '{"hello": "world"}'


def test_subset_dataset_include_missing_adds_all_columns() -> None:
    df = pd.DataFrame([{'id': 'x'}])
    out = subset_evaluation_dataset_df_for_upload(df, include_missing_columns=True)
    assert list(out.columns) == EVALUATION_DATASET_COLUMNS


def test_subset_results_derives_dataset_id_from_id() -> None:
    df = pd.DataFrame(
        [
            {
                'run_id': 'r1',
                'id': 'dataset_1',
                'metric_name': 'm',
                'metric_score': 1.0,
            }
        ]
    )
    out = subset_evaluation_results_df_for_upload(df, dataset_id_source='id')
    assert 'dataset_id' in out.columns
    assert out.loc[0, 'dataset_id'] == 'dataset_1'


def test_subset_results_include_missing_adds_all_columns() -> None:
    df = pd.DataFrame(
        [
            {
                'run_id': 'r1',
                'id': 'dataset_1',
                'metric_name': 'm',
            }
        ]
    )
    out = subset_evaluation_results_df_for_upload(
        df, include_missing_columns=True, dataset_id_source='id'
    )
    assert list(out.columns) == EVALUATION_RESULTS_COLUMNS


def test_uploader_requires_db_for_upload() -> None:
    uploader = EvaluationUploader(db=None)
    with pytest.raises(ValueError, match='No db provided'):
        uploader.upload_results(
            pd.DataFrame([{'run_id': 'r', 'id': 'x', 'metric_name': 'm'}])
        )


def test_upload_results_sanitizes_nan_in_signals_and_wraps_jsonb() -> None:
    fake_db = _FakeNeonConnection()
    df = pd.DataFrame(
        [
            {
                'run_id': 'r1',
                'id': 'dataset_1',
                'metric_name': 'm',
                'signals': {'items': [{'score': float('nan'), 'ok': True}]},
            }
        ]
    )

    uploaded = upload_evaluation_results_df(
        fake_db, df, dataset_id_source='id', on_conflict='do_nothing', chunk_size=1000
    )
    assert 'dataset_id' in uploaded.columns

    # Verify we actually executed an insert and committed
    calls = fake_db.conn.cursor_obj.executemany_calls
    assert len(calls) == 1
    _, rows = calls[0]
    assert len(rows) == 1
    row = rows[0]

    # signals is the 10th column in evaluation_results insert order
    # We can't rely on table order, but we can rely on the dataframe columns order used.
    signals_idx = list(uploaded.columns).index('signals')
    assert isinstance(row[signals_idx], Jsonb)
    assert row[signals_idx].obj == {'items': [{'score': None, 'ok': True}]}
    assert fake_db.conn.commits == 1


def test_upload_results_raises_descriptive_error_for_unserializable_jsonb() -> None:
    fake_db = _FakeNeonConnection()

    class _NotJsonSerializable:
        pass

    df = pd.DataFrame(
        [
            {
                'run_id': 'r1',
                'id': 'dataset_1',
                'metric_name': 'm',
                'signals': {'bad': _NotJsonSerializable()},
            }
        ]
    )

    with pytest.raises(
        EvaluationUploadError,
        match=r'Invalid JSONB value for evaluation_results\.signals at row 0',
    ):
        upload_evaluation_results_df(fake_db, df, dataset_id_source='id')


def test_upload_results_supports_upsert_mode() -> None:
    fake_db = _FakeNeonConnection()
    df = pd.DataFrame(
        [
            {
                'run_id': 'r1',
                'id': 'dataset_1',
                'metric_name': 'm',
                'metric_score': 0.1,
                'signals': {'ok': True},
            }
        ]
    )
    uploaded = upload_evaluation_results_df(
        fake_db, df, dataset_id_source='id', on_conflict='upsert'
    )
    calls = fake_db.conn.cursor_obj.executemany_calls
    assert len(calls) == 1
    query, _rows = calls[0]
    # psycopg.sql.Composed stringifies to a structured representation; check it includes upsert tokens.
    q_str = str(query)
    assert 'ON CONFLICT' in q_str
    assert 'DO UPDATE' in q_str
    # Ensure dataset_id was derived as usual
    assert uploaded.loc[0, 'dataset_id'] == 'dataset_1'
