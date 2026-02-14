from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd
import pytest

pytest.importorskip('psycopg')


class _FakeCursor:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[object, object | None]] = []
        self.copy_calls: list[tuple[object, list[tuple[object, ...]]]] = []
        self.rows: list[dict[str, object]] = []
        self._fetch_index = 0

    def execute(self, query, params=None) -> None:
        self.execute_calls.append((query, params))
        self._fetch_index = 0

    def executemany(self, query, params_seq) -> None:
        # Kept for backwards compatibility with older tests.
        self.copy_calls.append((query, list(params_seq)))

    def copy(self, query):
        return _FakeCopy(self, query)

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        if not self.rows:
            return None
        return self.rows[0]

    def fetchmany(self, size: int):
        start = self._fetch_index
        end = start + size
        batch = self.rows[start:end]
        self._fetch_index = min(end, len(self.rows))
        return batch

    def __enter__(self) -> '_FakeCursor':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.commits = 0

    def cursor(self, **kwargs) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1

    def __enter__(self) -> '_FakeConn':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeCopy:
    def __init__(self, cursor: _FakeCursor, query: object) -> None:
        self._cursor = cursor
        self._query = query
        self.rows: list[tuple[object, ...]] = []

    def write_row(self, row) -> None:
        self.rows.append(tuple(row))

    def __enter__(self) -> '_FakeCopy':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cursor.copy_calls.append((self._query, self.rows))
        return None


class _FakePool:
    def __init__(self, *, conninfo: str, min_size: int, max_size: int, kwargs: dict):
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self.kwargs = kwargs
        self.conn = _FakeConn()
        self.closed = False

    @contextmanager
    def connection(self):
        yield self.conn

    # Used by NeonConnection.get_connection()
    def getconn(self):
        return self.conn

    def putconn(self, conn) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeSettings:
    def __init__(
        self,
        *,
        database_url: str | None = 'postgresql://example',
        db_pool_min_size: int = 0,
        db_pool_max_size: int = 2,
        db_connect_timeout_seconds: int = 10,
        db_statement_timeout_ms: int = 60000,
        db_use_startup_statement_timeout: bool = False,
        db_application_name: str | None = 'test-app',
        db_upload_chunk_size: int = 2,
    ) -> None:
        self.database_url = database_url
        self.db_pool_min_size = db_pool_min_size
        self.db_pool_max_size = db_pool_max_size
        self.db_connect_timeout_seconds = db_connect_timeout_seconds
        self.db_statement_timeout_ms = db_statement_timeout_ms
        self.db_use_startup_statement_timeout = db_use_startup_statement_timeout
        self.db_application_name = db_application_name
        self.db_upload_chunk_size = db_upload_chunk_size


def test_neon_connection_requires_database_url(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(
        neon, 'get_neon_settings', lambda: _FakeSettings(database_url=None)
    )

    with pytest.raises(ValueError, match='DATABASE_URL not found'):
        neon.NeonConnection()


def test_neon_connection_validates_pool_sizes(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(
        neon,
        'get_neon_settings',
        lambda: _FakeSettings(db_pool_min_size=5, db_pool_max_size=1),
    )

    with pytest.raises(ValueError, match='DB_POOL_MIN_SIZE cannot exceed'):
        neon.NeonConnection(connection_string='postgresql://example')


def test_neon_connection_builds_pool_kwargs(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(
        database_url='postgresql://from-settings',
        db_pool_min_size=1,
        db_pool_max_size=3,
        db_connect_timeout_seconds=7,
        db_statement_timeout_ms=1234,
        db_use_startup_statement_timeout=True,  # triggers startup options
        db_application_name='my-app',
    )
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection()  # uses settings.database_url
    assert isinstance(db.pool, _FakePool)
    assert db.pool.conninfo == 'postgresql://from-settings'
    assert db.pool.min_size == 1
    assert db.pool.max_size == 3
    assert db.pool.kwargs['connect_timeout'] == 7
    assert db.pool.kwargs['application_name'] == 'my-app'
    # statement_timeout via startup options
    assert 'statement_timeout=1234' in db.pool.kwargs['options']


def test_statement_timeout_applied_per_connection_when_not_startup(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(
        db_statement_timeout_ms=5000, db_use_startup_statement_timeout=False
    )
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    # Trigger _connection() which applies statement timeout
    _ = db.fetch_one('SELECT 1')

    calls = db.pool.conn.cursor_obj.execute_calls
    assert any('statement_timeout' in str(q) for (q, _) in calls)


def test_create_table_columns_validates_sql_type(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: _FakeSettings())

    db = neon.NeonConnection(connection_string='postgresql://example')

    with pytest.raises(ValueError, match='Unsafe SQL fragment'):
        db.create_table(
            't',
            columns=[
                ('id', 'TEXT); DROP TABLE x;--'),
            ],
        )


def test_create_table_columns_executes_and_commits(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    # Disable statement-timeout execute call to simplify assertions
    settings = _FakeSettings(db_statement_timeout_ms=0)
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    db.create_table(
        'users',
        columns=[
            ('id', 'TEXT PRIMARY KEY'),
            ('name', 'TEXT'),
        ],
    )

    # At least one execute should have been made for CREATE TABLE
    calls = db.pool.conn.cursor_obj.execute_calls
    assert any('CREATE TABLE IF NOT EXISTS' in str(q) for (q, _) in calls)
    assert db.pool.conn.commits == 1


def test_fetch_all_timeout_override_precedence(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(
        db_statement_timeout_ms=5000,
        db_use_startup_statement_timeout=False,
    )
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example', timeout=2)
    db.fetch_all('SELECT 1', timeout_seconds=0)

    calls = db.pool.conn.cursor_obj.execute_calls
    assert any('statement_timeout' in str(q) and '0' in str(q) for (q, _) in calls), (
        'Expected per-call timeout override (0ms) to win over instance timeout.'
    )


def test_fetch_dataframe_timeout_override_applies(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(
        db_statement_timeout_ms=5000,
        db_use_startup_statement_timeout=False,
    )
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    _ = db.fetch_dataframe('SELECT 1', timeout_seconds=1.5)

    calls = db.pool.conn.cursor_obj.execute_calls
    assert any('statement_timeout' in str(q) and '1500' in str(q) for (q, _) in calls)


def test_fetch_chunks_yields_batched_rows(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(db_statement_timeout_ms=0)
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    db.pool.conn.cursor_obj.rows = [
        {'id': 1},
        {'id': 2},
        {'id': 3},
    ]

    chunks = list(db.fetch_chunks('SELECT id FROM t ORDER BY id', chunk_size=2))
    assert chunks == [
        [{'id': 1}, {'id': 2}],
        [{'id': 3}],
    ]


def test_fetch_dataframe_chunks_yields_dataframe_batches(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(db_statement_timeout_ms=0)
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    db.pool.conn.cursor_obj.rows = [
        {'id': 1, 'value': 'a'},
        {'id': 2, 'value': 'b'},
        {'id': 3, 'value': 'c'},
    ]

    dataframes = list(
        db.fetch_dataframe_chunks(
            'SELECT id, value FROM t ORDER BY id',
            chunk_size=2,
        )
    )
    assert len(dataframes) == 2
    assert list(dataframes[0]['id']) == [1, 2]
    assert list(dataframes[1]['id']) == [3]


def test_fetch_dataframe_chunked_returns_single_dataframe(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(db_statement_timeout_ms=0)
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    db.pool.conn.cursor_obj.rows = [
        {'id': 1, 'value': 'a'},
        {'id': 2, 'value': 'b'},
        {'id': 3, 'value': 'c'},
    ]

    df = db.fetch_dataframe_chunked(
        "SELECT * from athena_cases where created_at > '2026-01-01'",
        chunk_size=2,
        timeout_seconds=0,
    )
    assert list(df['id']) == [1, 2, 3]
    assert list(df['value']) == ['a', 'b', 'c']


def test_upload_dataframe_copy_writes_rows_and_converts_nan_to_none(
    monkeypatch,
) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(db_statement_timeout_ms=0, db_upload_chunk_size=2)
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    df = pd.DataFrame(
        [
            {'id': 1, 'v': 1.0},
            {'id': 2, 'v': float('nan')},
            {'id': 3, 'v': 3.0},
            {'id': 4, 'v': None},
            {'id': 5, 'v': 5.0},
        ]
    )
    db.upload_dataframe(df, 't')

    copy_calls = db.pool.conn.cursor_obj.copy_calls
    assert len(copy_calls) == 1

    # Ensure NaN became None in the payload
    all_rows = copy_calls[0][1]
    # rows are tuples in column order: (id, v)
    assert (2, None) in all_rows
    assert (4, None) in all_rows


def test_upload_dataframe_serializes_dict_list_tuple_values(monkeypatch) -> None:
    from eval_workbench.shared.database import neon

    settings = _FakeSettings(db_statement_timeout_ms=0, db_upload_chunk_size=2)
    monkeypatch.setattr(neon, 'ConnectionPool', _FakePool)
    monkeypatch.setattr(neon, 'get_neon_settings', lambda: settings)

    db = neon.NeonConnection(connection_string='postgresql://example')
    df = pd.DataFrame(
        [
            {'id': 1, 'meta': {'a': 1}, 'tags': ['x', 'y'], 'coords': (1, 2)},
        ]
    )
    db.upload_dataframe(df, 't')

    copy_calls = db.pool.conn.cursor_obj.copy_calls
    assert len(copy_calls) == 1
    all_rows = copy_calls[0][1]
    written = all_rows[0]
    assert written[0] == 1
    assert written[1] == json.dumps({'a': 1})
    assert written[2] == json.dumps(['x', 'y'])
    assert written[3] == json.dumps((1, 2))
