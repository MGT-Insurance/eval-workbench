from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pytest

pytest.importorskip('psycopg')


class _FakeCursor:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[object, object | None]] = []
        self.executemany_calls: list[tuple[object, list[tuple[object, ...]]]] = []

    def execute(self, query, params=None) -> None:
        self.execute_calls.append((query, params))

    def executemany(self, query, params_seq) -> None:
        self.executemany_calls.append((query, list(params_seq)))

    def fetchall(self):
        return []

    def fetchone(self):
        return None

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

    with pytest.raises(ValueError, match='Unsafe SQL type definition'):
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


def test_upload_dataframe_batches_and_converts_nan_to_none(monkeypatch) -> None:
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

    em_calls = db.pool.conn.cursor_obj.executemany_calls
    assert len(em_calls) == 3  # 5 rows in chunks of 2 => 3 batches
    batch_sizes = [len(rows) for (_, rows) in em_calls]
    assert batch_sizes == [2, 2, 1]

    # Ensure NaN became None in the payload
    all_rows = [r for (_, rows) in em_calls for r in rows]
    # rows are tuples in column order: (id, v)
    assert (2, None) in all_rows
