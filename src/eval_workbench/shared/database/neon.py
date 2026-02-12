import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pydantic import Field

from eval_workbench.shared.settings import RepoSettingsBase, build_settings_config

logger = logging.getLogger(__name__)


class NeonSettings(RepoSettingsBase):
    model_config = build_settings_config(from_path=Path(__file__))

    database_url: str | None = Field(
        default=None,
        description='Postgres connection URL.',
    )
    db_pool_min_size: int = Field(
        default=0,
        description='Minimum number of connections in the pool.',
    )
    db_pool_max_size: int = Field(
        default=20,
        description='Maximum number of connections in the pool.',
    )
    db_connect_timeout_seconds: int = Field(
        default=10,
        description='Connection timeout in seconds.',
    )
    db_statement_timeout_ms: int = Field(
        default=60000,
        description='Statement timeout in milliseconds. 0 disables timeout.',
    )
    db_use_startup_statement_timeout: bool = Field(
        default=False,
        description='Set statement_timeout via startup options (unsupported by Neon pooler).',
    )
    db_application_name: str | None = Field(
        default=None,
        description='Optional Postgres application_name.',
    )


@lru_cache(maxsize=1)
def get_neon_settings() -> NeonSettings:
    return NeonSettings()


def reset_neon_settings_cache() -> None:
    get_neon_settings.cache_clear()


def _build_connection_kwargs(settings: NeonSettings) -> dict:
    kwargs: dict = {
        'row_factory': dict_row,
        'connect_timeout': settings.db_connect_timeout_seconds,
    }
    if settings.db_application_name:
        kwargs['application_name'] = settings.db_application_name
    if (
        settings.db_statement_timeout_ms > 0
        and settings.db_use_startup_statement_timeout
    ):
        kwargs['options'] = f'-c statement_timeout={settings.db_statement_timeout_ms}'
    return kwargs


def _apply_statement_timeout_sync(
    conn: psycopg.Connection, settings: NeonSettings
) -> None:
    if (
        settings.db_statement_timeout_ms > 0
        and not settings.db_use_startup_statement_timeout
    ):
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL('SET statement_timeout = {}').format(
                    sql.Literal(settings.db_statement_timeout_ms)
                )
            )


def _set_statement_timeout_sync(
    conn: psycopg.Connection, statement_timeout_ms: int
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL('SET statement_timeout = {}').format(
                sql.Literal(statement_timeout_ms)
            )
        )


_DISALLOWED_SQL_CONTROL_TOKENS = (';', '--', '/*', '*/')


def _validate_trusted_sql_fragment(fragment: str, field_name: str) -> str:
    if any(token in fragment for token in _DISALLOWED_SQL_CONTROL_TOKENS):
        raise ValueError(
            f'Unsafe SQL fragment in {field_name!r}: control tokens are not allowed.'
        )
    return fragment


async def _apply_statement_timeout_async(
    conn: psycopg.AsyncConnection, settings: NeonSettings
) -> None:
    if (
        settings.db_statement_timeout_ms > 0
        and not settings.db_use_startup_statement_timeout
    ):
        async with conn.cursor() as cur:
            await cur.execute(
                sql.SQL('SET statement_timeout = {}').format(
                    sql.Literal(settings.db_statement_timeout_ms)
                )
            )


async def _set_statement_timeout_async(
    conn: psycopg.AsyncConnection, statement_timeout_ms: int
) -> None:
    async with conn.cursor() as cur:
        await cur.execute(
            sql.SQL('SET statement_timeout = {}').format(
                sql.Literal(statement_timeout_ms)
            )
        )


class NeonConnection:
    """
    Synchronous database manager for Neon (PostgreSQL).
    Best for scripts, data analysis, or standard web apps (Flask/Django).
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the connection pool.
        Args:
            connection_string: Postgres connection URL. If None, fetches DATABASE_URL from env.
            timeout: Optional default statement timeout for all
                operations from this connection instance. Uses seconds. 0 disables timeout.
        """
        self._settings = get_neon_settings()
        self.dsn = connection_string or self._settings.database_url
        if not self.dsn:
            raise ValueError(
                'DATABASE_URL not found in environment variables or arguments.'
            )
        if self._settings.db_pool_min_size > self._settings.db_pool_max_size:
            raise ValueError('DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE.')
        if timeout is not None and timeout < 0:
            raise ValueError('timeout must be >= 0.')
        self.timeout = timeout

        # Initialize Connection Pool
        self.pool = ConnectionPool(
            conninfo=self.dsn,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            kwargs=_build_connection_kwargs(self._settings),
        )
        logger.info('Sync Database connection pool initialized.')

    def close(self):
        """Close the connection pool gracefully."""
        if self.pool:
            self.pool.close()
            logger.info('Sync Database connection pool closed.')

    def __enter__(self) -> 'NeonConnection':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            if self.timeout is not None:
                _set_statement_timeout_sync(conn, int(self.timeout * 1000))
            else:
                _apply_statement_timeout_sync(conn, self._settings)
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def _connection(self):
        with self.pool.connection() as conn:
            if self.timeout is not None:
                _set_statement_timeout_sync(conn, int(self.timeout * 1000))
            else:
                _apply_statement_timeout_sync(conn, self._settings)
            yield conn

    def fetch_all(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
        except psycopg.Error as e:
            logger.error(f'Database query failed: {e}')
            raise

    def fetch_one(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchone()
        except psycopg.Error as e:
            logger.error(f'Database fetch_one failed: {e}')
            raise

    def execute_commit(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> None:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
                logger.info('Transaction committed successfully.')
        except psycopg.Error as e:
            logger.error(f'Database execution failed: {e}')
            raise

    def fetch_dataframe(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> pd.DataFrame:
        try:
            data = self.fetch_all(query, params)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f'Failed to fetch DataFrame: {e}')
            raise

    def create_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        columns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Create a table safely.

        Provide either a raw schema string or a list of columns as (name, type)
        tuples.

        Safety note:
            `schema` and `col_type` values are treated as SQL fragments and must be
            trusted, developer-authored input. A light guard blocks obvious SQL control
            tokens but this method is not intended for untrusted end-user input.
        """
        if (schema is None) == (columns is None):
            raise ValueError('Provide exactly one of schema or columns.')

        try:
            if columns is not None:
                column_defs = [
                    sql.SQL('{} {}').format(
                        sql.Identifier(name),
                        sql.SQL(_validate_trusted_sql_fragment(col_type, 'col_type')),
                    )
                    for name, col_type in columns
                ]
                schema_sql = sql.SQL(', ').join(column_defs)
            else:
                assert schema is not None
                schema_sql = sql.SQL(_validate_trusted_sql_fragment(schema, 'schema'))

            query = sql.SQL('CREATE TABLE IF NOT EXISTS {} ({})').format(
                sql.Identifier(table_name),
                schema_sql,
            )
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                conn.commit()
            logger.info(f"Table '{table_name}' checked/created successfully.")
        except psycopg.Error as e:
            logger.error(f"Failed to create table '{table_name}': {e}")
            raise

    def upload_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        if df.empty:
            logger.warning('DataFrame is empty, skipping upload.')
            return
        try:
            # Ensure missing values become actual Python `None` (SQL NULL),
            # even for numeric columns where pandas would otherwise keep NaN.
            df_clean = df.astype(object).where(pd.notna(df), None)
            columns = list(df_clean.columns)
            copy_query = sql.SQL('COPY {} ({}) FROM STDIN').format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, columns)),
            )
            with self._connection() as conn:
                with conn.cursor() as cur:
                    with cur.copy(copy_query) as copy:
                        for row in df_clean.itertuples(index=False, name=None):
                            copy.write_row(row)
                conn.commit()
            logger.info(
                f"Successfully uploaded {len(df_clean)} rows to '{table_name}'."
            )
        except psycopg.Error as e:
            logger.error(f"Failed to upload DataFrame to '{table_name}': {e}")
            raise

    def check_health(self) -> bool:
        try:
            self.fetch_one('SELECT 1')
            return True
        except Exception:
            return False


class AsyncNeonConnection:
    """
    Async database manager for Neon (PostgreSQL).
    Best for AI Agents, FastAPI, Sanic, or high-concurrency workloads.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the async connection pool.

        Args:
            connection_string: Postgres connection URL. If None, fetches DATABASE_URL from env.
            timeout: Optional default statement timeout for all
                operations from this connection instance. Uses seconds. 0 disables timeout.
        """
        self._settings = get_neon_settings()
        self.dsn = connection_string or self._settings.database_url
        if not self.dsn:
            raise ValueError('DATABASE_URL not found.')
        if self._settings.db_pool_min_size > self._settings.db_pool_max_size:
            raise ValueError('DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE.')
        if timeout is not None and timeout < 0:
            raise ValueError('timeout must be >= 0.')
        self.timeout = timeout
        self._pool_opened = False

        # Initialize Async Connection Pool
        self.pool = AsyncConnectionPool(
            conninfo=self.dsn,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            kwargs=_build_connection_kwargs(self._settings),
            open=False,
        )
        logger.info('Async Database connection pool initialized.')

    async def close(self):
        """Close the async pool."""
        if self.pool:
            await self.pool.close()
            self._pool_opened = False
            logger.info('Async Database connection pool closed.')

    async def __aenter__(self) -> 'AsyncNeonConnection':
        if self.pool and not self._pool_opened:
            await self.pool.open()
            self._pool_opened = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @asynccontextmanager
    async def _connection(self):
        if self.pool and not self._pool_opened:
            await self.pool.open()
            self._pool_opened = True
        async with self.pool.connection() as conn:
            if self.timeout is not None:
                await _set_statement_timeout_async(conn, int(self.timeout * 1000))
            else:
                await _apply_statement_timeout_async(conn, self._settings)
            yield conn

    async def fetch_all(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Async fetch all."""
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    return await cur.fetchall()
        except psycopg.Error as e:
            logger.error(f'Async DB query failed: {e}')
            raise

    async def fetch_one(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async fetch one."""
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    return await cur.fetchone()
        except psycopg.Error as e:
            logger.error(f'Async DB fetch_one failed: {e}')
            raise

    async def execute_commit(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> None:
        """Async execute and commit."""
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
        except psycopg.Error as e:
            logger.error(f'Async DB execution failed: {e}')
            raise

    async def fetch_dataframe(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> pd.DataFrame:
        """Async execute query and return pandas DataFrame."""
        try:
            data = await self.fetch_all(query, params)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, pd.DataFrame, data)
        except Exception as e:
            logger.error(f'Failed to fetch DataFrame asynchronously: {e}')
            raise

    async def create_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        columns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Async create table safely.

        Safety note:
            `schema` and `col_type` values are treated as SQL fragments and must be
            trusted, developer-authored input. A light guard blocks obvious SQL control
            tokens but this method is not intended for untrusted end-user input.
        """
        if (schema is None) == (columns is None):
            raise ValueError('Provide exactly one of schema or columns.')

        try:
            if columns is not None:
                column_defs = [
                    sql.SQL('{} {}').format(
                        sql.Identifier(name),
                        sql.SQL(_validate_trusted_sql_fragment(col_type, 'col_type')),
                    )
                    for name, col_type in columns
                ]
                schema_sql = sql.SQL(', ').join(column_defs)
            else:
                assert schema is not None
                schema_sql = sql.SQL(_validate_trusted_sql_fragment(schema, 'schema'))

            query = sql.SQL('CREATE TABLE IF NOT EXISTS {} ({})').format(
                sql.Identifier(table_name),
                schema_sql,
            )
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                await conn.commit()
            logger.info(f"Table '{table_name}' checked/created successfully (Async).")
        except psycopg.Error as e:
            logger.error(f"Failed to create table '{table_name}': {e}")
            raise

    async def upload_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Async upload pandas DataFrame."""
        if df.empty:
            logger.warning('DataFrame is empty, skipping upload.')
            return

        try:
            df_clean = df.astype(object).where(pd.notna(df), None)
            columns = list(df_clean.columns)
            copy_query = sql.SQL('COPY {} ({}) FROM STDIN').format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, columns)),
            )

            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    async with cur.copy(copy_query) as copy:
                        for row in df_clean.itertuples(index=False, name=None):
                            await copy.write_row(row)
                await conn.commit()

            logger.info(
                f"Successfully uploaded {len(df_clean)} rows to '{table_name}' (Async)."
            )
        except psycopg.Error as e:
            logger.error(f"Failed to upload DataFrame to '{table_name}': {e}")
            raise

    async def check_health(self) -> bool:
        try:
            await self.fetch_one('SELECT 1')
            return True
        except Exception:
            return False
