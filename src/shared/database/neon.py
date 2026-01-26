import asyncio
import functools
import logging
import re
from concurrent.futures import Executor
from contextlib import contextmanager, asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, TypeVar, Iterable
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, AsyncConnectionPool
import pandas as pd
from pydantic import Field

from shared.settings import RepoSettingsBase, build_settings_config

# Configure logging only if the application hasn't configured it yet.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

T = TypeVar("T")


class NeonSettings(RepoSettingsBase):
    model_config = build_settings_config(from_path=Path(__file__))

    database_url: str | None = Field(
        default=None,
        description="Postgres connection URL.",
    )
    db_pool_min_size: int = Field(
        default=0,
        description="Minimum number of connections in the pool.",
    )
    db_pool_max_size: int = Field(
        default=20,
        description="Maximum number of connections in the pool.",
    )
    db_connect_timeout_seconds: int = Field(
        default=10,
        description="Connection timeout in seconds.",
    )
    db_statement_timeout_ms: int = Field(
        default=60000,
        description="Statement timeout in milliseconds. 0 disables timeout.",
    )
    db_use_startup_statement_timeout: bool = Field(
        default=False,
        description="Set statement_timeout via startup options (unsupported by Neon pooler).",
    )
    db_application_name: str | None = Field(
        default=None,
        description="Optional Postgres application_name.",
    )
    db_upload_chunk_size: int = Field(
        default=1000,
        description="Rows per batch for upload_dataframe.",
    )


@lru_cache(maxsize=1)
def get_neon_settings() -> NeonSettings:
    return NeonSettings()


def reset_neon_settings_cache() -> None:
    get_neon_settings.cache_clear()


def _build_connection_kwargs(settings: NeonSettings) -> dict:
    kwargs: dict = {
        "row_factory": dict_row,
        "connect_timeout": settings.db_connect_timeout_seconds,
    }
    if settings.db_application_name:
        kwargs["application_name"] = settings.db_application_name
    if settings.db_statement_timeout_ms > 0 and settings.db_use_startup_statement_timeout:
        kwargs["options"] = f"-c statement_timeout={settings.db_statement_timeout_ms}"
    return kwargs


def _chunked(iterable: Iterable[Tuple[Any, ...]], size: int) -> Iterable[List[Tuple[Any, ...]]]:
    batch: List[Tuple[Any, ...]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


_SQL_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9_\s(),\.]+$")


def _validate_sql_type(sql_type: str) -> str:
    if not _SQL_TYPE_PATTERN.fullmatch(sql_type.strip()):
        raise ValueError(f"Unsafe SQL type definition: {sql_type!r}")
    return sql_type


def _apply_statement_timeout_sync(conn: psycopg.Connection, settings: NeonSettings) -> None:
    if settings.db_statement_timeout_ms > 0 and not settings.db_use_startup_statement_timeout:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET statement_timeout = {}").format(
                    sql.Literal(settings.db_statement_timeout_ms)
                )
            )


async def _apply_statement_timeout_async(conn: psycopg.AsyncConnection, settings: NeonSettings) -> None:
    if settings.db_statement_timeout_ms > 0 and not settings.db_use_startup_statement_timeout:
        async with conn.cursor() as cur:
            await cur.execute(
                sql.SQL("SET statement_timeout = {}").format(
                    sql.Literal(settings.db_statement_timeout_ms)
                )
            )


class NeonDatabaseManager:
    """
    Synchronous database manager for Neon (PostgreSQL).
    Best for scripts, data analysis, or standard web apps (Flask/Django).
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the connection pool.

        :param connection_string: Postgres connection URL. If None, fetches DATABASE_URL from env.
        """
        self._settings = get_neon_settings()
        self.dsn = connection_string or self._settings.database_url
        if not self.dsn:
            raise ValueError("DATABASE_URL not found in environment variables or arguments.")
        if self._settings.db_pool_min_size > self._settings.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE.")

        # Initialize Connection Pool
        self.pool = ConnectionPool(
            conninfo=self.dsn,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            kwargs=_build_connection_kwargs(self._settings),
        )
        logger.info("Sync Database connection pool initialized.")

    def close(self):
        """Close the connection pool gracefully."""
        if self.pool:
            self.pool.close()
            logger.info("Sync Database connection pool closed.")

    def __enter__(self) -> "NeonDatabaseManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            _apply_statement_timeout_sync(conn, self._settings)
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def _connection(self):
        with self.pool.connection() as conn:
            _apply_statement_timeout_sync(conn, self._settings)
            yield conn

    def fetch_all(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[Dict[str, Any]]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
        except psycopg.Error as e:
            logger.error(f"Database query failed: {e}")
            raise

    def fetch_one(self, query: str, params: Optional[Union[tuple, dict]] = None) -> Optional[Dict[str, Any]]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchone()
        except psycopg.Error as e:
            logger.error(f"Database fetch_one failed: {e}")
            raise

    def execute_commit(self, query: str, params: Optional[Union[tuple, dict]] = None) -> None:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
                logger.info("Transaction committed successfully.")
        except psycopg.Error as e:
            logger.error(f"Database execution failed: {e}")
            raise

    def fetch_dataframe(self, query: str, params: Optional[Union[tuple, dict]] = None) -> pd.DataFrame:
        try:
            data = self.fetch_all(query, params)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to fetch DataFrame: {e}")
            raise

    def create_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        columns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Create a table safely.

        Provide either a raw schema string (trusted only) or a list of columns
        as (name, type) tuples. Column types are validated against a safe pattern.
        """
        if (schema is None) == (columns is None):
            raise ValueError("Provide exactly one of schema or columns.")

        try:
            if columns is not None:
                column_defs = [
                    sql.SQL("{} {}").format(
                        sql.Identifier(name),
                        sql.SQL(_validate_sql_type(col_type)),
                    )
                    for name, col_type in columns
                ]
                schema_sql = sql.SQL(", ").join(column_defs)
            else:
                schema_sql = sql.SQL(schema)

            query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
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
            logger.warning("DataFrame is empty, skipping upload.")
            return
        try:
            settings = get_neon_settings()
            df_clean = df.where(pd.notnull(df), None)
            columns = list(df_clean.columns)
            query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.SQL(', ').join(sql.Placeholder() * len(columns))
            )
            with self._connection() as conn:
                with conn.cursor() as cur:
                    rows_iter = df_clean.itertuples(index=False, name=None)
                    for batch in _chunked(rows_iter, settings.db_upload_chunk_size):
                        cur.executemany(query, batch)
                conn.commit()
            logger.info(f"Successfully uploaded {len(df_clean)} rows to '{table_name}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to upload DataFrame to '{table_name}': {e}")
            raise

    def check_health(self) -> bool:
        try:
            self.fetch_one("SELECT 1")
            return True
        except Exception:
            return False


class AsyncNeonDatabaseManager:
    """
    Async database manager for Neon (PostgreSQL).
    Best for AI Agents, FastAPI, Sanic, or high-concurrency workloads.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self._settings = get_neon_settings()
        self.dsn = connection_string or self._settings.database_url
        if not self.dsn:
            raise ValueError("DATABASE_URL not found.")
        if self._settings.db_pool_min_size > self._settings.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE.")
        self._pool_opened = False

        # Initialize Async Connection Pool
        self.pool = AsyncConnectionPool(
            conninfo=self.dsn,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            kwargs=_build_connection_kwargs(self._settings),
            open=False,
        )
        logger.info("Async Database connection pool initialized.")

    async def close(self):
        """Close the async pool."""
        if self.pool:
            await self.pool.close()
            self._pool_opened = False
            logger.info("Async Database connection pool closed.")

    async def __aenter__(self) -> "AsyncNeonDatabaseManager":
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
            await _apply_statement_timeout_async(conn, self._settings)
            yield conn

    async def fetch_all(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[Dict[str, Any]]:
        """Async fetch all."""
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    return await cur.fetchall()
        except psycopg.Error as e:
            logger.error(f"Async DB query failed: {e}")
            raise

    async def fetch_one(self, query: str, params: Optional[Union[tuple, dict]] = None) -> Optional[Dict[str, Any]]:
        """Async fetch one."""
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    return await cur.fetchone()
        except psycopg.Error as e:
            logger.error(f"Async DB fetch_one failed: {e}")
            raise

    async def execute_commit(self, query: str, params: Optional[Union[tuple, dict]] = None) -> None:
        """Async execute and commit."""
        try:
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
        except psycopg.Error as e:
            logger.error(f"Async DB execution failed: {e}")
            raise

    async def fetch_dataframe(self, query: str, params: Optional[Union[tuple, dict]] = None) -> pd.DataFrame:
        """Async execute query and return pandas DataFrame."""
        try:
            # We await the data fetch, then create the dataframe synchronously (fast in memory)
            data = await self.fetch_all(query, params)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to fetch DataFrame asynchronously: {e}")
            raise

    async def create_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        columns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Async create table safely."""
        if (schema is None) == (columns is None):
            raise ValueError("Provide exactly one of schema or columns.")

        try:
            if columns is not None:
                column_defs = [
                    sql.SQL("{} {}").format(
                        sql.Identifier(name),
                        sql.SQL(_validate_sql_type(col_type)),
                    )
                    for name, col_type in columns
                ]
                schema_sql = sql.SQL(", ").join(column_defs)
            else:
                schema_sql = sql.SQL(schema)

            query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
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
            logger.warning("DataFrame is empty, skipping upload.")
            return

        try:
            settings = get_neon_settings()
            df_clean = df.where(pd.notnull(df), None)
            columns = list(df_clean.columns)
            query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table_name),
                sql.SQL(', ').join(map(sql.Identifier, columns)),
                sql.SQL(', ').join(sql.Placeholder() * len(columns))
            )

            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    rows_iter = df_clean.itertuples(index=False, name=None)
                    for batch in _chunked(rows_iter, settings.db_upload_chunk_size):
                        # Async executemany is efficient in psycopg 3
                        await cur.executemany(query, batch)
                await conn.commit()

            logger.info(f"Successfully uploaded {len(df_clean)} rows to '{table_name}' (Async).")
        except psycopg.Error as e:
            logger.error(f"Failed to upload DataFrame to '{table_name}': {e}")
            raise

    async def check_health(self) -> bool:
        try:
            await self.fetch_one("SELECT 1")
            return True
        except Exception:
            return False


class QueueExecutor:
    """
    Runs functions asynchronously using a worker pool pattern with asyncio.Queue.
    """

    def __init__(
        self,
        num_workers: int = 5,
        maxsize: int = 0,
        executor: Optional[Executor] = None,
    ):
        """
        Initializes the QueueExecutor.

        Args:
            num_workers (int): Number of worker coroutines to process tasks.
            maxsize (int): Maximum queue size. 0 means unlimited. When the queue
                is full, submit() will block until space is available (backpressure).
            executor (Optional[Executor]): Optional custom executor for sync functions.
                If None, uses the event loop's default executor.
        """
        if num_workers <= 0:
            raise ValueError('num_workers must be a positive integer')

        self.num_workers = num_workers
        self.maxsize = maxsize
        self.executor = executor
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self):
        """Starts the worker pool. Must be called before submitting tasks."""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(worker_id=i))
            for i in range(self.num_workers)
        ]

    async def stop(self, timeout: Optional[float] = None):
        """Stops the worker pool gracefully.

        Args:
            timeout (Optional[float]): Maximum time to wait for workers to finish.
                If None, waits indefinitely.
        """
        if not self._running:
            return

        self._running = False

        # Send sentinel values to stop workers
        for _ in range(self.num_workers):
            await self.queue.put(None)

        # Wait for workers to finish
        if timeout:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Cancel workers if timeout exceeded
                for worker in self._workers:
                    worker.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
        else:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks from the queue."""
        while self._running:
            try:
                item = await self.queue.get()

                # Sentinel value to stop worker
                if item is None:
                    self.queue.task_done()
                    break

                func, args, kwargs, future = item

                try:
                    # Execute the function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            self.executor, functools.partial(func, *args, **kwargs)
                        )

                    # Set the result on the future
                    if not future.cancelled():
                        future.set_result(result)

                except Exception as e:
                    # Set exception on the future
                    if not future.cancelled():
                        future.set_exception(e)

                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log unexpected errors but keep worker running
                logger.error(f'Worker {worker_id} encountered unexpected error: {e}')

    def submit(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> asyncio.Future:
        """Submits a function to be executed by the worker pool.

        Args:
            func (Callable): The function to execute (sync or async).
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            asyncio.Future: A future that will contain the result.

        Raises:
            RuntimeError: If the executor hasn't been started.
        """
        if not self._running:
            raise RuntimeError(
                'QueueExecutor must be started before submitting tasks. '
                "Use 'async with QueueExecutor(...)' or call 'await executor.start()'"
            )

        future = asyncio.Future()

        async def _submit():
            await self.queue.put((func, args, kwargs, future))
            return future

        return asyncio.create_task(
            _submit_and_return_future(self.queue, func, args, kwargs, future)
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False


async def _submit_and_return_future(
    queue: asyncio.Queue,
    func: Callable,
    args: Tuple,
    kwargs: dict,
    future: asyncio.Future,
) -> Any:
    """Helper to submit to queue and await the future."""
    await queue.put((func, args, kwargs, future))
    return await future
