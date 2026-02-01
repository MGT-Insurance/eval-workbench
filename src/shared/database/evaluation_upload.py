from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import pandas as pd


class EvaluationUploadError(RuntimeError):
    """
    Raised when subsetting/normalization/upload fails.

    This wraps lower-level exceptions (e.g. psycopg errors or JSON encoding issues)
    with more context: which table, which column, which row/batch.
    """


# Same order as the table DDL
EVALUATION_DATASET_COLUMNS: list[str] = [
    'id',
    'query',
    'expected_output',
    'actual_output',
    'additional_input',
    'acceptance_criteria',
    'data_metadata',
    'user_tags',
    'created_at',
    'conversation',
    'channel',
    'environment',
    'agent',
    'agent_component',
    'tools_called',
    'expected_tools',
    'retrieved_content',
    'judgment',
    'critique',
    'trace',
    'additional_output',
    'document_text',
    'actual_reference',
    'expected_reference',
    'latency',
    'trace_id',
    'observation_id',
    'error',
]


EVALUATION_RESULTS_COLUMNS: list[str] = [
    'run_id',
    'dataset_id',
    'metric_name',
    'metric_score',
    'passed',
    'explanation',
    'metric_type',
    'metric_category',
    'threshold',
    'signals',
    'metric_id',
    'parent',
    'weight',
    'source',
    'metadata',
    'evaluation_name',
    'cost_estimate',
    'model_name',
    'llm_provider',
    'timestamp',
    'evaluation_metadata',
    'version',
]

_EVALUATION_DATASET_JSONB_COLUMNS: set[str] = {
    'additional_input',
    'acceptance_criteria',
    'data_metadata',
    'user_tags',
    'conversation',
    'tools_called',
    'expected_tools',
    'retrieved_content',
    'judgment',
    'critique',
    'trace',
    'additional_output',
    'actual_reference',
    'expected_reference',
    'error',
}

_EVALUATION_RESULTS_JSONB_COLUMNS: set[str] = {
    'signals',
    'metadata',
    'evaluation_metadata',
}

_EVALUATION_DATASET_REQUIRED_COLUMNS: set[str] = {'id'}
_EVALUATION_RESULTS_REQUIRED_COLUMNS: set[str] = {'run_id', 'dataset_id', 'metric_name'}

_EVALUATION_DATASET_CONFLICT_COLUMNS: tuple[str, ...] = ('id',)
_EVALUATION_RESULTS_CONFLICT_COLUMNS: tuple[str, ...] = (
    'run_id',
    'dataset_id',
    'metric_name',
)


def subset_evaluation_dataset_df_for_upload(
    df: pd.DataFrame, *, include_missing_columns: bool = False
) -> pd.DataFrame:
    """
    Return a copy of `df` containing only columns used by `evaluation_dataset`.

    - Drops unknown/extra columns.
    - If `include_missing_columns=True`, adds any missing table columns with NULLs.
    - If you have `metadata` (DatasetItem field) but not `data_metadata` (aliased key),
      it will rename `metadata` -> `data_metadata` to match the DB schema.
    """
    if df.empty:
        # Preserve expected column order for downstream code paths.
        return pd.DataFrame(columns=EVALUATION_DATASET_COLUMNS)

    out = df.copy()

    if 'data_metadata' not in out.columns and 'metadata' in out.columns:
        out = out.rename(columns={'metadata': 'data_metadata'})

    keep = [c for c in EVALUATION_DATASET_COLUMNS if c in out.columns]
    out = out[keep]

    if include_missing_columns:
        for col in EVALUATION_DATASET_COLUMNS:
            if col not in out.columns:
                out[col] = None
        out = out[EVALUATION_DATASET_COLUMNS]

    return out


def subset_evaluation_results_df_for_upload(
    df: pd.DataFrame,
    *,
    include_missing_columns: bool = False,
    dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] = 'id',
) -> pd.DataFrame:
    """
    Return a copy of `df` containing only columns used by `evaluation_results`.

    Notes:
    - `evaluation_results.dataset_id` is required for the PK/FK. If it's missing,
      this helper can derive it from `id` (default) or `metric_id`.
    - Drops unknown/extra columns.
    - If `include_missing_columns=True`, adds any missing table columns with NULLs.
    """
    if df.empty:
        return pd.DataFrame(columns=EVALUATION_RESULTS_COLUMNS)

    out = df.copy()

    if 'dataset_id' not in out.columns:
        if dataset_id_source == 'dataset_id':
            raise ValueError("dataset_id_source='dataset_id' but column is missing")
        if dataset_id_source not in out.columns:
            raise ValueError(
                f'Cannot derive dataset_id: source column {dataset_id_source!r} is missing'
            )
        out['dataset_id'] = out[dataset_id_source]

    keep = [c for c in EVALUATION_RESULTS_COLUMNS if c in out.columns]
    out = out[keep]

    if include_missing_columns:
        for col in EVALUATION_RESULTS_COLUMNS:
            if col not in out.columns:
                out[col] = None
        out = out[EVALUATION_RESULTS_COLUMNS]

    return out


def _require_columns(df: pd.DataFrame, *, required: set[str], table_name: str) -> None:
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise EvaluationUploadError(
            f'Missing required columns for {table_name}: {missing}. '
            f'Present columns: {sorted(df.columns.tolist())}'
        )


def _as_records_for_upload(
    df: pd.DataFrame,
    *,
    jsonb_columns: set[str],
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to DB-safe records:
    - NaN/NaT -> None
    - JSONB-ish columns:
      - dict/list -> JSON
      - JSON strings -> parsed then JSON (to ensure validity)
      - everything else (None/str/number/bool) passed through
    """
    if df.empty:
        return []

    df_clean = df.astype(object).where(pd.notna(df), None)
    records: list[dict[str, Any]] = df_clean.to_dict(orient='records')

    def _sanitize_for_json(v: Any) -> Any:
        """
        Recursively make a value safe for Postgres JSON/JSONB:
        - NaN/NaT/pandas.NA -> None
        - +/-Inf -> None
        - dict/list sanitized recursively
        """
        # pandas / numpy missing values
        try:
            if v is pd.NA:
                return None
        except Exception:
            pass

        if v is None:
            return None

        # Handle float NaN/Inf
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v

        # Handle pandas Timestamp/NaT-like
        try:
            if pd.isna(v):  # type: ignore[arg-type]
                return None
        except Exception:
            pass

        if isinstance(v, dict):
            return {k: _sanitize_for_json(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_sanitize_for_json(item) for item in v]

        return v

    def _normalize_json_value(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (dict, list)):
            return _sanitize_for_json(v)
        if isinstance(v, str):
            s = v.strip()
            # If it looks like JSON, parse it so we don't store invalid JSON strings.
            if (s.startswith('{') and s.endswith('}')) or (
                s.startswith('[') and s.endswith(']')
            ):
                try:
                    return _sanitize_for_json(json.loads(s))
                except Exception:
                    # Keep raw string if it isn't valid JSON.
                    return v
            return v
        return _sanitize_for_json(v)

    for rec in records:
        for col in jsonb_columns:
            if col in rec:
                rec[col] = _normalize_json_value(rec[col])

    return records


def _validate_jsonb_value(
    *,
    table_name: str,
    column_name: str,
    row_index: int,
    value: Any,
) -> None:
    """
    Ensure a value is encodable as strict JSON (no NaN/Inf, no unserializable types).

    Postgres JSON/JSONB rejects NaN/Inf and requires that dict keys are strings.
    """
    if value is None:
        return
    try:
        json.dumps(value, allow_nan=False)
    except Exception as e:
        preview = repr(value)
        if len(preview) > 400:
            preview = preview[:400] + '... (truncated)'
        raise EvaluationUploadError(
            f'Invalid JSONB value for {table_name}.{column_name} at row {row_index}: {preview}. '
            f'Underlying error: {type(e).__name__}: {e}'
        ) from e


def _insert_records(
    *,
    db: Any,
    table_name: str,
    columns: Sequence[str],
    records: Sequence[dict[str, Any]],
    jsonb_columns: set[str],
    on_conflict: Literal['error', 'do_nothing', 'upsert'] = 'error',
    conflict_columns: Sequence[str] | None = None,
    chunk_size: int = 1000,
) -> None:
    """
    Batch insert records into Postgres using psycopg3 and the repo's NeonConnection.

    `db` is expected to be a `NeonConnection` instance.
    """
    if not records:
        return
    if not columns:
        raise EvaluationUploadError(
            f'No columns provided for insert into {table_name}. '
            'This usually means your DataFrame was empty or fully filtered.'
        )
    if len(set(columns)) != len(list(columns)):
        raise EvaluationUploadError(
            f'Duplicate columns provided for insert into {table_name}: {list(columns)}'
        )

    # Local imports to avoid adding heavy deps on module import.
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Jsonb

    def _chunked(
        items: Sequence[dict[str, Any]], size: int
    ) -> Iterable[Sequence[dict[str, Any]]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    insert_cols = list(columns)
    placeholders = [sql.SQL('%s') for _ in insert_cols]

    conflict_sql = sql.SQL('')
    if on_conflict == 'do_nothing':
        conflict_sql = sql.SQL(' ON CONFLICT DO NOTHING')
    elif on_conflict == 'upsert':
        if not conflict_columns:
            raise EvaluationUploadError(
                f"on_conflict='upsert' requires conflict_columns for table {table_name}"
            )
        conflict_cols = list(conflict_columns)
        missing_conflict = [c for c in conflict_cols if c not in insert_cols]
        if missing_conflict:
            raise EvaluationUploadError(
                f'Upsert conflict columns missing from insert for {table_name}: {missing_conflict}. '
                f'Insert columns: {insert_cols}'
            )
        update_cols = [c for c in insert_cols if c not in set(conflict_cols)]
        if not update_cols:
            conflict_sql = sql.SQL(' ON CONFLICT ({}) DO NOTHING').format(
                sql.SQL(', ').join(map(sql.Identifier, conflict_cols))
            )
        else:
            assignments = [
                sql.SQL('{} = EXCLUDED.{}').format(sql.Identifier(c), sql.Identifier(c))
                for c in update_cols
            ]
            conflict_sql = sql.SQL(' ON CONFLICT ({}) DO UPDATE SET {}').format(
                sql.SQL(', ').join(map(sql.Identifier, conflict_cols)),
                sql.SQL(', ').join(assignments),
            )

    query = sql.SQL('INSERT INTO {} ({}) VALUES ({}){}').format(
        sql.Identifier(table_name),
        sql.SQL(', ').join(map(sql.Identifier, insert_cols)),
        sql.SQL(', ').join(placeholders),
        conflict_sql,
    )

    # Use NeonConnection's internal connection context (applies statement_timeout).
    try:
        with db._connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                batch_start_index = 0
                for batch in _chunked(records, chunk_size):
                    rows = []
                    for i, rec in enumerate(batch):
                        row_index = batch_start_index + i
                        row = []
                        for col in insert_cols:
                            v = rec.get(col)
                            if col in jsonb_columns:
                                _validate_jsonb_value(
                                    table_name=table_name,
                                    column_name=col,
                                    row_index=row_index,
                                    value=v,
                                )
                            if col in jsonb_columns and v is not None:
                                row.append(Jsonb(v))
                            else:
                                row.append(v)
                        rows.append(tuple(row))
                    cur.executemany(query, rows)
                    batch_start_index += len(batch)
            conn.commit()
    except EvaluationUploadError:
        raise
    except psycopg.Error as e:
        raise EvaluationUploadError(
            f'Database insert failed for table {table_name}. '
            f'columns={list(insert_cols)} on_conflict={on_conflict!r} chunk_size={chunk_size}. '
            f'Underlying error: {type(e).__name__}: {e}'
        ) from e


def upload_evaluation_dataset_df(
    db: Any,
    df: pd.DataFrame,
    *,
    include_missing_columns: bool = False,
    on_conflict: Literal['error', 'do_nothing', 'upsert'] = 'error',
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Subset + upload `evaluation_dataset`.

    Returns the DataFrame that was uploaded (after subsetting/normalization).

    Validations:
    - Ensures required PK column(s) exist (`id`)
    - Normalizes JSONB-ish columns (e.g. dict/list or JSON strings)
    """
    upload_df = subset_evaluation_dataset_df_for_upload(
        df, include_missing_columns=include_missing_columns
    )
    _require_columns(
        upload_df,
        required=_EVALUATION_DATASET_REQUIRED_COLUMNS,
        table_name='evaluation_dataset',
    )
    records = _as_records_for_upload(
        upload_df, jsonb_columns=_EVALUATION_DATASET_JSONB_COLUMNS
    )
    _insert_records(
        db=db,
        table_name='evaluation_dataset',
        columns=upload_df.columns.tolist(),
        records=records,
        jsonb_columns=_EVALUATION_DATASET_JSONB_COLUMNS,
        on_conflict=on_conflict,
        conflict_columns=_EVALUATION_DATASET_CONFLICT_COLUMNS
        if on_conflict == 'upsert'
        else None,
        chunk_size=chunk_size,
    )
    return upload_df


def upload_evaluation_results_df(
    db: Any,
    df: pd.DataFrame,
    *,
    include_missing_columns: bool = False,
    dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] = 'id',
    on_conflict: Literal['error', 'do_nothing', 'upsert'] = 'error',
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Subset + upload `evaluation_results`.

    Returns the DataFrame that was uploaded (after subsetting/normalization).

    Validations:
    - Ensures required PK columns exist (`run_id`, `dataset_id`, `metric_name`)
    - If `dataset_id` isn't present, derives it from `dataset_id_source`
    - Normalizes JSONB-ish columns (especially `signals`)
    """
    upload_df = subset_evaluation_results_df_for_upload(
        df,
        include_missing_columns=include_missing_columns,
        dataset_id_source=dataset_id_source,
    )
    _require_columns(
        upload_df,
        required=_EVALUATION_RESULTS_REQUIRED_COLUMNS,
        table_name='evaluation_results',
    )
    records = _as_records_for_upload(
        upload_df, jsonb_columns=_EVALUATION_RESULTS_JSONB_COLUMNS
    )
    _insert_records(
        db=db,
        table_name='evaluation_results',
        columns=upload_df.columns.tolist(),
        records=records,
        jsonb_columns=_EVALUATION_RESULTS_JSONB_COLUMNS,
        on_conflict=on_conflict,
        conflict_columns=_EVALUATION_RESULTS_CONFLICT_COLUMNS
        if on_conflict == 'upsert'
        else None,
        chunk_size=chunk_size,
    )
    return upload_df


def subset_df_for_upload(
    df: pd.DataFrame,
    *,
    table: Literal['evaluation_dataset', 'evaluation_results'],
    include_missing_columns: bool = False,
    dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] = 'id',
) -> pd.DataFrame:
    """
    Convenience wrapper to subset a DataFrame to exactly the DB columns.

    This **does not** upload. Use `upload_df(...)` or `EvaluationUploader`.
    """
    if table == 'evaluation_dataset':
        return subset_evaluation_dataset_df_for_upload(
            df, include_missing_columns=include_missing_columns
        )
    if table == 'evaluation_results':
        return subset_evaluation_results_df_for_upload(
            df,
            include_missing_columns=include_missing_columns,
            dataset_id_source=dataset_id_source,
        )
    raise ValueError(f'Unknown table: {table!r}')


def upload_df(
    db: Any,
    df: pd.DataFrame,
    *,
    table: Literal['evaluation_dataset', 'evaluation_results'],
    include_missing_columns: bool = False,
    dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] = 'id',
    on_conflict: Literal['error', 'do_nothing', 'upsert'] = 'error',
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Convenience wrapper to subset + upload for either table.

    - `table="evaluation_dataset"` writes to `evaluation_dataset`
    - `table="evaluation_results"` writes to `evaluation_results`
    """
    if table == 'evaluation_dataset':
        return upload_evaluation_dataset_df(
            db,
            df,
            include_missing_columns=include_missing_columns,
            on_conflict=on_conflict,
            chunk_size=chunk_size,
        )
    if table == 'evaluation_results':
        return upload_evaluation_results_df(
            db,
            df,
            include_missing_columns=include_missing_columns,
            dataset_id_source=dataset_id_source,
            on_conflict=on_conflict,
            chunk_size=chunk_size,
        )
    raise ValueError(f'Unknown table: {table!r}')


@dataclass
class EvaluationUploader:
    """
    Helper to subset + upload evaluation dataset/results DataFrames.

    Typical usage:

        from shared.database.neon import NeonConnection
        from shared.database.evaluation_upload import EvaluationUploader

        with NeonConnection() as db:
            uploader = EvaluationUploader(db=db, on_conflict="do_nothing")
            uploader.upload_dataset(dataset_df)
            uploader.upload_results(metrics_df, dataset_id_source="id")

    Key options:
    - `on_conflict="do_nothing"`: skip duplicates instead of raising (idempotent loads)
    - `dataset_id_source="id"`: derive `evaluation_results.dataset_id` from a source column
    - `chunk_size`: number of rows per `executemany()` batch
    """

    db: Any | None = None
    include_missing_columns: bool = False
    dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] = 'id'
    on_conflict: Literal['error', 'do_nothing', 'upsert'] = 'error'
    chunk_size: int = 1000

    def subset_dataset(
        self, df: pd.DataFrame, *, include_missing_columns: bool | None = None
    ) -> pd.DataFrame:
        """Subset `dataset_df` to `evaluation_dataset` columns only."""
        return subset_evaluation_dataset_df_for_upload(
            df,
            include_missing_columns=(
                self.include_missing_columns
                if include_missing_columns is None
                else include_missing_columns
            ),
        )

    def subset_results(
        self,
        df: pd.DataFrame,
        *,
        include_missing_columns: bool | None = None,
        dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] | None = None,
    ) -> pd.DataFrame:
        """Subset `metrics_df` to `evaluation_results` columns only (derives `dataset_id` if needed)."""
        return subset_evaluation_results_df_for_upload(
            df,
            include_missing_columns=(
                self.include_missing_columns
                if include_missing_columns is None
                else include_missing_columns
            ),
            dataset_id_source=(
                self.dataset_id_source
                if dataset_id_source is None
                else dataset_id_source
            ),
        )

    def subset(
        self,
        df: pd.DataFrame,
        *,
        table: Literal['evaluation_dataset', 'evaluation_results'],
        include_missing_columns: bool | None = None,
        dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] | None = None,
    ) -> pd.DataFrame:
        """Subset to the specified table's columns (no upload)."""
        return subset_df_for_upload(
            df,
            table=table,
            include_missing_columns=(
                self.include_missing_columns
                if include_missing_columns is None
                else include_missing_columns
            ),
            dataset_id_source=(
                self.dataset_id_source
                if dataset_id_source is None
                else dataset_id_source
            ),
        )

    def upload_dataset(
        self,
        df: pd.DataFrame,
        *,
        db: Any | None = None,
        include_missing_columns: bool | None = None,
        on_conflict: Literal['error', 'do_nothing', 'upsert'] | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Subset + upload to `evaluation_dataset`.

        Raises `EvaluationUploadError` with context if:
        - required columns are missing
        - JSONB values are not valid JSON
        - the database rejects the insert
        """
        _db = self.db if db is None else db
        if _db is None:
            raise ValueError(
                'No db provided. Pass db=... or set EvaluationUploader(db=...).'
            )
        return upload_evaluation_dataset_df(
            _db,
            df,
            include_missing_columns=(
                self.include_missing_columns
                if include_missing_columns is None
                else include_missing_columns
            ),
            on_conflict=self.on_conflict if on_conflict is None else on_conflict,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )

    def upload_results(
        self,
        df: pd.DataFrame,
        *,
        db: Any | None = None,
        include_missing_columns: bool | None = None,
        dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] | None = None,
        on_conflict: Literal['error', 'do_nothing', 'upsert'] | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Subset + upload to `evaluation_results`.

        `dataset_id` is required; if it's missing, it will be derived from
        `dataset_id_source` (default: `"id"`).
        """
        _db = self.db if db is None else db
        if _db is None:
            raise ValueError(
                'No db provided. Pass db=... or set EvaluationUploader(db=...).'
            )
        return upload_evaluation_results_df(
            _db,
            df,
            include_missing_columns=(
                self.include_missing_columns
                if include_missing_columns is None
                else include_missing_columns
            ),
            dataset_id_source=(
                self.dataset_id_source
                if dataset_id_source is None
                else dataset_id_source
            ),
            on_conflict=self.on_conflict if on_conflict is None else on_conflict,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )

    def upload(
        self,
        df: pd.DataFrame,
        *,
        table: Literal['evaluation_dataset', 'evaluation_results'],
        db: Any | None = None,
        include_missing_columns: bool | None = None,
        dataset_id_source: Literal['dataset_id', 'id', 'metric_id'] | None = None,
        on_conflict: Literal['error', 'do_nothing', 'upsert'] | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Generic subset + upload for either table."""
        _db = self.db if db is None else db
        if _db is None:
            raise ValueError(
                'No db provided. Pass db=... or set EvaluationUploader(db=...).'
            )
        return upload_df(
            _db,
            df,
            table=table,
            include_missing_columns=(
                self.include_missing_columns
                if include_missing_columns is None
                else include_missing_columns
            ),
            dataset_id_source=(
                self.dataset_id_source
                if dataset_id_source is None
                else dataset_id_source
            ),
            on_conflict=self.on_conflict if on_conflict is None else on_conflict,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )
