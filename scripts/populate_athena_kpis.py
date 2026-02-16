#!/usr/bin/env python3
"""
Populate the agent_kpi_logs table from athena_cases and evaluation_results.

Supports reading from one database (source) and writing to another (target).
Safe to re-run - deduplicates against existing dataset_id+kpi_name in target.

Usage:
  python scripts/populate_athena_kpis.py --since-days 2

Environment variables:
  - SOURCE_DATABASE_URL: Source Postgres DB URL (optional)
  - TARGET_DATABASE_URL: Target Postgres DB URL (optional)
  - DATABASE_URL: Fallback URL if source/target URLs are not provided

Write modes:
  - incremental (default): insert only rows not already in agent_kpi_logs
  - backfill: overwrite matching (dataset_id, kpi_name) rows in the lookback window
"""

from __future__ import annotations

import argparse
import logging
import os

from eval_workbench.shared.database.neon import NeonConnection

# ---------------------------------------------------------------------------
# Case-derived KPIs — SELECT queries (run against source DB)
#
# athena_cases columns used:
#   - quote_locator            TEXT  (unique case identifier → dataset_id)
#   - created_at               TIMESTAMPTZ
#   - recommendation_entries   JSONB (array of recommendation objects)
#
# Each recommendation entry object contains:
#   - briefRecommendation   TEXT
#   - underwritingFlags     JSONB array
#   - executedAt            TEXT (ISO timestamp)
#   - executionTimeMs       INTEGER
#
# We use a lateral join to extract the latest recommendation entry per case.
# ---------------------------------------------------------------------------

_LATEST_ENTRY_CTE = """
WITH latest_entry AS (
    SELECT
        ac.quote_locator,
        ac.created_at,
        entry->>'briefRecommendation'    AS brief_recommendation,
        entry->'underwritingFlags'       AS underwriting_flags,
        entry->>'executedAt'             AS executed_at,
        (entry->>'executionTimeMs')::DOUBLE PRECISION AS execution_time_ms
    FROM athena_cases ac,
    LATERAL (
        SELECT e AS entry
        FROM jsonb_array_elements(ac.recommendation_entries) AS e
        ORDER BY e->>'executedAt' DESC NULLS LAST
        LIMIT 1
    ) lat
    WHERE ac.created_at >= NOW() - INTERVAL '%(since_days)s days'
      AND ac.recommendation_entries IS NOT NULL
      AND jsonb_array_length(ac.recommendation_entries) > 0
)
"""

SELECT_STP_RATE = (
    _LATEST_ENTRY_CTE
    + """
SELECT
    'athena'                 AS source_name,
    'stp_rate'               AS kpi_name,
    'operational_efficiency' AS kpi_category,
    le.quote_locator         AS dataset_id,
    CASE
        WHEN jsonb_array_length(COALESCE(le.underwriting_flags, '[]'::jsonb)) = 0
         AND le.brief_recommendation IS NOT NULL
        THEN 1.0
        ELSE 0.0
    END                      AS numeric_value,
    'underwriter'            AS source_component,
    le.created_at
FROM latest_entry le
"""
)

SELECT_TIME_TO_QUOTE = (
    _LATEST_ENTRY_CTE
    + """
SELECT
    'athena'                 AS source_name,
    'time_to_quote'          AS kpi_name,
    'operational_efficiency' AS kpi_category,
    le.quote_locator         AS dataset_id,
    le.execution_time_ms / 1000.0 AS numeric_value,
    'underwriter'            AS source_component,
    le.created_at
FROM latest_entry le
WHERE le.execution_time_ms IS NOT NULL
"""
)

SELECT_REFERRAL_RATE = (
    _LATEST_ENTRY_CTE
    + """
SELECT
    'athena'                 AS source_name,
    'referral_rate'          AS kpi_name,
    'operational_efficiency' AS kpi_category,
    le.quote_locator         AS dataset_id,
    CASE
        WHEN jsonb_array_length(COALESCE(le.underwriting_flags, '[]'::jsonb)) > 0
        THEN 1.0
        ELSE 0.0
    END                      AS numeric_value,
    'underwriter'            AS source_component,
    le.created_at
FROM latest_entry le
"""
)

SELECT_BINDABLE_QUOTE_RATE = (
    _LATEST_ENTRY_CTE
    + """
SELECT
    'athena'                 AS source_name,
    'bindable_quote_rate'    AS kpi_name,
    'commercial_impact'      AS kpi_category,
    le.quote_locator         AS dataset_id,
    CASE
        WHEN jsonb_array_length(COALESCE(le.underwriting_flags, '[]'::jsonb)) = 0
        THEN 1.0
        ELSE 0.0
    END                      AS numeric_value,
    'underwriter'            AS source_component,
    le.created_at
FROM latest_entry le
"""
)

# ---------------------------------------------------------------------------
# Evaluation-derived KPIs — SELECT queries (run against source DB)
#
# evaluation_results columns used:
#   - dataset_id     TEXT
#   - metric_name    TEXT
#   - metric_score   DOUBLE PRECISION
#   - signals        JSONB  (flat Pydantic model_dump)
#   - timestamp      TIMESTAMPTZ
# ---------------------------------------------------------------------------

SELECT_DECISION_VARIANCE = """
SELECT
    'athena'            AS source_name,
    'decision_variance' AS kpi_name,
    'risk_accuracy'     AS kpi_category,
    er.dataset_id,
    CASE
        WHEN er.signals->>'outcome_match' = 'true' THEN 0.0
        ELSE 1.0
    END                 AS numeric_value,
    'evaluator'         AS source_component,
    er.timestamp        AS created_at
FROM evaluation_results er
WHERE er.metric_name = 'Decision Quality'
  AND er.timestamp >= NOW() - INTERVAL '%(since_days)s days'
"""

SELECT_REFERRAL_ACCURACY = """
SELECT
    'athena'              AS source_name,
    'referral_accuracy'   AS kpi_name,
    'risk_accuracy'       AS kpi_category,
    er.dataset_id,
    er.metric_score       AS numeric_value,
    'evaluator'           AS source_component,
    er.timestamp          AS created_at
FROM evaluation_results er
WHERE er.metric_name = 'Refer Reason'
  AND er.metric_score IS NOT NULL
  AND er.timestamp >= NOW() - INTERVAL '%(since_days)s days'
"""

SELECT_FAITHFULNESS_SCORE = """
SELECT
    'athena'              AS source_name,
    'faithfulness_score'  AS kpi_name,
    'data_integrity'      AS kpi_category,
    er.dataset_id,
    er.metric_score       AS numeric_value,
    'evaluator'           AS source_component,
    er.timestamp          AS created_at
FROM evaluation_results er
WHERE er.metric_name = 'UW Faithfulness'
  AND er.metric_score IS NOT NULL
  AND er.timestamp >= NOW() - INTERVAL '%(since_days)s days'
"""

SELECT_HALLUCINATION_COUNT = """
SELECT
    'athena'               AS source_name,
    'hallucination_count'  AS kpi_name,
    'data_integrity'       AS kpi_category,
    er.dataset_id,
    (er.signals->>'hallucinations')::DOUBLE PRECISION AS numeric_value,
    'evaluator'            AS source_component,
    er.timestamp           AS created_at
FROM evaluation_results er
WHERE er.metric_name = 'UW Faithfulness'
  AND er.signals->>'hallucinations' IS NOT NULL
  AND er.timestamp >= NOW() - INTERVAL '%(since_days)s days'
"""

# ---------------------------------------------------------------------------
# Write query (run against target DB)
# ---------------------------------------------------------------------------

INSERT_KPI_ROW = """
INSERT INTO agent_kpi_logs
    (source_name, kpi_name, kpi_category, dataset_id, numeric_value,
     source_component, created_at)
VALUES (%(source_name)s, %(kpi_name)s, %(kpi_category)s, %(dataset_id)s,
        %(numeric_value)s, %(source_component)s, %(created_at)s)
"""

# We need a unique constraint for ON CONFLICT to work. If it doesn't exist,
# fall back to NOT EXISTS deduplication.
INSERT_KPI_ROW_DEDUP = """
INSERT INTO agent_kpi_logs
    (source_name, kpi_name, kpi_category, dataset_id, numeric_value,
     source_component, created_at)
SELECT %(source_name)s, %(kpi_name)s, %(kpi_category)s, %(dataset_id)s,
       %(numeric_value)s, %(source_component)s, %(created_at)s
WHERE NOT EXISTS (
    SELECT 1 FROM agent_kpi_logs
    WHERE dataset_id = %(dataset_id)s AND kpi_name = %(kpi_name)s
)
"""

DELETE_KPI_ROW_BY_KEY = """
DELETE FROM agent_kpi_logs
WHERE dataset_id = %(dataset_id)s
  AND kpi_name = %(kpi_name)s
"""

# ---------------------------------------------------------------------------
# KPI definitions
# ---------------------------------------------------------------------------

ALL_KPI_QUERIES = [
    ('stp_rate', SELECT_STP_RATE),
    ('time_to_quote', SELECT_TIME_TO_QUOTE),
    ('referral_rate', SELECT_REFERRAL_RATE),
    ('bindable_quote_rate', SELECT_BINDABLE_QUOTE_RATE),
    ('decision_variance', SELECT_DECISION_VARIANCE),
    ('referral_accuracy', SELECT_REFERRAL_ACCURACY),
    ('faithfulness_score', SELECT_FAITHFULNESS_SCORE),
    ('hallucination_count', SELECT_HALLUCINATION_COUNT),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def populate_athena_kpis(
    source_db: NeonConnection,
    target_db: NeonConnection,
    since_days: int = 30,
    write_mode: str = 'incremental',
) -> dict[str, int]:
    """Populate agent_kpi_logs by reading from source_db and writing to target_db.

    Returns a dict mapping kpi_name -> rows inserted.
    """
    if write_mode not in {'incremental', 'backfill'}:
        raise ValueError("write_mode must be 'incremental' or 'backfill'.")

    results: dict[str, int] = {}

    for kpi_name, select_query in ALL_KPI_QUERIES:
        query = select_query % {'since_days': since_days}
        rows = source_db.fetch_all(query)

        if not rows:
            results[kpi_name] = 0
            continue

        inserted = 0
        with target_db._connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    if write_mode == 'backfill':
                        # Backfill mode refreshes matched keys in-place.
                        cur.execute(DELETE_KPI_ROW_BY_KEY, row)
                        cur.execute(INSERT_KPI_ROW, row)
                        inserted += 1
                    else:
                        cur.execute(INSERT_KPI_ROW_DEDUP, row)
                        inserted += cur.rowcount
            conn.commit()

        results[kpi_name] = inserted

    return results


def _resolve_db_urls(
    source_database_url: str | None,
    target_database_url: str | None,
) -> tuple[str, str]:
    source_url = (
        source_database_url
        or os.environ.get('SOURCE_DATABASE_URL')
        or os.environ.get('DATABASE_URL')
    )
    target_url = (
        target_database_url
        or os.environ.get('TARGET_DATABASE_URL')
        or os.environ.get('DATABASE_URL')
    )
    if not source_url:
        raise ValueError(
            'Missing source database URL. Set SOURCE_DATABASE_URL or DATABASE_URL.'
        )
    if not target_url:
        raise ValueError(
            'Missing target database URL. Set TARGET_DATABASE_URL or DATABASE_URL.'
        )
    return source_url, target_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Populate agent_kpi_logs from athena_cases and evaluation_results.'
    )
    parser.add_argument(
        '--since-days',
        type=int,
        default=30,
        help='Lookback window in days for source KPI rows (default: 30).',
    )
    parser.add_argument(
        '--source-database-url',
        default=None,
        help='Source DB URL (falls back to SOURCE_DATABASE_URL then DATABASE_URL).',
    )
    parser.add_argument(
        '--target-database-url',
        default=None,
        help='Target DB URL (falls back to TARGET_DATABASE_URL then DATABASE_URL).',
    )
    parser.add_argument(
        '--write-mode',
        choices=['incremental', 'backfill'],
        default='incremental',
        help='Write behavior: incremental (default) or backfill overwrite.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    if args.since_days <= 0:
        logger.error('--since-days must be greater than 0.')
        return 2

    try:
        source_url, target_url = _resolve_db_urls(
            source_database_url=args.source_database_url,
            target_database_url=args.target_database_url,
        )
    except ValueError as e:
        logger.error('%s', e)
        return 2

    with NeonConnection(connection_string=source_url) as source_db:
        with NeonConnection(connection_string=target_url) as target_db:
            results = populate_athena_kpis(
                source_db=source_db,
                target_db=target_db,
                since_days=args.since_days,
                write_mode=args.write_mode,
            )

    total_inserted = sum(results.values())
    logger.info('Inserted KPI rows: %s', total_inserted)
    for kpi_name, count in results.items():
        logger.info('  %s: %s', kpi_name, count)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
