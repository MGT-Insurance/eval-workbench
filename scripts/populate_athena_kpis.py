"""
Populate the agent_kpi_logs table from athena_cases and evaluation_results.

Supports reading from one database (source) and writing to another (target).
Safe to re-run — deduplicates against existing dataset_id+kpi_name in target.
"""

from __future__ import annotations

from shared.database.neon import NeonConnection

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
ON CONFLICT DO NOTHING
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
) -> dict[str, int]:
    """Populate agent_kpi_logs by reading from source_db and writing to target_db.

    Returns a dict mapping kpi_name -> rows inserted.
    """
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
                    cur.execute(INSERT_KPI_ROW_DEDUP, row)
                    inserted += cur.rowcount
            conn.commit()

        results[kpi_name] = inserted

    return results
