from shared.database.neon import NeonConnection

# ---------------------------------------------------------------------------
# agent_kpi_logs — EAV table for all agent KPI events.
# Each row is a single KPI observation. The DuckDB dashboard layer pivots
# kpi_name into columns for the dashboard view.
# ---------------------------------------------------------------------------

AGENT_KPI_LOGS_SCHEMA = """
id                  TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
source_name         TEXT NOT NULL,
kpi_name            TEXT NOT NULL,
kpi_category        TEXT NOT NULL,

dataset_id          TEXT,

numeric_value       DOUBLE PRECISION,
text_value          TEXT,
json_value          JSONB,

source_component    TEXT,
source_step         TEXT,
environment         TEXT NOT NULL DEFAULT 'production',

tags                JSONB,
metadata            JSONB
""".strip()

AGENT_KPI_LOGS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_kpi_logs_source_kpi
    ON agent_kpi_logs (source_name, kpi_name);
CREATE INDEX IF NOT EXISTS idx_kpi_logs_category
    ON agent_kpi_logs (kpi_category);
CREATE INDEX IF NOT EXISTS idx_kpi_logs_dataset_id
    ON agent_kpi_logs (dataset_id);
CREATE INDEX IF NOT EXISTS idx_kpi_logs_created_at
    ON agent_kpi_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_kpi_logs_source_kpi_time
    ON agent_kpi_logs (source_name, kpi_name, created_at DESC);
""".strip()


def create_athena_kpi_objects(db: NeonConnection) -> None:
    db.create_table('agent_kpi_logs', schema=AGENT_KPI_LOGS_SCHEMA)
    db.execute_commit(AGENT_KPI_LOGS_INDEXES)


# ---------------------------------------------------------------------------
# Sample queries — run interactively with NeonConnection to test
# ---------------------------------------------------------------------------

# -- Insert test KPI rows --------------------------------------------------
#
# INSERT_SAMPLE_ROWS = """
# INSERT INTO agent_kpi_logs
#     (source_name, kpi_name, kpi_category, dataset_id, numeric_value, source_component)
# VALUES
#     ('athena', 'stp_rate',                 'operational_efficiency', 'QL-001', 1.0,  'underwriter'),
#     ('athena', 'stp_rate',                 'operational_efficiency', 'QL-002', 0.0,  'underwriter'),
#     ('athena', 'time_to_quote',            'operational_efficiency', 'QL-001', 12.4, 'underwriter'),
#     ('athena', 'time_to_quote',            'operational_efficiency', 'QL-002', 45.1, 'underwriter'),
#     ('athena', 'referral_rate',            'operational_efficiency', 'QL-002', 1.0,  'underwriter'),
#     ('athena', 'bindable_quote_rate',      'commercial_impact',     'QL-001', 1.0,  'underwriter'),
#     ('athena', 'decision_variance',        'risk_accuracy',         'QL-001', 0.0,  'evaluator'),
#     ('athena', 'decision_variance',        'risk_accuracy',         'QL-002', 1.0,  'evaluator'),
#     ('athena', 'referral_accuracy',        'risk_accuracy',         'QL-002', 0.85, 'evaluator'),
#     ('athena', 'faithfulness_score',       'data_integrity',        'QL-001', 0.95, 'evaluator'),
#     ('athena', 'hallucination_count',      'data_integrity',        'QL-001', 0.0,  'evaluator'),
#     ('athena', 'hallucination_count',      'data_integrity',        'QL-002', 2.0,  'evaluator'),
#     ('athena', 'data_extraction_confidence','risk_accuracy',        'QL-001', 0.92, 'pdf_parser'),
#     ('athena', 'data_extraction_confidence','risk_accuracy',        'QL-002', 0.41, 'pdf_parser'),
#     ('athena', 'quote_amount',             'commercial_impact',     'QL-001', 15000.0, 'underwriter');
# """.strip()

# -- Read back all rows ----------------------------------------------------
#
# SELECT_ALL = "SELECT * FROM agent_kpi_logs ORDER BY created_at DESC;"

# -- Filter by KPI name ----------------------------------------------------
#
# SELECT_BY_KPI = """
# SELECT created_at, source_name, dataset_id, numeric_value
# FROM agent_kpi_logs
# WHERE kpi_name = 'stp_rate'
# ORDER BY created_at DESC;
# """.strip()

# -- Filter by category ----------------------------------------------------
#
# SELECT_BY_CATEGORY = """
# SELECT kpi_name, COUNT(*) AS observations, AVG(numeric_value) AS avg_value
# FROM agent_kpi_logs
# WHERE kpi_category = 'operational_efficiency'
# GROUP BY kpi_name;
# """.strip()

# -- Time-series for a single KPI (last 30 days) --------------------------
#
# SELECT_TIME_SERIES = """
# SELECT
#     date_trunc('day', created_at) AS period,
#     COUNT(*)                      AS observations,
#     AVG(numeric_value)            AS avg_value,
#     MIN(numeric_value)            AS min_value,
#     MAX(numeric_value)            AS max_value
# FROM agent_kpi_logs
# WHERE kpi_name = 'time_to_quote'
#   AND created_at >= NOW() - INTERVAL '30 days'
# GROUP BY 1
# ORDER BY 1;
# """.strip()

# -- Per-dataset_id summary (all KPIs for one case) ------------------------
#
# SELECT_PER_CASE = """
# SELECT kpi_name, numeric_value, text_value, json_value, source_component
# FROM agent_kpi_logs
# WHERE dataset_id = 'QL-001'
# ORDER BY kpi_name;
# """.strip()

# -- DuckDB pivot (the "explosion" for the dashboard view) -----------------
#
# DUCKDB_PIVOT = """
# -- Run in DuckDB after ATTACHing Postgres:
# --   INSTALL postgres; LOAD postgres;
# --   ATTACH '<connection_string>' AS pg (TYPE POSTGRES);
#
# CREATE OR REPLACE VIEW dashboard_view AS
# SELECT
#     created_at::DATE                                                        AS report_date,
#     source_name,
#     dataset_id,
#     MAX(CASE WHEN kpi_name = 'stp_rate'                  THEN numeric_value END) AS stp_rate,
#     MAX(CASE WHEN kpi_name = 'time_to_quote'             THEN numeric_value END) AS time_to_quote,
#     MAX(CASE WHEN kpi_name = 'referral_rate'             THEN numeric_value END) AS referral_rate,
#     MAX(CASE WHEN kpi_name = 'bindable_quote_rate'       THEN numeric_value END) AS bindable_quote_rate,
#     MAX(CASE WHEN kpi_name = 'decision_variance'         THEN numeric_value END) AS decision_variance,
#     MAX(CASE WHEN kpi_name = 'referral_accuracy'         THEN numeric_value END) AS referral_accuracy,
#     MAX(CASE WHEN kpi_name = 'faithfulness_score'        THEN numeric_value END) AS faithfulness_score,
#     MAX(CASE WHEN kpi_name = 'hallucination_count'       THEN numeric_value END) AS hallucination_count,
#     MAX(CASE WHEN kpi_name = 'data_extraction_confidence' THEN numeric_value END) AS data_extraction_confidence,
#     MAX(CASE WHEN kpi_name = 'quote_amount'              THEN numeric_value END) AS quote_amount
# FROM pg.agent_kpi_logs
# WHERE created_at >= NOW() - INTERVAL '30 days'
# GROUP BY 1, 2, 3;
# """.strip()

# -- Cleanup (drop everything) --------------------------------------------
#
# DROP_KPI_OBJECTS = """
# DROP TABLE IF EXISTS agent_kpi_logs;
# """.strip()

# -- Truncate (keep table, delete data) ------------------------------------
#
# TRUNCATE_KPI_LOGS = "TRUNCATE TABLE agent_kpi_logs;"


if __name__ == '__main__':
    with NeonConnection() as db:
        create_athena_kpi_objects(db)
