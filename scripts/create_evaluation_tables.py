from shared.database.neon import NeonConnection

EVALUATION_DATASET_SCHEMA = """
dataset_id TEXT PRIMARY KEY,
query TEXT,
expected_output TEXT,
actual_output TEXT,

additional_input JSONB,
acceptance_criteria JSONB,
dataset_metadata JSONB,
user_tags JSONB,
created_at TIMESTAMPTZ DEFAULT NOW(),

conversation JSONB,
source_type TEXT,
environment TEXT,
source_name TEXT,
source_component TEXT,

tools_called JSONB,
expected_tools JSONB,
retrieved_content JSONB,

judgment JSONB,
critique JSONB,
trace JSONB,
additional_output JSONB,
document_text TEXT,
actual_reference JSONB,
expected_reference JSONB,

latency DOUBLE PRECISION,
trace_id TEXT,
observation_id TEXT,
has_errors BOOLEAN
""".strip()


EVALUATION_RESULTS_SCHEMA = """
run_id TEXT,
dataset_id TEXT REFERENCES evaluation_dataset(dataset_id),
metric_name TEXT,

metric_score DOUBLE PRECISION,
passed BOOLEAN,
explanation TEXT,
metric_type TEXT,
metric_category TEXT,
threshold DOUBLE PRECISION,
signals JSONB,

metric_id TEXT,
parent TEXT,
weight DOUBLE PRECISION,
source TEXT,
metric_metadata JSONB,

evaluation_name TEXT,
eval_mode TEXT,
cost_estimate DOUBLE PRECISION,
model_name TEXT,
llm_provider TEXT,
timestamp TIMESTAMPTZ DEFAULT NOW(),
evaluation_metadata JSONB,
version TEXT,

PRIMARY KEY (run_id, dataset_id, metric_name)
""".strip()

EVALUATION_VIEW_SQL = """
CREATE OR REPLACE VIEW evaluation_view AS
SELECT
    -- Context from Dataset (evaluation_dataset)
    d.dataset_id,
    d.query,
    d.actual_output,
    d.expected_output,
    d.source_type,
    d.environment,
    d.source_name,
    d.source_component,
    d.user_tags,
    d.additional_input,
    d.acceptance_criteria,
    d.dataset_metadata,
    d.created_at AS dataset_created_at,
    d.conversation,
    d.tools_called,
    d.expected_tools,
    d.retrieved_content,
    d.judgment,
    d.critique,
    d.trace,
    d.additional_output,
    d.document_text,
    d.actual_reference,
    d.expected_reference,
    d.latency,
    d.trace_id,
    d.observation_id,
    d.has_errors,

    -- Metrics from Results (evaluation_results)
    r.metric_id,
    r.run_id,
    r.evaluation_name,
    r.metric_name,
    r.metric_score,
    r.passed,
    r.explanation,
    r.metric_type,
    r.metric_category,
    r.threshold,
    r.signals,
    r.parent,
    r.weight,
    r.source,
    r.metric_metadata,
    r.eval_mode,
    r.cost_estimate,
    r.model_name,
    r.llm_provider,
    r.timestamp,
    r.evaluation_metadata,
    r.version
FROM evaluation_results r
JOIN evaluation_dataset d ON r.dataset_id = d.dataset_id;
""".strip()

EVALUATION_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_eval_results_dataset_id
    ON evaluation_results(dataset_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_id
    ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_timestamp
    ON evaluation_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_eval_dataset_created_at
    ON evaluation_dataset(created_at);
""".strip()


def create_evaluation_tables_and_view(db: NeonConnection) -> None:
    db.create_table('evaluation_dataset', schema=EVALUATION_DATASET_SCHEMA)
    db.create_table('evaluation_results', schema=EVALUATION_RESULTS_SCHEMA)
    db.execute_commit(EVALUATION_INDEXES_SQL)
    db.execute_commit(EVALUATION_VIEW_SQL)


# -----------------------------------------------------------------------------
# Reference snippets (commented out) for later use
# -----------------------------------------------------------------------------

# Drop everything (purges data by dropping the objects).
#
# DROP_EVALUATION_OBJECTS_SQL = """
# DROP VIEW IF EXISTS evaluation_view;
# DROP TABLE IF EXISTS evaluation_results;
# DROP TABLE IF EXISTS evaluation_dataset;
# """.strip()
#
# def drop_evaluation_tables_and_view(db: NeonConnection) -> None:
#     db.execute_commit(DROP_EVALUATION_OBJECTS_SQL)


# Delete data but keep the tables (fast).
#
# Note: `evaluation_results` has an FK to `evaluation_dataset`, so either truncate
# both at once (recommended) or truncate results first.
#
# TRUNCATE_EVALUATION_TABLES_SQL = """
# TRUNCATE TABLE evaluation_results, evaluation_dataset;
# """.strip()
#
# def truncate_evaluation_tables(db: NeonConnection) -> None:
#     db.execute_commit(TRUNCATE_EVALUATION_TABLES_SQL)


if __name__ == '__main__':
    with NeonConnection() as db:
        create_evaluation_tables_and_view(db)
