from shared.database.neon import NeonConnection

RULE_EXTRACTIONS_SCHEMA = """
id                  TEXT PRIMARY KEY,
created_at          TIMESTAMPTZ DEFAULT NOW(),
batch_id            TEXT,
agent_name          TEXT NOT NULL DEFAULT 'athena',

raw_text            TEXT,
raw_text_hash       TEXT,

risk_factor         TEXT NOT NULL,
risk_category       TEXT,
rule_name           TEXT NOT NULL,
product_type        TEXT,
action              TEXT,
outcome_description TEXT,

mitigants           JSONB,
source              TEXT,
source_type         TEXT,
confidence          TEXT,

threshold           JSONB,
threshold_type      TEXT,
historical_exceptions TEXT,

decision_quality    TEXT,
compound_trigger    TEXT,
data_fields         JSONB,

ingestion_status    TEXT NOT NULL DEFAULT 'pending',
ingestion_error     TEXT,
ingested_at         TIMESTAMPTZ,

-- Provenance fields (added for CSV/Slack ingestion pipeline)
kb_entry_id         VARCHAR,
learning_id         VARCHAR,
source_dataset      VARCHAR,
source_category     VARCHAR,
proposal_kind       VARCHAR,
approval_status     VARCHAR,
slack_channel_id    VARCHAR,
slack_thread_ts     VARCHAR,
langfuse_trace_id   VARCHAR,
extractor_version   VARCHAR,
superseded_by       VARCHAR,
confidence_factors  JSONB,
evidence_snippet    TEXT,
review_status       VARCHAR DEFAULT 'pending_review',
reviewed_by         VARCHAR,
reviewed_at         TIMESTAMPTZ
""".strip()

RULE_EXTRACTIONS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_rule_extractions_status     ON rule_extractions (ingestion_status);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_agent      ON rule_extractions (agent_name);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_batch      ON rule_extractions (batch_id);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_rule_name  ON rule_extractions (rule_name);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_risk       ON rule_extractions (risk_factor, product_type);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_review     ON rule_extractions (review_status) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS idx_rule_extractions_kb_entry   ON rule_extractions (kb_entry_id) WHERE kb_entry_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rule_extractions_learning   ON rule_extractions (learning_id) WHERE learning_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rule_extractions_slack      ON rule_extractions (slack_channel_id, slack_thread_ts) WHERE slack_channel_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rule_extractions_source_ds  ON rule_extractions (source_dataset);
""".strip()

# ALTER TABLE statements for migrating existing tables
PROVENANCE_MIGRATION = """
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS kb_entry_id VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS learning_id VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS source_dataset VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS source_category VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS proposal_kind VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS approval_status VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS slack_channel_id VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS slack_thread_ts VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS langfuse_trace_id VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS extractor_version VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS confidence_factors JSONB;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS evidence_snippet TEXT;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS review_status VARCHAR DEFAULT 'pending_review';
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS reviewed_by VARCHAR;
ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;
""".strip()


def create_rule_extractions_table(db: NeonConnection) -> None:
    db.create_table('rule_extractions', schema=RULE_EXTRACTIONS_SCHEMA)
    db.execute_commit(RULE_EXTRACTIONS_INDEXES)


def migrate_rule_extractions_table(db: NeonConnection) -> None:
    """Add provenance columns to an existing rule_extractions table."""
    for statement in PROVENANCE_MIGRATION.split('\n'):
        statement = statement.strip()
        if statement:
            db.execute_commit(statement)
    db.execute_commit(RULE_EXTRACTIONS_INDEXES)


if __name__ == '__main__':
    import sys

    with NeonConnection() as db:
        if '--migrate' in sys.argv:
            migrate_rule_extractions_table(db)
            print('Migration complete: provenance columns added.')
        else:
            create_rule_extractions_table(db)
            print('Table created.')
