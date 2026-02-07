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
ingested_at         TIMESTAMPTZ
""".strip()

RULE_EXTRACTIONS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_rule_extractions_status     ON rule_extractions (ingestion_status);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_agent      ON rule_extractions (agent_name);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_batch      ON rule_extractions (batch_id);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_rule_name  ON rule_extractions (rule_name);
CREATE INDEX IF NOT EXISTS idx_rule_extractions_risk       ON rule_extractions (risk_factor, product_type);
""".strip()


def create_rule_extractions_table(db: NeonConnection) -> None:
    db.create_table('rule_extractions', schema=RULE_EXTRACTIONS_SCHEMA)
    db.execute_commit(RULE_EXTRACTIONS_INDEXES)


if __name__ == '__main__':
    with NeonConnection() as db:
        create_rule_extractions_table(db)
