from __future__ import annotations

import hashlib
import logging
import uuid

from psycopg.types.json import Json

from eval_workbench.shared.database.neon import NeonConnection
from eval_workbench.shared.memory.enums import (
    IngestionStatus,
    ProposalKind,
    ReviewStatus,
    SourceCategory,
    SourceDataset,
)

logger = logging.getLogger(__name__)

_PROVENANCE_SCHEMA_ENSURED = False

# Post-launch provenance/lifecycle columns that may be missing on older tables.
_PROVENANCE_MIGRATION_COLUMNS: dict[str, str] = {
    'kb_entry_id': 'VARCHAR',
    'learning_id': 'VARCHAR',
    'source_dataset': 'VARCHAR',
    'source_category': 'VARCHAR',
    'proposal_kind': 'VARCHAR',
    'approval_status': 'VARCHAR',
    'slack_channel_id': 'VARCHAR',
    'slack_thread_ts': 'VARCHAR',
    'langfuse_trace_id': 'VARCHAR',
    'extractor_version': 'VARCHAR',
    'superseded_by': 'VARCHAR',
    'confidence_factors': 'JSONB',
    'evidence_snippet': 'TEXT',
    "review_status": "VARCHAR DEFAULT 'pending_review'",
    'reviewed_by': 'VARCHAR',
    'reviewed_at': 'TIMESTAMPTZ',
}


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def compute_text_hash(text: str) -> str:
    """SHA-256 of input text for dedup."""
    return hashlib.sha256(text.encode()).hexdigest()


def compute_extractor_version(system_prompt: str, model: str, formatting_version: str = '1') -> str:
    """Short hash of extractor configuration for provenance tracking."""
    payload = f'{system_prompt}{model}{formatting_version}'
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core persistence — save_extractions
# ---------------------------------------------------------------------------

# Columns that always get written (core extraction fields)
_CORE_COLUMNS = [
    'id', 'batch_id', 'agent_name',
    'raw_text', 'raw_text_hash',
    'risk_factor', 'risk_category', 'rule_name', 'product_type',
    'action', 'outcome_description',
    'mitigants', 'source', 'source_type', 'confidence',
    'threshold', 'threshold_type', 'historical_exceptions',
    'decision_quality', 'compound_trigger', 'data_fields',
]

# Provenance columns (optional, populated by CSV/Slack pipelines)
_PROVENANCE_COLUMNS = [
    'kb_entry_id', 'learning_id', 'source_dataset', 'source_category',
    'proposal_kind', 'approval_status',
    'slack_channel_id', 'slack_thread_ts', 'langfuse_trace_id',
    'extractor_version', 'confidence_factors', 'evidence_snippet',
    'review_status',
]

_ALL_COLUMNS = _CORE_COLUMNS + _PROVENANCE_COLUMNS


def _json_or_none(value):
    """Wrap value in Json for psycopg if not None."""
    return Json(value) if value is not None else None


def _ensure_rule_extractions_provenance_schema(db: NeonConnection) -> None:
    """Best-effort migration for older rule_extractions tables.

    This keeps ingestion robust when environments have an existing table that
    predates provenance/lifecycle columns (kb_entry_id, review_status, etc.).
    """
    global _PROVENANCE_SCHEMA_ENSURED
    if _PROVENANCE_SCHEMA_ENSURED:
        return

    for column_name, column_type in _PROVENANCE_MIGRATION_COLUMNS.items():
        db.execute_commit(
            f'ALTER TABLE rule_extractions ADD COLUMN IF NOT EXISTS {column_name} {column_type};',
        )

    _PROVENANCE_SCHEMA_ENSURED = True


def save_extractions(
    db: NeonConnection,
    rules: list[dict],
    *,
    batch_id: str,
    agent_name: str = 'athena',
    raw_text: str = '',
    provenance: dict | None = None,
) -> list[str]:
    """INSERT extracted rules into rule_extractions. Returns list of generated IDs.

    ``raw_text`` is the original input text that produced these extractions.
    ``raw_text_hash`` is computed automatically from ``raw_text``.

    ``provenance`` is an optional dict of provenance fields that apply to all
    rules in this batch (kb_entry_id, learning_id, source_dataset, etc.).
    Per-rule provenance can also be set directly on the rule dict.
    """
    _ensure_rule_extractions_provenance_schema(db)
    raw_text_hash = compute_text_hash(raw_text) if raw_text else None
    prov = provenance or {}
    ids: list[str] = []

    for rule in rules:
        rule_id = str(uuid.uuid4())
        ids.append(rule_id)

        # Merge provenance: rule-level overrides batch-level
        merged_prov = {**prov, **{k: v for k, v in rule.items() if k in _PROVENANCE_COLUMNS and v is not None}}

        # Validate enum fields
        if 'review_status' in merged_prov:
            ReviewStatus(merged_prov['review_status'])
        if 'proposal_kind' in merged_prov:
            ProposalKind(merged_prov['proposal_kind'])
        if 'source_dataset' in merged_prov:
            SourceDataset(merged_prov['source_dataset'])
        if 'source_category' in merged_prov:
            SourceCategory(merged_prov['source_category'])

        columns = list(_ALL_COLUMNS)
        placeholders = ['%s'] * len(columns)

        values = [
            # Core fields
            rule_id,
            batch_id,
            agent_name,
            raw_text or None,
            raw_text_hash,
            rule.get('risk_factor', 'Unknown'),
            rule.get('risk_category'),
            rule.get('rule_name', 'Unknown Rule'),
            rule.get('product_type'),
            rule.get('action'),
            rule.get('outcome_description'),
            _json_or_none(rule.get('mitigants')),
            rule.get('source'),
            rule.get('source_type'),
            rule.get('confidence'),
            _json_or_none(rule.get('threshold')),
            rule.get('threshold_type'),
            rule.get('historical_exceptions'),
            rule.get('decision_quality'),
            rule.get('compound_trigger'),
            _json_or_none(rule.get('data_fields')),
            # Provenance fields
            merged_prov.get('kb_entry_id'),
            merged_prov.get('learning_id'),
            merged_prov.get('source_dataset'),
            merged_prov.get('source_category'),
            merged_prov.get('proposal_kind'),
            merged_prov.get('approval_status'),
            merged_prov.get('slack_channel_id'),
            merged_prov.get('slack_thread_ts'),
            merged_prov.get('langfuse_trace_id'),
            merged_prov.get('extractor_version'),
            _json_or_none(merged_prov.get('confidence_factors')),
            merged_prov.get('evidence_snippet'),
            merged_prov.get('review_status', ReviewStatus.PENDING_REVIEW.value),
        ]

        col_str = ', '.join(columns)
        ph_str = ', '.join(placeholders)

        db.execute_commit(
            f'INSERT INTO rule_extractions ({col_str}) VALUES ({ph_str})',
            tuple(values),
        )

    logger.info('Saved %d extractions (batch_id=%s)', len(ids), batch_id)
    return ids


# ---------------------------------------------------------------------------
# Identity-based dedup
# ---------------------------------------------------------------------------

def find_existing_by_identity(
    db: NeonConnection,
    *,
    agent_name: str = 'athena',
    kb_entry_id: str | None = None,
    learning_id: str | None = None,
    slack_channel_id: str | None = None,
    slack_thread_ts: str | None = None,
    extractor_version: str | None = None,
) -> list[dict]:
    """Find existing (non-superseded) extractions for an identity key.

    Returns rows matching the identity key with ``superseded_by IS NULL``.
    """
    _ensure_rule_extractions_provenance_schema(db)
    if kb_entry_id:
        return db.fetch_all(
            """
            SELECT id, extractor_version FROM rule_extractions
             WHERE agent_name = %s AND kb_entry_id = %s AND superseded_by IS NULL
             ORDER BY created_at DESC
            """,
            (agent_name, kb_entry_id),
        )
    if learning_id:
        return db.fetch_all(
            """
            SELECT id, extractor_version FROM rule_extractions
             WHERE agent_name = %s AND learning_id = %s AND superseded_by IS NULL
             ORDER BY created_at DESC
            """,
            (agent_name, learning_id),
        )
    if slack_channel_id and slack_thread_ts:
        return db.fetch_all(
            """
            SELECT id, extractor_version FROM rule_extractions
             WHERE agent_name = %s
               AND slack_channel_id = %s
               AND slack_thread_ts = %s
               AND superseded_by IS NULL
             ORDER BY created_at DESC
            """,
            (agent_name, slack_channel_id, slack_thread_ts),
        )
    return []


def supersede_rows(db: NeonConnection, old_ids: list[str], new_id: str) -> None:
    """Mark old extraction rows as superseded by a new row."""
    for old_id in old_ids:
        db.execute_commit(
            'UPDATE rule_extractions SET superseded_by = %s WHERE id = %s',
            (new_id, old_id),
        )


# ---------------------------------------------------------------------------
# Hash-based dedup (legacy, secondary safety net)
# ---------------------------------------------------------------------------

def has_extractions_for_raw_text_hash(
    db: NeonConnection,
    *,
    agent_name: str,
    raw_text_hash: str,
) -> bool:
    """Return True if we've already stored extractions for this raw_text_hash.

    Note: many extracted rules can share the same raw_text_hash (one per rule),
    so this checks for existence of any row with the hash rather than enforcing
    uniqueness at the table level.
    """
    row = db.fetch_one(
        """
        SELECT 1
          FROM rule_extractions
         WHERE agent_name = %s
           AND raw_text_hash = %s
         LIMIT 1
        """,
        (agent_name, raw_text_hash),
    )
    return row is not None


# ---------------------------------------------------------------------------
# Status updates
# ---------------------------------------------------------------------------

def mark_ingested(db: NeonConnection, rule_id: str) -> None:
    """UPDATE ingestion_status='ingested', ingested_at=NOW()."""
    db.execute_commit(
        """
        UPDATE rule_extractions
           SET ingestion_status = 'ingested', ingested_at = NOW()
         WHERE id = %s
        """,
        (rule_id,),
    )


def mark_failed(db: NeonConnection, rule_id: str, error: str) -> None:
    """UPDATE ingestion_status='failed', ingestion_error=error."""
    db.execute_commit(
        """
        UPDATE rule_extractions
           SET ingestion_status = 'failed', ingestion_error = %s
         WHERE id = %s
        """,
        (error, rule_id),
    )


def mark_reviewed(
    db: NeonConnection,
    rule_id: str,
    *,
    review_status: ReviewStatus | str,
    reviewed_by: str,
) -> None:
    """Set review_status + reviewer info on an extraction."""
    status = ReviewStatus(review_status)  # validate and normalise
    db.execute_commit(
        """
        UPDATE rule_extractions
           SET review_status = %s, reviewed_by = %s, reviewed_at = NOW()
         WHERE id = %s
        """,
        (status.value, reviewed_by, rule_id),
    )


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_pending(
    db: NeonConnection,
    *,
    agent_name: str = 'athena',
    batch_id: str | None = None,
) -> list[dict]:
    """SELECT rules WHERE ingestion_status='pending'.

    Returns rule dicts ready for ``_rule_to_ingest_payload``.
    """
    if batch_id:
        return db.fetch_all(
            """
            SELECT * FROM rule_extractions
             WHERE ingestion_status = 'pending'
               AND agent_name = %s
               AND batch_id = %s
             ORDER BY created_at
            """,
            (agent_name, batch_id),
        )
    return db.fetch_all(
        """
        SELECT * FROM rule_extractions
         WHERE ingestion_status = 'pending'
           AND agent_name = %s
         ORDER BY created_at
        """,
        (agent_name,),
    )


def fetch_approved_pending_ingestion(
    db: NeonConnection,
    *,
    agent_name: str = 'athena',
    batch_id: str | None = None,
) -> list[dict]:
    """SELECT rules that are reviewed-approved but not yet ingested into the graph.

    Filters: review_status='approved' AND ingestion_status='pending' AND superseded_by IS NULL.
    """
    _ensure_rule_extractions_provenance_schema(db)
    base = """
        SELECT * FROM rule_extractions
         WHERE review_status = 'approved'
           AND ingestion_status = 'pending'
           AND superseded_by IS NULL
           AND agent_name = %s
    """
    params: list = [agent_name]
    if batch_id:
        base += ' AND batch_id = %s'
        params.append(batch_id)
    base += ' ORDER BY created_at'
    return db.fetch_all(base, tuple(params))


def fetch_all_extractions(
    db: NeonConnection,
    *,
    agent_name: str = 'athena',
    limit: int = 500,
) -> list[dict]:
    """SELECT all extractions for an agent, ordered by created_at DESC."""
    return db.fetch_all(
        """
        SELECT * FROM rule_extractions
         WHERE agent_name = %s
         ORDER BY created_at DESC
         LIMIT %s
        """,
        (agent_name, limit),
    )
