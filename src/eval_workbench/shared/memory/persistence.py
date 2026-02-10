from __future__ import annotations

import hashlib
import logging
import uuid

from psycopg.types.json import Json

from eval_workbench.shared.database.neon import NeonConnection

logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """SHA-256 of input text for dedup."""
    return hashlib.sha256(text.encode()).hexdigest()


def save_extractions(
    db: NeonConnection,
    rules: list[dict],
    *,
    batch_id: str,
    agent_name: str = 'athena',
    raw_text: str = '',
) -> list[str]:
    """INSERT extracted rules into rule_extractions. Returns list of generated IDs.

    ``raw_text`` is the original input text that produced these extractions.
    ``raw_text_hash`` is computed automatically from ``raw_text``.
    """
    raw_text_hash = compute_text_hash(raw_text) if raw_text else None
    ids: list[str] = []

    for rule in rules:
        rule_id = str(uuid.uuid4())
        ids.append(rule_id)

        db.execute_commit(
            """
            INSERT INTO rule_extractions (
                id, batch_id, agent_name,
                raw_text, raw_text_hash,
                risk_factor, risk_category, rule_name, product_type,
                action, outcome_description,
                mitigants, source, source_type, confidence,
                threshold, threshold_type, historical_exceptions,
                decision_quality, compound_trigger, data_fields
            ) VALUES (
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            """,
            (
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
                Json(rule.get('mitigants'))
                if rule.get('mitigants') is not None
                else None,
                rule.get('source'),
                rule.get('source_type'),
                rule.get('confidence'),
                Json(rule.get('threshold'))
                if rule.get('threshold') is not None
                else None,
                rule.get('threshold_type'),
                rule.get('historical_exceptions'),
                rule.get('decision_quality'),
                rule.get('compound_trigger'),
                Json(rule.get('data_fields'))
                if rule.get('data_fields') is not None
                else None,
            ),
        )

    logger.info('Saved %d extractions (batch_id=%s)', len(ids), batch_id)
    return ids


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
