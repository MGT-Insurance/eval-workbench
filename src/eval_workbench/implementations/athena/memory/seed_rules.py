"""Manually curated seed rules for the Athena knowledge graph.

These are pre-defined referral trigger rules that can be loaded into
Neon and optionally ingested into Zep without running LLM extraction.

Usage::

    from eval_workbench.implementations.athena.memory.seed_rules import (
        REFERRAL_TRIGGER_RULES,
        seed_to_neon,
        seed_to_graph,
    )

    # Insert into Neon only
    with NeonConnection() as db:
        ids = seed_to_neon(db)

    # Insert into Neon + ingest into Zep graph
    with NeonConnection() as db:
        ids = seed_to_graph(db, store)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


SEED_RULES_PATH = Path(__file__).with_name('seed_rules.yaml')


def _load_seed_rules(path: Path = SEED_RULES_PATH) -> list[dict]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or []
    except FileNotFoundError:
        logger.warning('Seed rules file not found at %s', path)
        return []

    if not isinstance(data, list):
        raise ValueError(f'Seed rules YAML must be a list, got {type(data).__name__}')
    return data


REFERRAL_TRIGGER_RULES: list[dict] = _load_seed_rules()



def seed_to_neon(
    db,
    *,
    rules: list[dict] | None = None,
    agent_name: str = 'athena',
    source_label: str = 'manual_seed',
) -> list[str]:
    """Insert seed rules into rule_extractions table.

    Returns list of generated IDs. Uses ``source_label`` as the
    batch raw_text so seeded rules are identifiable.

    Parameters
    ----------
    db:
        NeonConnection instance.
    rules:
        Rules to seed. Defaults to ``REFERRAL_TRIGGER_RULES``.
    agent_name:
        Agent name for the extractions.
    source_label:
        Label stored in ``raw_text`` to identify seeded rules.
    """
    from eval_workbench.shared.memory.persistence import save_extractions

    rules = rules or REFERRAL_TRIGGER_RULES
    batch_id = str(uuid.uuid4())

    ids = save_extractions(
        db,
        rules,
        batch_id=batch_id,
        agent_name=agent_name,
        raw_text=source_label,
    )
    logger.info(
        'Seeded %d rules into Neon (batch_id=%s, source=%s)',
        len(ids), batch_id, source_label,
    )
    return ids


def seed_to_graph(
    db,
    store,
    *,
    rules: list[dict] | None = None,
    agent_name: str = 'athena',
    source_label: str = 'manual_seed',
) -> list[str]:
    """Insert seed rules into Neon AND ingest into Zep graph.

    Saves to Neon first, then ingests each rule into the graph store,
    updating ingestion status on success or failure.

    Returns list of generated IDs.
    """
    from eval_workbench.implementations.athena.memory.pipeline import (
        AthenaRulePipeline,
    )
    from eval_workbench.shared.memory.persistence import mark_failed, mark_ingested

    rules = rules or REFERRAL_TRIGGER_RULES
    ids = seed_to_neon(db, rules=rules, agent_name=agent_name, source_label=source_label)

    for i, rule in enumerate(rules):
        try:
            payload = AthenaRulePipeline._rule_to_ingest_payload(rule)
            store.ingest(payload)
            mark_ingested(db, ids[i])
        except Exception as exc:
            logger.warning('Failed to ingest seed rule %s: %s', rule.get('rule_name'), exc)
            mark_failed(db, ids[i], str(exc))

    logger.info('Seeded %d rules into Neon + Zep graph.', len(ids))
    return ids
