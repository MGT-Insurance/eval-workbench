#!/usr/bin/env python3
"""
Part A: Batch ingestion of knowledge base entries and agent learnings.

Reads from Neon DB tables (production default) or CSV dumps (fallback).
Extracts underwriting rules from historical retrospectives and human feedback,
creating proposal rows in rule_extractions (ingestion_status='pending').
Graph ingestion happens later via run_from_db() after human review.

Usage:
  # Preview what would be ingested (no LLM calls) — reads from DB
  python scripts/ingest_knowledge_sources.py --preview

  # Dry run: extract first 5 per category, inspect quality
  python scripts/ingest_knowledge_sources.py --dry-run

  # Full extraction run
  python scripts/ingest_knowledge_sources.py --run

  # Read from CSV dumps instead of DB
  python scripts/ingest_knowledge_sources.py --run --from-csv
  python scripts/ingest_knowledge_sources.py --run --from-csv \
    --kb-csv data/knowledge_base_entries.csv \
    --learnings-csv data/agent_learnings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

from eval_workbench.shared.database.neon import NeonConnection
from eval_workbench.shared.memory.enums import (
    ProposalKind,
    ReviewStatus,
    SourceCategory,
    SourceDataset,
)
from eval_workbench.shared.memory.persistence import (
    compute_extractor_version,
    find_existing_by_identity,
    save_extractions,
    supersede_rows,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

_DEFAULT_CONFIG_PATH = (
    _PROJECT_ROOT
    / 'src'
    / 'eval_workbench'
    / 'implementations'
    / 'athena'
    / 'config'
    / 'ingestion_config.yaml'
)


def load_ingestion_config(path: Path | None = None) -> dict:
    """Load operational markers, UW allowlist, patterns from YAML."""
    path = path or _DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Operational marker filtering
# ============================================================================


def _normalize(text: str) -> str:
    """Lowercase, replace _/- with spaces, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().replace('_', ' ').replace('-', ' ')).strip()


def _has_operational_markers(text: str, config: dict) -> bool:
    """Check if text contains operational markers, respecting UW allowlist.

    If text contains a UW adjustment term, only filter if a strong regex pattern matches.
    """
    normed = _normalize(text)
    markers = config.get('keywords', [])
    patterns = [re.compile(p, re.I) for p in config.get('patterns', [])]
    uw_allowlist = config.get('uw_allowlist', [])

    has_uw_term = any(term in normed for term in uw_allowlist)
    has_keyword = any(m in normed for m in markers)
    has_pattern = any(p.search(text) for p in patterns)

    # If text contains UW adjustment terms (ACV, deductible, etc.),
    # only filter if a strong regex pattern matches — keyword alone is insufficient
    if has_uw_term:
        return has_pattern  # Strong match only
    return has_keyword or has_pattern


# ============================================================================
# Triage
# ============================================================================


def triage_entry(
    kb_content: str,
    exec_summary: str,
    retro: dict,
    op_config: dict,
    uncertain_alignments: list[str] | None = None,
) -> str:
    """Classify an entry into uw_rule, data_quality, operational, mixed, or triage_uncertain.

    Returns a SourceCategory value string.
    """
    uncertain_alignments = uncertain_alignments or ['partial', 'divergent']
    combined_text = f'{kb_content} {exec_summary}'

    has_op = _has_operational_markers(combined_text, op_config)

    # Check for data quality patterns
    normed = _normalize(combined_text)
    data_quality_markers = [
        'magic dust',
        'discrepancy',
        'data mismatch',
        'exposure_value_units',
    ]
    has_dq = any(m in normed for m in data_quality_markers)

    # Check for UW content
    uw_allowlist = op_config.get('uw_allowlist', [])
    has_uw = any(term in normed for term in uw_allowlist)

    # Get retrospective signals for borderline handling
    dq = retro.get('decisionQuality', {})
    alignment = dq.get('alignment', '')
    insights = retro.get('classOfBusinessInsights', {})
    key_learnings = insights.get('keyLearnings', [])

    if has_op and has_uw:
        return SourceCategory.MIXED.value
    if has_op and not has_uw:
        # Borderline: keyword-only operational, but uncertain alignment + sparse learnings
        if alignment in uncertain_alignments and len(key_learnings) <= 1:
            return SourceCategory.TRIAGE_UNCERTAIN.value
        return SourceCategory.OPERATIONAL.value
    if has_dq:
        return SourceCategory.DATA_QUALITY.value
    return SourceCategory.UW_RULE.value


# ============================================================================
# Extractable input builder
# ============================================================================


def extractable_input(kb_content: str, retro: dict, config: dict) -> str:
    """Build extraction input from KB content + retrospective fields.

    Only includes fields that should produce UW rules.
    Applies per-bullet operational marker filtering with UW allowlist protection.
    """
    parts = [kb_content]

    # Include executiveSummary ONLY if it passes operational-marker check
    exec_summary = retro.get('executiveSummary', '')
    if exec_summary and not _has_operational_markers(exec_summary, config):
        parts.append(exec_summary)

    # Per-bullet filtering for keyLearnings and futureRecommendations
    insights = retro.get('classOfBusinessInsights', {})
    for learning in insights.get('keyLearnings', []):
        if not _has_operational_markers(learning, config):
            parts.append(f'Key Learning: {learning}')
    for rec in insights.get('futureRecommendations', []):
        if not _has_operational_markers(rec, config):
            parts.append(f'Future Recommendation: {rec}')

    return '\n\n'.join(parts)


def format_human_feedback(kb_content: str, metadata: dict) -> str:
    """Format human feedback entry for extraction."""
    parts = [kb_content]
    submitter = metadata.get('submittedBy', '')
    if submitter:
        parts.append(f'Submitted by: {submitter}')
    product_type = metadata.get('productType', '')
    if product_type:
        parts.append(f'Product Type: {product_type}')
    return '\n\n'.join(parts)


# ============================================================================
# Post-extraction operational filter
# ============================================================================


def has_operational_leakage(rule: dict, config: dict) -> bool:
    """Check if extracted rule contains operational markers that leaked through."""
    scan_fields = [
        'risk_factor',
        'rule_name',
        'outcome_description',
        'historical_exceptions',
        'source',
    ]
    text_parts = []
    for field in scan_fields:
        val = rule.get(field, '')
        if val:
            text_parts.append(str(val))
    data_fields = rule.get('data_fields', [])
    if data_fields:
        text_parts.append(str(data_fields))
    combined = ' '.join(text_parts)
    return _has_operational_markers(combined, config)


# ============================================================================
# Confidence model
# ============================================================================


def compute_confidence(
    source_quality: str,
    triage_category: str,
    rule: dict,
) -> tuple[str, dict]:
    """Compute confidence level and factors for an extracted rule.

    Returns (confidence_level, confidence_factors_dict).
    """
    has_threshold = bool(rule.get('threshold'))
    has_action = bool(rule.get('action'))
    has_risk_factor = bool(rule.get('risk_factor') and rule['risk_factor'] != 'Unknown')

    # Determine extraction_specificity
    if has_threshold and has_action and has_risk_factor:
        extraction_specificity = 'high'
    elif has_action and has_risk_factor:
        extraction_specificity = 'medium'
    else:
        extraction_specificity = 'low'

    factors = {
        'source_quality': source_quality,
        'triage_category': triage_category,
        'extraction_specificity': extraction_specificity,
        'has_threshold': has_threshold,
        'has_action': has_action,
        'has_risk_factor': has_risk_factor,
    }

    # Confidence rules
    low_sources = {'production_partial', 'pending', 'operational'}
    low_categories = {'mixed'}

    if (
        source_quality in low_sources
        or triage_category in low_categories
        or extraction_specificity == 'low'
    ):
        return 'low', factors
    if (
        source_quality == 'approved_sme'
        and triage_category == 'uw_rule'
        and extraction_specificity == 'high'
    ):
        return 'high', factors
    return 'medium', factors


# ============================================================================
# Data loading — DB (production) and CSV (fallback)
# ============================================================================


def load_kb_from_db(db: NeonConnection) -> list[dict]:
    """Load knowledge_base_entries from Neon DB."""
    return db.fetch_all(
        """
        SELECT id, agent_id, source_learning_id, title, content,
               metadata_json, status, approved_by_user, approved_at,
               usage_count, last_used_at, is_test, created_at, updated_at
          FROM knowledge_base_entries
         ORDER BY created_at
        """
    )


def load_learnings_from_db(db: NeonConnection) -> list[dict]:
    """Load agent_learnings from Neon DB."""
    return db.fetch_all(
        """
        SELECT id, agent_id, source_type, source_id, content_json,
               is_test, created_at, created_by_workflow
          FROM agent_learnings
         ORDER BY created_at
        """
    )


def load_csv(path: str | Path) -> list[dict]:
    """Load a CSV file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'CSV not found: {path}')
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_json_parse(value) -> dict | list | None:
    """Parse a JSON string or pass through an already-parsed value.

    When loading from DB (JSONB columns), psycopg returns dicts directly.
    When loading from CSV, the values are JSON strings that need parsing.
    """
    if isinstance(value, (dict, list)):
        return value  # Already parsed (JSONB column from DB)
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


# ============================================================================
# Ingestion index (preview)
# ============================================================================


def build_ingestion_index(
    kb_rows: list[dict],
    learning_rows: list[dict],
    op_config: dict,
    triage_config: dict,
) -> dict:
    """Build a preview of what would be ingested, skipped, and triaged."""
    # Index learnings by ID for lookup
    learnings_by_id = {}
    for row in learning_rows:
        lid = row.get('id', '')
        if lid:
            learnings_by_id[lid] = row

    triage_counts: Counter = Counter()
    entries_by_category: dict[str, list] = defaultdict(list)
    skipped: list[dict] = []
    uncertain_alignments = triage_config.get(
        'uncertain_alignments', ['partial', 'divergent']
    )

    for kb in kb_rows:
        kb_id = kb.get('id', '')
        metadata = safe_json_parse(kb.get('metadata_json', '')) or {}
        content = kb.get('content', '')
        status = kb.get('status', '')
        source_type_raw = metadata.get('sourceType', '')
        # source_learning_id is a top-level DB column; CSV stores it in metadata_json too
        source_learning_id = kb.get('source_learning_id', '') or metadata.get(
            'source_learning_id', ''
        )

        # Skip rejected entries
        if status == 'rejected':
            skipped.append({'id': kb_id, 'reason': 'rejected', 'status': status})
            continue

        # Determine entry type
        is_human_feedback = (
            source_type_raw in ('chat_feedback', 'feedback') and not source_learning_id
        )
        is_retrospective = bool(source_learning_id)

        if is_human_feedback:
            category = 'human_feedback'
            triage_counts[category] += 1
            entries_by_category[category].append(
                {
                    'kb_id': kb_id,
                    'source_type': 'sme',
                    'approval_status': 'approved'
                    if status == 'approved'
                    else 'pending',
                }
            )
            continue

        if is_retrospective:
            # Look up the agent_learnings row for retrospective data
            learning = learnings_by_id.get(source_learning_id, {})
            content_json = safe_json_parse(learning.get('content_json', '')) or {}
            retro = content_json.get('retrospective', {})
            agent_name = content_json.get('agentName', 'athena')
            model = content_json.get('model', '')

            # Skip error retrospectives
            if model == 'error':
                skipped.append(
                    {
                        'id': kb_id,
                        'learning_id': source_learning_id,
                        'reason': 'error_model',
                    }
                )
                continue

            # Skip non-athena agents (separate pipeline)
            if agent_name.lower() != 'athena':
                skipped.append(
                    {
                        'id': kb_id,
                        'learning_id': source_learning_id,
                        'reason': f'agent_{agent_name}',
                    }
                )
                continue

            exec_summary = retro.get('executiveSummary', '')
            triage_result = triage_entry(
                content, exec_summary, retro, op_config, uncertain_alignments
            )
            triage_counts[triage_result] += 1

            dq = retro.get('decisionQuality', {})
            entries_by_category[triage_result].append(
                {
                    'kb_id': kb_id,
                    'learning_id': source_learning_id,
                    'alignment': dq.get('alignment', ''),
                    'product_type': content_json.get('productType', ''),
                }
            )
            continue

        # Uncategorized
        skipped.append({'id': kb_id, 'reason': 'uncategorized'})

    return {
        'triage_counts': dict(triage_counts),
        'entries_by_category': {k: v for k, v in entries_by_category.items()},
        'skipped': skipped,
        'total_entries': len(kb_rows),
        'total_extractable': sum(
            len(v)
            for k, v in entries_by_category.items()
            if k not in (SourceCategory.OPERATIONAL.value,)
        ),
        'estimated_llm_calls': sum(
            len(v)
            for k, v in entries_by_category.items()
            if k not in (SourceCategory.OPERATIONAL.value,)
        ),
    }


# ============================================================================
# Main extraction pipeline
# ============================================================================


def run_extraction(
    kb_rows: list[dict],
    learning_rows: list[dict],
    ingestion_config: dict,
    db: NeonConnection,
    *,
    dry_run: bool = False,
    dry_run_limit: int = 5,
) -> dict:
    """Run the full extraction pipeline.

    Returns summary stats.
    """
    from eval_workbench.implementations.athena.memory.extractors import (
        EXTRACTION_SYSTEM_PROMPT,
        RuleExtractor,
    )

    op_config = ingestion_config.get('operational_markers', {})
    triage_config = ingestion_config.get('triage', {})
    priority_order = ingestion_config.get(
        'extraction_priority', ['divergent', 'partial', 'aligned', 'pending']
    )
    uncertain_alignments = triage_config.get(
        'uncertain_alignments', ['partial', 'divergent']
    )

    extractor = RuleExtractor(model='gpt-4o')
    ext_version = compute_extractor_version(EXTRACTION_SYSTEM_PROMPT, 'gpt-4o')
    batch_id = str(uuid.uuid4())

    # Index learnings
    learnings_by_id = {}
    for row in learning_rows:
        lid = row.get('id', '')
        if lid:
            learnings_by_id[lid] = row

    # Build sorted work items
    work_items: list[dict] = []

    for kb in kb_rows:
        kb_id = kb.get('id', '')
        metadata = safe_json_parse(kb.get('metadata_json', '')) or {}
        content = kb.get('content', '')
        status = kb.get('status', '')
        source_type_raw = metadata.get('sourceType', '')
        # source_learning_id is a top-level DB column; CSV stores it in metadata_json too
        source_learning_id = kb.get('source_learning_id', '') or metadata.get(
            'source_learning_id', ''
        )

        if status == 'rejected':
            continue

        is_human_feedback = (
            source_type_raw in ('chat_feedback', 'feedback') and not source_learning_id
        )
        is_retrospective = bool(source_learning_id)

        if is_human_feedback:
            text = format_human_feedback(content, metadata)
            work_items.append(
                {
                    'kb_id': kb_id,
                    'text': text,
                    'source_dataset': SourceDataset.KB_FEEDBACK.value,
                    'source_type': 'sme',
                    'source_category': SourceCategory.UW_RULE.value,
                    'source_quality': 'approved_sme'
                    if status == 'approved'
                    else 'pending',
                    'approval_status': 'approved'
                    if status == 'approved'
                    else 'pending',
                    'product_type': metadata.get('productType', ''),
                    'alignment': '',
                    'sort_key': (-1, 0),  # Feedback first
                }
            )
            continue

        if is_retrospective:
            learning = learnings_by_id.get(source_learning_id, {})
            content_json = safe_json_parse(learning.get('content_json', '')) or {}
            retro = content_json.get('retrospective', {})
            agent_name = content_json.get('agentName', 'athena')
            model = content_json.get('model', '')

            if model == 'error' or agent_name.lower() != 'athena':
                continue

            exec_summary = retro.get('executiveSummary', '')
            triage_result = triage_entry(
                content, exec_summary, retro, op_config, uncertain_alignments
            )

            # Skip pure operational
            if triage_result == SourceCategory.OPERATIONAL.value:
                continue

            dq = retro.get('decisionQuality', {})
            alignment = dq.get('alignment', '')
            product_type = content_json.get('productType', '')

            text = extractable_input(content, retro, op_config)

            # If all extractable content was filtered, reclassify as operational
            if text.strip() == content.strip() or not text.strip():
                continue

            source_quality = (
                f'production_{alignment}' if alignment else 'production_partial'
            )
            priority_idx = (
                priority_order.index(alignment)
                if alignment in priority_order
                else len(priority_order)
            )

            work_items.append(
                {
                    'kb_id': kb_id,
                    'learning_id': source_learning_id,
                    'text': text,
                    'source_dataset': SourceDataset.RETROSPECTIVE.value,
                    'source_type': 'production',
                    'source_category': triage_result,
                    'source_quality': source_quality,
                    'approval_status': status,
                    'product_type': product_type,
                    'alignment': alignment,
                    'sort_key': (priority_idx, 0),
                }
            )

    # Sort by priority
    work_items.sort(key=lambda x: x['sort_key'])

    if dry_run:
        # Limit per category
        by_cat: dict[str, list] = defaultdict(list)
        for item in work_items:
            cat = item['source_category']
            if len(by_cat[cat]) < dry_run_limit:
                by_cat[cat].append(item)
        work_items = [item for items in by_cat.values() for item in items]

    stats = Counter()
    total = len(work_items)

    for idx, item in enumerate(work_items):
        kb_id = item['kb_id']
        learning_id = item.get('learning_id')
        text = item['text']

        logger.info(
            '[%d/%d] Extracting kb_id=%s learning_id=%s category=%s',
            idx + 1,
            total,
            kb_id,
            learning_id,
            item['source_category'],
        )

        # Identity-based dedup
        existing = find_existing_by_identity(
            db,
            kb_entry_id=kb_id
            if item['source_dataset'] == SourceDataset.KB_FEEDBACK.value
            else None,
            learning_id=learning_id,
            extractor_version=ext_version,
        )
        same_version = [
            e for e in existing if e.get('extractor_version') == ext_version
        ]
        if same_version:
            logger.info('Skipping (same extractor version): kb_id=%s', kb_id)
            stats['skipped_dedup'] += 1
            continue

        # Extract
        try:
            rules = extractor.extract_batch([text])
        except Exception as exc:
            logger.error('Extraction failed for kb_id=%s: %s', kb_id, exc)
            stats['extraction_errors'] += 1
            continue

        if not rules:
            stats['no_rules_extracted'] += 1
            continue

        # Post-extraction filter + metadata override
        filtered_rules = []
        for rule in rules:
            # Post-extraction operational filter
            if has_operational_leakage(rule, op_config):
                rule['source_category'] = SourceCategory.OPERATIONAL_EXTRACTED.value
                stats['operational_extracted'] += 1
            else:
                stats['rules_extracted'] += 1

            # Metadata override from CSV ground truth
            rule['source_type'] = item['source_type']
            rule['decision_quality'] = item.get('alignment') or rule.get(
                'decision_quality'
            )
            if item.get('product_type'):
                rule['product_type'] = item['product_type']

            # Confidence model
            confidence, confidence_factors = compute_confidence(
                source_quality=item['source_quality'],
                triage_category=item['source_category'],
                rule=rule,
            )
            rule['confidence'] = confidence

            filtered_rules.append(rule)

        # Build provenance
        provenance = {
            'kb_entry_id': kb_id,
            'learning_id': learning_id,
            'source_dataset': item['source_dataset'],
            'source_category': item['source_category'],
            'proposal_kind': ProposalKind.EXTRACTED_RULE.value,
            'approval_status': item['approval_status'],
            'extractor_version': ext_version,
            'confidence_factors': confidence_factors,
            'review_status': ReviewStatus.PENDING_REVIEW.value,
        }

        # Save extractions
        new_ids = save_extractions(
            db,
            filtered_rules,
            batch_id=batch_id,
            agent_name='athena',
            raw_text=text,
            provenance=provenance,
        )

        # Supersede old versions
        if existing and new_ids:
            old_ids = [e['id'] for e in existing]
            supersede_rows(db, old_ids, new_ids[0])
            stats['superseded'] += len(old_ids)

    stats['total_processed'] = total
    stats['batch_id'] = batch_id
    return dict(stats)


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Batch ingestion of knowledge sources for knowledge graph'
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--preview', action='store_true', help='Preview ingestion index (no LLM calls)'
    )
    mode.add_argument(
        '--dry-run', action='store_true', help='Extract first 5 per category'
    )
    mode.add_argument('--run', action='store_true', help='Full extraction run')

    parser.add_argument(
        '--from-csv',
        action='store_true',
        help='Read from CSV dumps instead of Neon DB tables (default: read from DB)',
    )
    parser.add_argument(
        '--kb-csv',
        type=str,
        default=None,
        help='Path to knowledge_base_entries.csv (implies --from-csv)',
    )
    parser.add_argument(
        '--learnings-csv',
        type=str,
        default=None,
        help='Path to agent_learnings.csv (implies --from-csv)',
    )
    parser.add_argument(
        '--config', type=str, default=None, help='Path to ingestion_config.yaml'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    ingestion_config = load_ingestion_config(Path(args.config) if args.config else None)
    op_config = ingestion_config.get('operational_markers', {})
    triage_config = ingestion_config.get('triage', {})

    # Explicit CSV paths imply --from-csv
    use_csv = args.from_csv or args.kb_csv is not None or args.learnings_csv is not None

    with NeonConnection() as db:
        if use_csv:
            data_dir = _PROJECT_ROOT / 'data'
            kb_path = (
                Path(args.kb_csv)
                if args.kb_csv
                else data_dir / 'knowledge_base_entries.csv'
            )
            learnings_path = (
                Path(args.learnings_csv)
                if args.learnings_csv
                else data_dir / 'agent_learnings (1).csv'
            )
            logger.info(
                'Loading from CSV: kb=%s, learnings=%s', kb_path, learnings_path
            )
            kb_rows = load_csv(kb_path)
            learning_rows = load_csv(learnings_path)
        else:
            logger.info('Loading from Neon DB tables...')
            kb_rows = load_kb_from_db(db)
            learning_rows = load_learnings_from_db(db)

        logger.info(
            'Loaded %d KB entries, %d learnings', len(kb_rows), len(learning_rows)
        )

        if args.preview:
            index = build_ingestion_index(
                kb_rows, learning_rows, op_config, triage_config
            )
            print('\n=== INGESTION PREVIEW ===')
            print(f'Source: {"CSV" if use_csv else "Neon DB"}')
            print(f'Total KB entries: {index["total_entries"]}')
            print(f'Total extractable: {index["total_extractable"]}')
            print(f'Estimated LLM calls: {index["estimated_llm_calls"]}')
            print('\nTriage counts:')
            for cat, count in sorted(index['triage_counts'].items()):
                print(f'  {cat}: {count}')
            print(f'\nSkipped: {len(index["skipped"])}')
            for s in index['skipped'][:10]:
                print(f'  {s["id"]}: {s["reason"]}')
            if len(index['skipped']) > 10:
                print(f'  ... and {len(index["skipped"]) - 10} more')
            return 0

        stats = run_extraction(
            kb_rows,
            learning_rows,
            ingestion_config,
            db,
            dry_run=args.dry_run,
        )

    print('\n=== EXTRACTION RESULTS ===')
    for key, value in sorted(stats.items()):
        print(f'  {key}: {value}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
