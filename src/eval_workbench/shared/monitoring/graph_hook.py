"""
Part B: Post-monitoring hook for graph ingestion proposals.

Processes monitoring results to create rule_extraction proposals from:
- Path 1: Graph hints from ProductAnalyzer (stored verbatim, no LLM call)
- Path 2: Rule extraction from intervention/escalation threads (LLM extraction)

All outputs are proposals (ingestion_status='pending') — NO graph writes happen here.
Graph ingestion happens later via run_from_db() after human review.

Gated behind GRAPH_INGESTION=true env var in monitoring_entrypoint.py.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import yaml

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


def _normalize(text: str) -> str:
    """Lowercase, replace _/- with spaces, collapse whitespace."""
    import re
    return re.sub(r'\s+', ' ', text.lower().replace('_', ' ').replace('-', ' ')).strip()


def _has_operational_markers(text: str, config: dict) -> bool:
    """Check if text contains operational markers, respecting UW allowlist."""
    import re
    normed = _normalize(text)
    markers = config.get('keywords', [])
    patterns = [re.compile(p, re.I) for p in config.get('patterns', [])]
    uw_allowlist = config.get('uw_allowlist', [])

    has_uw_term = any(term in normed for term in uw_allowlist)
    has_keyword = any(m in normed for m in markers)
    has_pattern = any(p.search(text) for p in patterns)

    if has_uw_term:
        return has_pattern
    return has_keyword or has_pattern


def _has_operational_leakage(rule: dict, config: dict) -> bool:
    """Check if extracted rule contains operational markers that leaked through."""
    scan_fields = ['risk_factor', 'rule_name', 'outcome_description',
                   'historical_exceptions', 'source']
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


# Hard gates for Path 2 extraction (from ingestion_config.yaml defaults)
_DEFAULT_EXCLUDED_INTERVENTION_TYPES = {'approval', 'support', 'clarification'}
_DEFAULT_EXCLUDED_FAILED_STEPS = {'data_integrity_failure', 'system_tooling_failure', 'chat_interface'}

# Correction keywords for evidence snippet extraction
_DEFAULT_CORRECTION_KEYWORDS = [
    'correct', 'actually', 'should be', 'override',
    'decline', 'approve', 'not right', 'change to',
]


def _load_config() -> dict:
    """Load ingestion config. Returns empty dict if not found."""
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parent.parent.parent
        / 'implementations'
        / 'athena'
        / 'config'
        / 'ingestion_config.yaml'
    )
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _extract_evidence_snippet(
    messages: list[dict],
    intervention_turn: int | None,
    correction_keywords: list[str] | None = None,
    context_window: int = 2,
) -> str:
    """Extract 1-3 messages around intervention marker as evidence snippet.

    Heuristic: messages containing correction keywords within ±context_window
    messages of the intervention point.
    """
    keywords = correction_keywords or _DEFAULT_CORRECTION_KEYWORDS
    if not messages or intervention_turn is None:
        return ''

    start = max(0, intervention_turn - context_window)
    end = min(len(messages), intervention_turn + context_window + 1)
    window = messages[start:end]

    evidence_parts = []
    for msg in window:
        content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
        if not content:
            continue
        content_lower = content.lower()
        if any(kw in content_lower for kw in keywords):
            # Truncate long messages
            snippet = content[:300] + '...' if len(content) > 300 else content
            evidence_parts.append(snippet)

    return ' | '.join(evidence_parts[:3])


def _passes_hard_gates(
    obj_signals: Any,
    fb_signals: Any | None,
    config: dict,
) -> bool:
    """Check if a thread passes hard gates for Path 2 extraction.

    All gates must pass:
    1. Thread has an Athena recommendation (resolution_status not skipped/no_recommendation)
    2. has_intervention OR is_escalated
    3. intervention_type NOT in excluded set
    4. failed_step NOT in excluded set
    """
    gate_config = config.get('slack_extraction_gates', {})
    excluded_interventions = set(gate_config.get(
        'excluded_intervention_types', _DEFAULT_EXCLUDED_INTERVENTION_TYPES
    ))
    excluded_failures = set(gate_config.get(
        'excluded_failed_steps', _DEFAULT_EXCLUDED_FAILED_STEPS
    ))

    # Gate 1: Must have recommendation
    resolution = getattr(obj_signals, 'resolution', None)
    if resolution:
        final_status = getattr(resolution, 'final_status', 'pending')
        # No recommendation = no rule to attribute
        if final_status in ('skipped_no_human', 'no_recommendation'):
            return False

    # Gate 2: Must have intervention or escalation
    intervention = getattr(obj_signals, 'intervention', None)
    escalation = getattr(obj_signals, 'escalation', None)
    has_intervention = getattr(intervention, 'has_intervention', False) if intervention else False
    is_escalated = getattr(escalation, 'is_escalated', False) if escalation else False
    if not has_intervention and not is_escalated:
        return False

    # Gate 3: intervention_type not in exclusion set
    int_type = getattr(intervention, 'intervention_type', 'no_intervention') if intervention else 'no_intervention'
    if int_type in excluded_interventions:
        return False

    # Gate 4: failed_step not in exclusion set (if feedback available)
    if fb_signals:
        failed_step = getattr(fb_signals, 'failed_step', None)
        if failed_step and failed_step in excluded_failures:
            return False

    return True


def _format_thread_for_extraction(
    messages: list,
    obj_signals: Any,
    subj_signals: Any | None,
    fb_signals: Any | None,
) -> str:
    """Format a Slack thread for rule extraction (Path 2).

    Enriches context with already-computed signals (NOT re-inferred).
    """
    intervention = getattr(obj_signals, 'intervention', None)
    escalation = getattr(obj_signals, 'escalation', None)

    parts = [
        'Source: Live Slack Conversation (Intervention Detected)',
        f'Channel: {getattr(obj_signals, "channel_id", "unknown")}',
        f'Thread: {getattr(obj_signals, "thread_id", "unknown")}',
    ]

    if intervention:
        parts.append(f'Intervention Type: {getattr(intervention, "intervention_type", "unknown")}')
    if subj_signals:
        parts.append(f'Override Type: {getattr(subj_signals, "override_type", "no_override")}')
        parts.append(f'Frustration Cause: {getattr(subj_signals, "frustration_cause", "none")}')
    if fb_signals:
        parts.append(
            f'Attribution: {getattr(fb_signals, "failed_step", "unknown")} '
            f'@ {getattr(fb_signals, "confidence", "low")}'
        )

    parts.append('')
    parts.append('--- Thread Conversation ---')

    # Format messages
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
        else:
            role = getattr(msg, 'role', 'unknown')
            content = getattr(msg, 'content', '')
        if content:
            parts.append(f'[Turn {i}] {role}: {content}')

    if intervention:
        summary = getattr(intervention, 'intervention_summary', '')
        details = getattr(intervention, 'issue_details', '')
        if summary or details:
            parts.append('')
            parts.append('--- Intervention Context ---')
            if summary:
                parts.append(summary)
            if details:
                parts.append(details)

    return '\n'.join(parts)


def process_graph_hints(
    results: Any,
    db: Any,
    config: dict | None = None,
) -> dict:
    """Path 1: Store graph_hints from ProductAnalyzer as proposal rows.

    Each graph_hint becomes a proposal_kind='graph_hint' row with thread evidence.
    No LLM extraction needed — the ProductAnalyzer already did the work.

    Returns stats dict.
    """
    config = config or _load_config()
    batch_id = str(uuid.uuid4())
    stats = {'hints_stored': 0, 'hints_skipped': 0}

    # results is an EvaluationResults — iterate through items
    if not hasattr(results, 'results') or not results.results:
        return stats

    for item_id, item_results in _iter_evaluation_items(results):
        composite = _get_composite_result(item_results)
        if not composite:
            continue

        prod_signals = _get_product_signals(composite)
        if not prod_signals:
            continue

        graph_hints = getattr(prod_signals, 'graph_hints', [])
        if not graph_hints:
            continue

        obj_signals = _get_objective_signals(composite)
        channel_id = getattr(obj_signals, 'channel_id', None) if obj_signals else None
        thread_ts = getattr(obj_signals, 'thread_id', None) if obj_signals else None

        for hint in graph_hints:
            hint_dict = hint.model_dump() if hasattr(hint, 'model_dump') else dict(hint)

            # Build a pseudo-rule from the hint
            rule = {
                'risk_factor': hint_dict.get('rule_name_hint', 'Unknown'),
                'rule_name': hint_dict.get('rule_name_hint', f'graph_hint_{hint_dict.get("target_node_type", "unknown")}'),
                'risk_category': hint_dict.get('target_node_type', 'other'),
                'action': hint_dict.get('suggested_action', 'review'),
                'outcome_description': f'Graph hint: {hint_dict.get("suggested_action", "review")} '
                                       f'{hint_dict.get("target_node_type", "unknown")}',
                'source': f'slack_thread:{channel_id}:{thread_ts}' if channel_id else 'slack_thread',
                'source_type': 'production',
                'confidence': 'medium',
            }

            # Build evidence from learnings
            learnings = getattr(prod_signals, 'learnings', [])
            categories = getattr(prod_signals, 'learning_categories', [])
            evidence_parts = []
            for i, learning in enumerate(learnings):
                cat = categories[i] if i < len(categories) else 'other'
                if cat in ('rules', 'guardrails'):
                    evidence_parts.append(learning)

            provenance = {
                'slack_channel_id': channel_id,
                'slack_thread_ts': thread_ts,
                'source_dataset': SourceDataset.SLACK_GRAPH_HINT.value,
                'source_category': SourceCategory.UW_RULE.value,
                'proposal_kind': ProposalKind.GRAPH_HINT.value,
                'confidence_factors': hint_dict,
                'evidence_snippet': ' | '.join(evidence_parts[:3]) if evidence_parts else '',
                'review_status': ReviewStatus.PENDING_REVIEW.value,
            }

            # Dedup: same thread + same hint name
            existing = find_existing_by_identity(
                db,
                slack_channel_id=channel_id,
                slack_thread_ts=thread_ts,
            )
            # For graph hints, skip if any exist for this thread (hints are idempotent)
            if existing:
                stats['hints_skipped'] += 1
                continue

            save_extractions(
                db,
                [rule],
                batch_id=batch_id,
                agent_name='athena',
                raw_text=f'graph_hint:{hint_dict}',
                provenance=provenance,
            )
            stats['hints_stored'] += 1

    return stats


def process_thread_extractions(
    results: Any,
    db: Any,
    config: dict | None = None,
) -> dict:
    """Path 2: Extract rules from intervention/escalation threads.

    Applies hard gates, formats thread with computed signals, runs LLM extraction.
    Stores as proposal_kind='extracted_rule' with ingestion_status='pending'.

    Returns stats dict.
    """
    from eval_workbench.implementations.athena.memory.extractors import (
        EXTRACTION_SYSTEM_PROMPT,
        RuleExtractor,
    )

    config = config or _load_config()
    op_config = config.get('operational_markers', {})
    evidence_config = config.get('evidence_snippet', {})
    correction_keywords = evidence_config.get('correction_keywords', _DEFAULT_CORRECTION_KEYWORDS)
    context_window = evidence_config.get('context_window', 2)

    extractor = RuleExtractor(model='gpt-4o')
    ext_version = compute_extractor_version(EXTRACTION_SYSTEM_PROMPT, 'gpt-4o')
    batch_id = str(uuid.uuid4())

    stats = {'extracted': 0, 'gate_filtered': 0, 'errors': 0, 'dedup_skipped': 0, 'operational_filtered': 0}

    if not hasattr(results, 'results') or not results.results:
        return stats

    for item_id, item_results in _iter_evaluation_items(results):
        composite = _get_composite_result(item_results)
        if not composite:
            continue

        obj_signals = _get_objective_signals(composite)
        subj_signals = _get_subjective_signals(composite)
        fb_signals = _get_feedback_signals(composite)

        if not obj_signals:
            stats['gate_filtered'] += 1
            continue

        # Hard gates
        if not _passes_hard_gates(obj_signals, fb_signals, config):
            stats['gate_filtered'] += 1
            continue

        channel_id = getattr(obj_signals, 'channel_id', None)
        thread_ts = getattr(obj_signals, 'thread_id', None)

        # Dedup
        existing = find_existing_by_identity(
            db,
            slack_channel_id=channel_id,
            slack_thread_ts=thread_ts,
            extractor_version=ext_version,
        )
        same_version = [e for e in existing if e.get('extractor_version') == ext_version]
        if same_version:
            stats['dedup_skipped'] += 1
            continue

        # Get messages from the dataset item
        messages = _get_messages_from_item(item_id, item_results)

        # Format for extraction
        text = _format_thread_for_extraction(messages, obj_signals, subj_signals, fb_signals)

        # Extract
        try:
            rules = extractor.extract_batch([text])
        except Exception as exc:
            logger.error('Extraction failed for thread %s/%s: %s', channel_id, thread_ts, exc)
            stats['errors'] += 1
            continue

        if not rules:
            continue

        # Post-extraction filter + metadata from computed signals
        intervention = getattr(obj_signals, 'intervention', None)
        int_turn = getattr(intervention, 'intervention_turn_index', None) if intervention else None

        evidence = _extract_evidence_snippet(
            [{'content': getattr(m, 'content', '') if not isinstance(m, dict) else m.get('content', '')}
             for m in messages],
            int_turn,
            correction_keywords,
            context_window,
        )

        for rule in rules:
            # Post-extraction operational filter
            if _has_operational_leakage(rule, op_config):
                rule['source_category'] = SourceCategory.OPERATIONAL_EXTRACTED.value
                stats['operational_filtered'] += 1

            # Process rule check: tag vague process rules
            if (rule.get('risk_category') == 'process'
                    and rule.get('action') == 'verify'
                    and not rule.get('threshold')):
                rule['source_category'] = SourceCategory.PROCESS_FROM_SLACK.value

            # Override metadata from computed signals (NOT re-inferred)
            if subj_signals:
                override_type = getattr(subj_signals, 'override_type', None)
                if override_type and override_type != 'no_override':
                    rule['decision_quality'] = 'divergent'
            if fb_signals:
                rule['source'] = f'attribution:{getattr(fb_signals, "failed_step", "unknown")}'

        provenance = {
            'slack_channel_id': channel_id,
            'slack_thread_ts': thread_ts,
            'source_dataset': SourceDataset.SLACK_THREAD.value,
            'source_category': SourceCategory.UW_RULE.value,
            'proposal_kind': ProposalKind.EXTRACTED_RULE.value,
            'extractor_version': ext_version,
            'evidence_snippet': evidence,
            'review_status': ReviewStatus.PENDING_REVIEW.value,
            'confidence_factors': {
                'source_quality': 'production',
                'triage_category': 'uw_rule',
                'extraction_specificity': 'medium',
                'attribution_confidence': getattr(fb_signals, 'confidence', 'low') if fb_signals else 'low',
            },
        }

        new_ids = save_extractions(
            db,
            rules,
            batch_id=batch_id,
            agent_name='athena',
            raw_text=text,
            provenance=provenance,
        )

        # Supersede older versions
        if existing and new_ids:
            old_ids = [e['id'] for e in existing]
            supersede_rows(db, old_ids, new_ids[0])

        stats['extracted'] += len(rules)

    return stats


# ============================================================================
# Result navigation helpers
# ============================================================================

def _iter_evaluation_items(results: Any):
    """Iterate over evaluation results, yielding (item_id, item_results) pairs."""
    # EvaluationResults stores results as list of MetricEvaluationResult
    # The structure depends on the axion framework
    if hasattr(results, 'items') and hasattr(results, 'results'):
        # Paired items + results
        items = results.items if hasattr(results, 'items') else []
        result_list = results.results if isinstance(results.results, list) else []
        for i, result in enumerate(result_list):
            item_id = items[i].id if i < len(items) else str(i)
            yield item_id, result
    elif hasattr(results, 'to_dataframe'):
        # Try dataframe approach
        try:
            df = results.to_dataframe()
            for idx, row in df.iterrows():
                yield str(idx), row
        except Exception:
            pass


def _get_composite_result(item_result: Any) -> Any | None:
    """Extract composite result from an item's evaluation results."""
    if hasattr(item_result, 'metadata') and item_result.metadata:
        return item_result
    if hasattr(item_result, 'signals'):
        return item_result
    return item_result


def _get_objective_signals(composite: Any) -> Any | None:
    """Extract objective signals from composite result."""
    if hasattr(composite, 'signals') and hasattr(composite.signals, 'objective'):
        return composite.signals.objective
    if hasattr(composite, 'metadata') and isinstance(composite.metadata, dict):
        obj = composite.metadata.get('objective')
        if obj and hasattr(obj, 'signals'):
            return obj.signals
    return None


def _get_subjective_signals(composite: Any) -> Any | None:
    """Extract subjective signals from composite result."""
    if hasattr(composite, 'signals') and hasattr(composite.signals, 'subjective'):
        return composite.signals.subjective
    if hasattr(composite, 'metadata') and isinstance(composite.metadata, dict):
        subj = composite.metadata.get('subjective')
        if subj and hasattr(subj, 'signals'):
            return subj.signals
    return None


def _get_feedback_signals(composite: Any) -> Any | None:
    """Extract feedback attribution signals from composite result."""
    if hasattr(composite, 'signals') and hasattr(composite.signals, 'feedback'):
        return composite.signals.feedback
    if hasattr(composite, 'metadata') and isinstance(composite.metadata, dict):
        fb = composite.metadata.get('feedback')
        if fb and hasattr(fb, 'signals'):
            return fb.signals
    return None


def _get_product_signals(composite: Any) -> Any | None:
    """Extract product signals from composite result."""
    if hasattr(composite, 'signals') and hasattr(composite.signals, 'product'):
        return composite.signals.product
    if hasattr(composite, 'metadata') and isinstance(composite.metadata, dict):
        prod = composite.metadata.get('product')
        if prod and hasattr(prod, 'signals'):
            return prod.signals
    return None


def _get_messages_from_item(item_id: str, item_result: Any) -> list:
    """Extract conversation messages from an evaluation item."""
    # Try to get from the dataset item if available
    if hasattr(item_result, 'item') and hasattr(item_result.item, 'conversation'):
        conv = item_result.item.conversation
        return conv.messages if hasattr(conv, 'messages') else []
    return []


# ============================================================================
# Main entrypoint
# ============================================================================

def post_process_for_graph(results: Any, db: Any) -> dict:
    """Main entrypoint called from monitoring_entrypoint.py after monitor.run_async().

    Runs both Path 1 (graph hints) and Path 2 (thread extraction).
    Returns combined stats.
    """
    config = _load_config()
    stats: dict = {}

    logger.info('Starting graph hook post-processing...')

    # Path 1: Graph hints (no LLM calls)
    try:
        hint_stats = process_graph_hints(results, db, config)
        stats['path1'] = hint_stats
        logger.info('Path 1 (graph hints): %s', hint_stats)
    except Exception as exc:
        logger.error('Path 1 failed: %s', exc)
        stats['path1_error'] = str(exc)

    # Path 2: Thread extraction (LLM calls)
    try:
        extraction_stats = process_thread_extractions(results, db, config)
        stats['path2'] = extraction_stats
        logger.info('Path 2 (thread extraction): %s', extraction_stats)
    except Exception as exc:
        logger.error('Path 2 failed: %s', exc)
        stats['path2_error'] = str(exc)

    logger.info('Graph hook post-processing complete: %s', stats)
    return stats
