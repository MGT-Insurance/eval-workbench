from __future__ import annotations

import logging
import uuid
from pathlib import Path

from eval_workbench.implementations.athena.memory.extractors import RuleExtractor
from eval_workbench.shared.memory.ontology import OntologyDefinition
from eval_workbench.shared.memory.persistence import (
    fetch_all_extractions,
    fetch_pending,
    mark_failed,
    mark_ingested,
    save_extractions,
)
from eval_workbench.shared.memory.pipeline import BasePipeline, PipelineResult
from eval_workbench.shared.memory.store import BaseGraphStore, GraphIngestPayload

logger = logging.getLogger(__name__)


class AthenaRulePipeline(BasePipeline):
    """Ingestion pipeline for Athena underwriting rules.

    Combines a ``RuleExtractor`` (LLM-based) with a ``BaseGraphStore`` to
    extract rules from raw text and ingest them as graph edges.
    """

    def __init__(
        self,
        store: BaseGraphStore,
        extractor: RuleExtractor | None = None,
        extractor_model: str = 'gpt-4o',
        db=None,
    ) -> None:
        super().__init__(store=store)
        self.extractor = extractor or RuleExtractor(model=extractor_model)
        self.db = db  # Optional NeonConnection for persistence

    def extract(self, raw_data: str | list | dict, **kwargs) -> list[dict]:
        """Extract structured rules from raw data via the RuleExtractor."""
        if isinstance(raw_data, list):
            extracted: list[dict] = []
            for item in raw_data:
                extracted.extend(self.extractor.extract_batch([str(item)]))
            return extracted

        if isinstance(raw_data, str):
            texts = [raw_data]
        elif isinstance(raw_data, dict):
            texts = [str(v) for v in raw_data.values()]
        else:
            texts = [str(raw_data)]

        return self.extractor.extract_batch(texts)

    @staticmethod
    def _rule_to_ingest_payload(rule: dict) -> GraphIngestPayload:
        """Convert an extracted rule dict into a GraphIngestPayload.

        Builds edges following the Athena ontology:
        - RiskFactor --TRIGGERS--> Rule
        - Rule --RESULTS_IN--> Outcome
        - Mitigant --OVERRIDES--> Rule (for each mitigant)
        - Rule --DERIVED_FROM--> Source (if source is present)
        """
        risk_factor = rule.get('risk_factor', 'Unknown')
        rule_name = rule.get('rule_name', 'Unknown Rule')
        product_type = rule.get('product_type', 'ALL')
        action = rule.get('action', 'refer')
        outcome_desc = rule.get('outcome_description', '')
        mitigants = rule.get('mitigants', [])
        source = rule.get('source', '')
        source_type = rule.get('source_type', 'unknown')
        confidence = rule.get('confidence', 'medium')
        risk_category = rule.get('risk_category', '')
        threshold = rule.get('threshold')
        threshold_type = rule.get('threshold_type')
        historical_exceptions = rule.get('historical_exceptions')
        decision_quality = rule.get('decision_quality')
        compound_trigger = rule.get('compound_trigger')
        data_fields = rule.get('data_fields', [])

        edges: list[dict] = []

        # Build trigger properties — always include core fields, add threshold
        # metadata when present so the fact text stored in Zep is rich enough
        # for semantic search to surface threshold nuances.
        trigger_props: dict = {
            'confidence': confidence,
            'product_type': product_type,
        }
        if threshold_type:
            trigger_props['threshold_type'] = threshold_type
        if threshold:
            trigger_props['threshold'] = threshold
        if historical_exceptions:
            trigger_props['historical_exceptions'] = historical_exceptions
        if decision_quality:
            trigger_props['decision_quality'] = decision_quality
        if compound_trigger:
            trigger_props['compound_trigger'] = compound_trigger
        if data_fields:
            trigger_props['data_fields'] = data_fields

        # RiskFactor --TRIGGERS--> Rule
        edges.append(
            {
                'source': risk_factor,
                'source_type': 'RiskFactor',
                'source_properties': {
                    'category': risk_category,
                },
                'relation': 'TRIGGERS',
                'target': rule_name,
                'target_type': 'Rule',
                'target_properties': {
                    'product_type': product_type,
                    'action': action,
                    'threshold_type': threshold_type or '',
                },
                'properties': trigger_props,
            }
        )

        # Rule --RESULTS_IN--> Outcome
        # Include threshold context in the outcome description so it's stored
        # as part of the fact and retrievable via semantic search.
        full_outcome = outcome_desc
        if threshold and threshold_type:
            field = threshold.get('field', '') if isinstance(threshold, dict) else ''
            value = threshold.get('value', '') if isinstance(threshold, dict) else ''
            full_outcome += f' [threshold: {field} {value}, {threshold_type}]'
        if historical_exceptions:
            full_outcome += f' [historical: {historical_exceptions}]'

        outcome_name = f'{action.replace("_", " ").title()}'
        edges.append(
            {
                'source': rule_name,
                'source_type': 'Rule',
                'relation': 'RESULTS_IN',
                'target': outcome_name,
                'target_type': 'Outcome',
                'target_properties': {
                    'description': full_outcome,
                },
            }
        )

        # Mitigant --OVERRIDES--> Rule (for each mitigant)
        for mitigant in mitigants:
            if isinstance(mitigant, str) and mitigant.strip():
                edges.append(
                    {
                        'source': mitigant.strip(),
                        'source_type': 'Mitigant',
                        'relation': 'OVERRIDES',
                        'target': rule_name,
                        'target_type': 'Rule',
                    }
                )

        # Rule --DERIVED_FROM--> Source
        if source:
            edges.append(
                {
                    'source': rule_name,
                    'source_type': 'Rule',
                    'relation': 'DERIVED_FROM',
                    'target': source,
                    'target_type': 'Source',
                    'target_properties': {
                        'type': source_type,
                    },
                }
            )

        return GraphIngestPayload(edges=edges)

    def run(
        self,
        raw_data: str | list | dict,
        *,
        batch_size: int = 5,
        **kwargs,
    ) -> PipelineResult:
        """Run the full extract-and-ingest pipeline.

        Overrides the base ``run`` to use ``_rule_to_ingest_payload`` for
        each extracted rule, building proper graph edges per the Athena ontology.

        When a ``db`` (NeonConnection) is attached, extracted rules are persisted
        to the ``rule_extractions`` table before ingestion into Zep, and each
        rule's ingestion status is tracked.
        """
        result = PipelineResult()

        rules = self.extract(raw_data, **kwargs)
        result.items_processed = len(rules)

        # Persist extractions to Neon if db is available
        rule_ids: dict[int, str] = {}
        if self.db:
            batch_id = str(uuid.uuid4())
            ids = save_extractions(
                self.db,
                rules,
                batch_id=batch_id,
                agent_name='athena',
                raw_text=str(raw_data),
            )
            rule_ids = dict(zip(range(len(rules)), ids))

        for i in range(0, len(rules), batch_size):
            batch = rules[i : i + batch_size]
            for j, rule in enumerate(batch):
                idx = i + j
                try:
                    payload = self._rule_to_ingest_payload(rule)
                    self.store.ingest(payload)
                    result.items_ingested += 1
                    if self.db and idx in rule_ids:
                        mark_ingested(self.db, rule_ids[idx])
                except Exception as exc:
                    result.items_failed += 1
                    result.errors.append(f'{rule.get("rule_name", "?")}: {exc}')
                    logger.warning('Failed to ingest rule: %s', exc)
                    if self.db and idx in rule_ids:
                        mark_failed(self.db, rule_ids[idx], str(exc))

        return result

    def run_from_db(
        self,
        *,
        batch_id: str | None = None,
        batch_size: int = 5,
        force: bool = False,
    ) -> PipelineResult:
        """Re-ingest rules from Neon into the graph store (no LLM extraction).

        By default loads only rules with ``ingestion_status='pending'``.
        Pass ``force=True`` to reload ALL rules regardless of status
        (useful when rebuilding a graph backend from scratch).
        """
        if not self.db:
            raise ValueError('NeonConnection required for run_from_db')

        result = PipelineResult()
        if force:
            rules = fetch_all_extractions(self.db, agent_name='athena', limit=10000)
        else:
            rules = fetch_pending(self.db, agent_name='athena', batch_id=batch_id)
        result.items_processed = len(rules)

        for i in range(0, len(rules), batch_size):
            batch = rules[i : i + batch_size]
            for rule in batch:
                rule_id = rule['id']
                try:
                    payload = self._rule_to_ingest_payload(rule)
                    self.store.ingest(payload)
                    result.items_ingested += 1
                    mark_ingested(self.db, rule_id)
                except Exception as exc:
                    result.items_failed += 1
                    result.errors.append(f'{rule.get("rule_name", "?")}: {exc}')
                    logger.warning('Failed to ingest rule from db: %s', exc)
                    mark_failed(self.db, rule_id, str(exc))

        return result

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        ontology: OntologyDefinition | None = None,
        db=None,
    ) -> AthenaRulePipeline:
        """Create an AthenaRulePipeline from a YAML config file.

        Expected YAML structure::

            memory:
              backend: "falkor"  # or "zep"
              zep:
                api_key: "${ZEP_API_KEY}"
                ...
              falkor:
                host: "localhost"
                port: 6379
                graph_name_template: "{agent_name}_rules"
              extractor:
                model: "gpt-4o"
              pipeline:
                batch_size: 5

        Pass an optional ``db`` (NeonConnection) to enable persistence.
        """
        from eval_workbench.shared.config import load_config

        cfg = load_config(path)
        memory_cfg = cfg.get('memory', {})
        extractor_model = memory_cfg.get('extractor', {}).get('model', 'gpt-4o')
        backend = memory_cfg.get('backend', 'zep')

        if backend == 'falkor':
            from eval_workbench.shared.memory.falkor import FalkorGraphStore

            store = FalkorGraphStore.from_yaml(
                path,
                agent_name='athena',
                ontology=ontology,
            )
        else:
            from eval_workbench.shared.memory.zep import ZepGraphStore

            store = ZepGraphStore.from_yaml(
                path,
                agent_name='athena',
                ontology=ontology,
            )

        extractor = RuleExtractor(model=extractor_model)
        return cls(store=store, extractor=extractor, db=db)
