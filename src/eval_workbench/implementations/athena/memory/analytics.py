from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Protocol, TypeGuard

from eval_workbench.shared.memory.analytics import BaseGraphAnalytics
from eval_workbench.shared.memory.store import BaseGraphStore, GraphSearchResult

logger = logging.getLogger(__name__)


class _CypherStore(Protocol):
    def cypher(self, query: str, params: dict | None = None) -> Any: ...


def _has_cypher(store: BaseGraphStore) -> TypeGuard[_CypherStore]:
    """Check if the store supports direct Cypher queries (FalkorDB)."""
    return hasattr(store, 'cypher') and callable(getattr(store, 'cypher', None))


class AthenaGraphAnalytics(BaseGraphAnalytics):
    """Domain-specific analytics layer for Athena's underwriting knowledge graph.

    Automatically uses Cypher queries when backed by FalkorDB for
    exact property filtering. Falls back to semantic search + post-hoc
    reconstruction when using Zep.
    """

    def query(
        self, input: str, *, product_type: str | None = None, **kwargs
    ) -> list[dict]:
        """Search for underwriting rules triggered by a risk factor.

        Parameters
        ----------
        input:
            Risk factor or query string (e.g. "Gas Station").
        product_type:
            Optional product type filter (e.g. "LRO", "BOP").
        """
        search_query = input
        if product_type:
            search_query = f'{input} {product_type}'

        limit = kwargs.get('limit', 10)
        result = self.store.search(search_query, limit=limit)
        return self._parse_triggers(result, input, product_type)

    def trace(
        self, input: str, *, product_type: str | None = None, **kwargs
    ) -> list[dict]:
        """Trace the full decision path for a risk factor.

        Returns a structured list showing RiskFactor -> Rule -> Outcome
        with any applicable mitigants.
        """
        search_query = input
        if product_type:
            search_query = f'{input} {product_type}'

        limit = kwargs.get('limit', 25)
        result = self.store.search(search_query, limit=limit)

        if result.is_empty:
            return []

        node_map = result.node_map
        paths: list[dict] = []

        triggers = result.edges_by_relation('TRIGGERS')
        results_in = result.edges_by_relation('RESULTS_IN')
        overrides = result.edges_by_relation('OVERRIDES')

        # Build rule -> outcome mapping
        rule_outcomes: dict[str, list[dict]] = {}
        for edge in results_in:
            target = node_map.get(edge.target_node_uuid)
            outcome_name = target.name if target else edge.target_node_uuid
            rule_outcomes.setdefault(edge.source_node_uuid, []).append(
                {
                    'outcome': outcome_name,
                    'properties': edge.properties,
                }
            )

        # Build rule -> mitigants mapping
        rule_mitigants: dict[str, list[str]] = {}
        for edge in overrides:
            source = node_map.get(edge.source_node_uuid)
            mitigant_name = source.name if source else edge.source_node_uuid
            rule_mitigants.setdefault(edge.target_node_uuid, []).append(mitigant_name)

        # Assemble full paths
        for edge in triggers:
            source = node_map.get(edge.source_node_uuid)
            target = node_map.get(edge.target_node_uuid)
            risk_name = source.name if source else edge.source_node_uuid
            rule_name = target.name if target else edge.target_node_uuid

            path = {
                'risk_factor': risk_name,
                'rule': rule_name,
                'trigger_properties': edge.properties,
                'outcomes': rule_outcomes.get(edge.target_node_uuid, []),
                'mitigants': rule_mitigants.get(edge.target_node_uuid, []),
            }
            paths.append(path)

        return paths

    def export(self, **kwargs) -> list[dict]:
        """Export all knowledge from the graph as structured dicts."""
        limit = kwargs.get('limit', 500)
        result = self.store.search('*', limit=limit)
        if result.is_empty:
            return []

        return self._parse_triggers(result, query='*', product_type=None)

    def check_risk_appetite(
        self,
        risk_factor: str,
        *,
        product_type: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """Domain alias for ``query()``."""
        return self.query(risk_factor, product_type=product_type, **kwargs)

    def get_rule(self, rule_name: str, *, limit: int = 25) -> dict | None:
        """Look up a specific rule and all its connected edges.

        Returns a dict with the rule name, its trigger (risk factor),
        outcomes, mitigants, and sources — or ``None`` if not found.
        """
        result = self.store.search(rule_name, limit=limit)
        if result.is_empty:
            return None

        node_map = result.node_map
        triggers = result.edges_by_relation('TRIGGERS')
        results_in = result.edges_by_relation('RESULTS_IN')
        overrides = result.edges_by_relation('OVERRIDES')
        derived = result.edges_by_relation('DERIVED_FROM')

        # Find the trigger edge whose target is this rule
        risk_factor = None
        trigger_props = {}
        for edge in triggers:
            target = node_map.get(edge.target_node_uuid)
            if target and target.name.lower() == rule_name.lower():
                source = node_map.get(edge.source_node_uuid)
                risk_factor = source.name if source else None
                trigger_props = edge.properties
                rule_uuid = edge.target_node_uuid
                break
        else:
            # Fallback: take the first trigger
            if triggers:
                edge = triggers[0]
                target = node_map.get(edge.target_node_uuid)
                source = node_map.get(edge.source_node_uuid)
                risk_factor = source.name if source else None
                trigger_props = edge.properties
                rule_uuid = edge.target_node_uuid
            else:
                return None

        outcomes = []
        for edge in results_in:
            if edge.source_node_uuid == rule_uuid:
                target = node_map.get(edge.target_node_uuid)
                outcomes.append(target.name if target else edge.target_node_uuid)

        mitigants = []
        for edge in overrides:
            if edge.target_node_uuid == rule_uuid:
                source = node_map.get(edge.source_node_uuid)
                mitigants.append(source.name if source else edge.source_node_uuid)

        sources = []
        for edge in derived:
            if edge.source_node_uuid == rule_uuid:
                target = node_map.get(edge.target_node_uuid)
                sources.append(target.name if target else edge.target_node_uuid)

        return {
            'rule_name': rule_name,
            'risk_factor': risk_factor,
            'trigger_properties': trigger_props,
            'outcomes': outcomes,
            'mitigants': mitigants,
            'sources': sources,
        }

    def _build_rule_index(self, result: GraphSearchResult) -> list[dict]:
        """Build a deduplicated list of rules from any search result.

        Zep returns edges as semantic facts — structured properties like
        ``action`` are embedded in the fact text, not as filterable fields.
        This helper reconstructs rule records from whatever edge types Zep
        returns (TRIGGERS, RESULTS_IN, OVERRIDES, DERIVED_FROM) by
        cross-referencing nodes.
        """
        if result.is_empty:
            return []

        node_map = result.node_map

        # Accumulate info per rule UUID
        rule_info: dict[str, dict] = {}

        # TRIGGERS edges: RiskFactor -> Rule
        for edge in result.edges_by_relation('TRIGGERS'):
            target = node_map.get(edge.target_node_uuid)
            source = node_map.get(edge.source_node_uuid)
            if not target:
                continue
            rule_uuid = edge.target_node_uuid
            if rule_uuid not in rule_info:
                rule_info[rule_uuid] = {
                    'rule_name': target.name,
                    'risk_factor': source.name if source else '',
                    'action': '',
                    'product_type': '',
                    'threshold_type': '',
                    'outcomes': [],
                    'fact': '',
                }
            props = edge.properties
            info = rule_info[rule_uuid]
            info['action'] = info['action'] or props.get('action', '')
            info['product_type'] = info['product_type'] or props.get('product_type', '')
            info['threshold_type'] = info['threshold_type'] or props.get(
                'threshold_type', ''
            )
            info['fact'] = info['fact'] or props.get('fact', '')

        # RESULTS_IN edges: Rule -> Outcome
        for edge in result.edges_by_relation('RESULTS_IN'):
            source = node_map.get(edge.source_node_uuid)
            target = node_map.get(edge.target_node_uuid)
            if not source:
                continue
            rule_uuid = edge.source_node_uuid
            outcome_name = target.name if target else ''
            if rule_uuid not in rule_info:
                rule_info[rule_uuid] = {
                    'rule_name': source.name,
                    'risk_factor': '',
                    'action': '',
                    'product_type': '',
                    'threshold_type': '',
                    'outcomes': [],
                    'fact': '',
                }
            info = rule_info[rule_uuid]
            if outcome_name and outcome_name not in info['outcomes']:
                info['outcomes'].append(outcome_name)
            # Infer action from outcome name if not set
            if not info['action'] and outcome_name:
                info['action'] = outcome_name.lower().replace(' ', '_')
            info['fact'] = info['fact'] or edge.properties.get('fact', '')

        return list(rule_info.values())

    def list_all_rules(self, *, limit: int = 500) -> list[dict]:
        """Return every rule in the graph with its action and product type."""
        if _has_cypher(self.store):
            res = self.store.cypher(
                'MATCH (f:RiskFactor)-[t:TRIGGERS]->(r:Rule) '
                'RETURN f.name AS risk_factor, r.name AS rule_name, '
                'r.action AS action, r.product_type AS product_type, '
                'r.threshold_type AS threshold_type '
                f'LIMIT {limit}'
            )
            return [
                {
                    'risk_factor': row[0] or '',
                    'rule_name': row[1] or '',
                    'action': row[2] or '',
                    'product_type': row[3] or '',
                    'threshold_type': row[4] or '',
                }
                for row in res.result_set
            ]

        result = self.store.search('*', limit=limit)
        return self._build_rule_index(result)

    def rules_by_action(self, action: str, *, limit: int = 500) -> list[dict]:
        """Return all rules matching a given action (e.g. 'decline', 'refer')."""
        if _has_cypher(self.store):
            from eval_workbench.shared.memory.falkor.store import _escape_cypher

            escaped = _escape_cypher(action.lower())
            res = self.store.cypher(
                'MATCH (f:RiskFactor)-[t:TRIGGERS]->(r:Rule) '
                f"WHERE toLower(r.action) CONTAINS '{escaped}' "
                'RETURN f.name AS risk_factor, r.name AS rule_name, '
                'r.action AS action, r.product_type AS product_type, '
                'r.threshold_type AS threshold_type '
                f'LIMIT {limit}'
            )
            return [
                {
                    'risk_factor': row[0] or '',
                    'rule_name': row[1] or '',
                    'action': row[2] or '',
                    'product_type': row[3] or '',
                    'threshold_type': row[4] or '',
                }
                for row in res.result_set
            ]

        result = self.store.search(action, limit=limit)
        all_rules = self._build_rule_index(result)

        action_lower = action.lower()
        return [
            r
            for r in all_rules
            if action_lower in r.get('action', '').lower()
            or action_lower in r.get('fact', '').lower()
            or action_lower in ' '.join(r.get('outcomes', [])).lower()
        ]

    def rules_by_product(self, product_type: str, *, limit: int = 500) -> list[dict]:
        """Return all rules applicable to a product type (e.g. 'BOP', 'Property').

        Includes rules with ``product_type='ALL'``.
        """
        if _has_cypher(self.store):
            from eval_workbench.shared.memory.falkor.store import _escape_cypher

            escaped = _escape_cypher(product_type)
            res = self.store.cypher(
                'MATCH (f:RiskFactor)-[t:TRIGGERS]->(r:Rule) '
                f"WHERE r.product_type = '{escaped}' OR r.product_type = 'ALL' "
                'RETURN f.name AS risk_factor, r.name AS rule_name, '
                'r.action AS action, r.product_type AS product_type, '
                'r.threshold_type AS threshold_type '
                f'LIMIT {limit}'
            )
            return [
                {
                    'risk_factor': row[0] or '',
                    'rule_name': row[1] or '',
                    'action': row[2] or '',
                    'product_type': row[3] or '',
                    'threshold_type': row[4] or '',
                }
                for row in res.result_set
            ]

        result = self.store.search(product_type, limit=limit)
        all_rules = self._build_rule_index(result)

        return [
            r
            for r in all_rules
            if not r.get('product_type')
            or r['product_type'] in ('ALL', product_type)
            or product_type.lower() in r.get('fact', '').lower()
        ]

    def uncovered_risks(self, *, limit: int = 500) -> list[str]:
        """Find risk factor nodes that have no TRIGGERS edges to any rule.

        Identifies potential gaps in the knowledge base.
        """
        result = self.store.search('*', limit=limit)
        if result.is_empty:
            return []

        node_map = result.node_map

        # Collect all node UUIDs that appear as sources in TRIGGERS edges
        triggered_sources: set[str] = set()
        trigger_target_uuids: set[str] = set()
        for edge in result.edges_by_relation('TRIGGERS'):
            triggered_sources.add(edge.source_node_uuid)
            trigger_target_uuids.add(edge.target_node_uuid)

        # Any node that appears as a source in non-TRIGGERS edges but never
        # as a source in a TRIGGERS edge could be an orphan risk factor.
        # More practically: nodes that aren't targets of TRIGGERS (not rules)
        # and aren't sources of TRIGGERS (not risk factors) are orphans.
        all_node_uuids = set(node_map.keys())
        # Nodes involved in TRIGGERS (either side) are covered
        covered = triggered_sources | trigger_target_uuids
        orphan_uuids = all_node_uuids - covered

        return [node_map[uid].name for uid in orphan_uuids if uid in node_map]

    def mitigants_for_rule(self, rule_name: str, *, limit: int = 25) -> list[str]:
        """Return all mitigant names that OVERRIDE a given rule."""
        if _has_cypher(self.store):
            from eval_workbench.shared.memory.falkor.store import _escape_cypher

            escaped = _escape_cypher(rule_name)
            res = self.store.cypher(
                f"MATCH (m:Mitigant)-[:OVERRIDES]->(r:Rule {{name: '{escaped}'}}) "
                f'RETURN m.name LIMIT {limit}'
            )
            return [row[0] for row in res.result_set if row[0]]

        result = self.store.search(rule_name, limit=limit)
        if result.is_empty:
            return []

        node_map = result.node_map
        mitigants: list[str] = []

        for edge in result.edges_by_relation('OVERRIDES'):
            target = node_map.get(edge.target_node_uuid)
            if target and target.name.lower() == rule_name.lower():
                source = node_map.get(edge.source_node_uuid)
                mitigants.append(source.name if source else edge.source_node_uuid)

        return mitigants

    def rules_mitigated_by(self, mitigant: str, *, limit: int = 25) -> list[str]:
        """Return all rule names that a given mitigant OVERRIDES."""
        if _has_cypher(self.store):
            from eval_workbench.shared.memory.falkor.store import _escape_cypher

            escaped = _escape_cypher(mitigant)
            res = self.store.cypher(
                f"MATCH (m:Mitigant {{name: '{escaped}'}})-[:OVERRIDES]->(r:Rule) "
                f'RETURN r.name LIMIT {limit}'
            )
            return [row[0] for row in res.result_set if row[0]]

        result = self.store.search(mitigant, limit=limit)
        if result.is_empty:
            return []

        node_map = result.node_map
        rules: list[str] = []

        for edge in result.edges_by_relation('OVERRIDES'):
            source = node_map.get(edge.source_node_uuid)
            if source and source.name.lower() == mitigant.lower():
                target = node_map.get(edge.target_node_uuid)
                rules.append(target.name if target else edge.target_node_uuid)

        return rules

    def unmitigated_declines(self, *, limit: int = 500) -> list[dict]:
        """Find decline rules with zero mitigants — hard stops with no overrides."""
        if _has_cypher(self.store):
            res = self.store.cypher(
                "MATCH (f:RiskFactor)-[:TRIGGERS]->(r:Rule {action: 'decline'}) "
                'WHERE NOT EXISTS { MATCH (:Mitigant)-[:OVERRIDES]->(r) } '
                'RETURN f.name AS risk_factor, r.name AS rule_name, '
                'r.product_type AS product_type '
                f'LIMIT {limit}'
            )
            return [
                {
                    'rule_name': row[1] or '',
                    'risk_factor': row[0] or '',
                    'product_type': row[2] or '',
                }
                for row in res.result_set
            ]

        result = self.store.search('decline', limit=limit)
        if result.is_empty:
            return []

        node_map = result.node_map

        # Collect rule UUIDs that have at least one OVERRIDES edge
        mitigated_rules: set[str] = set()
        for edge in result.edges_by_relation('OVERRIDES'):
            mitigated_rules.add(edge.target_node_uuid)

        rules: list[dict] = []
        seen: set[str] = set()
        for edge in result.edges_by_relation('TRIGGERS'):
            props = edge.properties
            fact = props.get('fact', '')
            action = props.get('action', '')
            if 'decline' not in action.lower() and 'decline' not in fact.lower():
                continue

            if edge.target_node_uuid in mitigated_rules:
                continue

            target = node_map.get(edge.target_node_uuid)
            source = node_map.get(edge.source_node_uuid)
            rule_name = target.name if target else edge.target_node_uuid
            if rule_name in seen:
                continue
            seen.add(rule_name)

            rules.append(
                {
                    'rule_name': rule_name,
                    'risk_factor': source.name if source else edge.source_node_uuid,
                    'product_type': props.get('product_type', ''),
                }
            )

        return rules

    def conflicting_rules(self, risk_factor: str, *, limit: int = 50) -> list[dict]:
        """Find rules where the same risk factor triggers conflicting actions.

        For example, a risk factor that triggers both 'decline' and 'refer'
        for the same product type.
        """
        result = self.store.search(risk_factor, limit=limit)
        if result.is_empty:
            return []

        node_map = result.node_map

        # Group trigger edges by (risk_factor_name, product_type)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for edge in result.edges_by_relation('TRIGGERS'):
            source = node_map.get(edge.source_node_uuid)
            target = node_map.get(edge.target_node_uuid)
            risk_name = source.name if source else edge.source_node_uuid
            props = edge.properties
            product = props.get('product_type', 'ALL')

            groups[(risk_name, product)].append(
                {
                    'rule_name': target.name if target else edge.target_node_uuid,
                    'action': props.get('action', ''),
                    'properties': props,
                }
            )

        conflicts: list[dict] = []
        for (risk_name, product), entries in groups.items():
            actions = {e['action'] for e in entries if e['action']}
            if len(actions) > 1:
                conflicts.append(
                    {
                        'risk_factor': risk_name,
                        'product_type': product,
                        'actions': sorted(actions),
                        'rules': entries,
                    }
                )

        return conflicts

    def overlapping_thresholds(self, field: str, *, limit: int = 100) -> list[dict]:
        """Find rules with thresholds on the same field but different values.

        Searches for edges whose threshold metadata references the given
        field name and groups them for comparison.
        """
        result = self.store.search(field, limit=limit)
        if result.is_empty:
            return []

        node_map = result.node_map
        matches: list[dict] = []

        for edge in result.edges_by_relation('TRIGGERS'):
            props = edge.properties
            threshold = props.get('threshold')
            if not isinstance(threshold, dict):
                continue
            if threshold.get('field', '') != field:
                continue

            target = node_map.get(edge.target_node_uuid)
            source = node_map.get(edge.source_node_uuid)
            matches.append(
                {
                    'rule_name': target.name if target else edge.target_node_uuid,
                    'risk_factor': source.name if source else edge.source_node_uuid,
                    'threshold': threshold,
                    'threshold_type': props.get('threshold_type', ''),
                    'product_type': props.get('product_type', ''),
                }
            )

        return matches

    def evaluate(
        self,
        risk_factor: str,
        *,
        product_type: str | None = None,
        context: dict | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """Evaluate a risk factor against the knowledge graph.

        Returns applicable rules with their outcomes and indicates which
        mitigants from ``context`` (if any) could override each rule.

        Parameters
        ----------
        risk_factor:
            The risk to evaluate (e.g. "Gas Station").
        product_type:
            Optional product filter.
        context:
            Submission data dict. Keys are checked against known mitigant
            names (case-insensitive substring match).
        """
        paths = self.trace(risk_factor, product_type=product_type, limit=limit)
        if not paths:
            return []

        context_keys = []
        if context:
            # Flatten context values to strings for matching
            context_keys = [str(v).lower() for v in context.values() if v] + [
                k.lower() for k in context
            ]

        evaluated: list[dict] = []
        for path in paths:
            applicable_mitigants = []
            if context_keys and path.get('mitigants'):
                for mitigant in path['mitigants']:
                    mitigant_lower = mitigant.lower()
                    for ck in context_keys:
                        if ck in mitigant_lower or mitigant_lower in ck:
                            applicable_mitigants.append(mitigant)
                            break

            evaluated.append(
                {
                    'risk_factor': path['risk_factor'],
                    'rule': path['rule'],
                    'trigger_properties': path['trigger_properties'],
                    'outcomes': path['outcomes'],
                    'all_mitigants': path['mitigants'],
                    'applicable_mitigants': applicable_mitigants,
                    'mitigated': len(applicable_mitigants) > 0,
                }
            )

        return evaluated

    def summary(self, *, limit: int = 500) -> dict:
        """Return aggregate statistics about the knowledge graph.

        Returns counts of nodes by type, edges by relation, rules by
        action, and rules by product type.
        """
        if _has_cypher(self.store):
            return self._summary_cypher(self.store)

        result = self.store.search('*', limit=limit)
        if result.is_empty:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'nodes_by_type': {},
                'edges_by_relation': {},
                'rules_by_action': {},
                'rules_by_product': {},
            }

        node_map = result.node_map

        # Count edges by relation
        edges_by_relation: dict[str, int] = defaultdict(int)
        for edge in result.edges:
            edges_by_relation[edge.relation] += 1

        # Classify nodes by examining their role in edges
        node_types: dict[str, set[str]] = defaultdict(set)
        rules_by_action: dict[str, int] = defaultdict(int)
        rules_by_product: dict[str, int] = defaultdict(int)

        for edge in result.edges_by_relation('TRIGGERS'):
            source = node_map.get(edge.source_node_uuid)
            target = node_map.get(edge.target_node_uuid)
            if source:
                node_types['RiskFactor'].add(source.name)
            if target:
                node_types['Rule'].add(target.name)

            props = edge.properties
            action = props.get('action', 'unknown')
            product = props.get('product_type', 'unknown')
            rules_by_action[action] += 1
            rules_by_product[product] += 1

        for edge in result.edges_by_relation('RESULTS_IN'):
            target = node_map.get(edge.target_node_uuid)
            if target:
                node_types['Outcome'].add(target.name)

        for edge in result.edges_by_relation('OVERRIDES'):
            source = node_map.get(edge.source_node_uuid)
            if source:
                node_types['Mitigant'].add(source.name)

        for edge in result.edges_by_relation('DERIVED_FROM'):
            target = node_map.get(edge.target_node_uuid)
            if target:
                node_types['Source'].add(target.name)

        nodes_by_type = {k: len(v) for k, v in node_types.items()}

        return {
            'total_nodes': len(result.nodes),
            'total_edges': len(result.edges),
            'nodes_by_type': dict(nodes_by_type),
            'edges_by_relation': dict(edges_by_relation),
            'rules_by_action': dict(rules_by_action),
            'rules_by_product': dict(rules_by_product),
        }

    def _summary_cypher(self, store: _CypherStore) -> dict:
        """Build summary stats using direct Cypher queries."""

        # Total nodes
        res = store.cypher('MATCH (n) RETURN count(n)')
        total_nodes = res.result_set[0][0] if res.result_set else 0

        # Total edges
        res = store.cypher('MATCH ()-[r]->() RETURN count(r)')
        total_edges = res.result_set[0][0] if res.result_set else 0

        # Nodes by label
        nodes_by_type: dict[str, int] = {}
        for label in ('RiskFactor', 'Rule', 'Outcome', 'Mitigant', 'Source'):
            res = store.cypher(f'MATCH (n:{label}) RETURN count(n)')
            count = res.result_set[0][0] if res.result_set else 0
            if count:
                nodes_by_type[label] = count

        # Edges by relation
        edges_by_relation: dict[str, int] = {}
        for rel in ('TRIGGERS', 'RESULTS_IN', 'OVERRIDES', 'DERIVED_FROM'):
            res = store.cypher(f'MATCH ()-[r:{rel}]->() RETURN count(r)')
            count = res.result_set[0][0] if res.result_set else 0
            if count:
                edges_by_relation[rel] = count

        # Rules by action
        res = store.cypher(
            'MATCH (r:Rule) WHERE r.action IS NOT NULL '
            'RETURN r.action, count(r) ORDER BY count(r) DESC'
        )
        rules_by_action = {row[0]: row[1] for row in res.result_set if row[0]}

        # Rules by product
        res = store.cypher(
            'MATCH (r:Rule) WHERE r.product_type IS NOT NULL '
            'RETURN r.product_type, count(r) ORDER BY count(r) DESC'
        )
        rules_by_product = {row[0]: row[1] for row in res.result_set if row[0]}

        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'nodes_by_type': nodes_by_type,
            'edges_by_relation': edges_by_relation,
            'rules_by_action': rules_by_action,
            'rules_by_product': rules_by_product,
        }

    @staticmethod
    def _parse_triggers(
        result: GraphSearchResult,
        query: str,
        product_type: str | None,
    ) -> list[dict]:
        """Parse search results into structured trigger dicts."""
        if result.is_empty:
            return []

        node_map = result.node_map
        triggers: list[dict] = []

        for edge in result.edges:
            source = node_map.get(edge.source_node_uuid)
            target = node_map.get(edge.target_node_uuid)

            entry = {
                'relation': edge.relation,
                'source': source.name if source else edge.source_node_uuid,
                'target': target.name if target else edge.target_node_uuid,
                'properties': edge.properties,
            }

            # Filter by product_type if specified
            if product_type:
                props = edge.properties
                edge_product = props.get('product_type', '')
                if edge_product and edge_product not in ('ALL', product_type):
                    continue

            triggers.append(entry)

        return triggers
