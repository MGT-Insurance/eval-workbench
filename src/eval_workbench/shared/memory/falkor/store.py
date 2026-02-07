from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from eval_workbench.shared.memory.ontology import OntologyDefinition
from eval_workbench.shared.memory.settings import FalkorSettings, get_falkor_settings
from eval_workbench.shared.memory.store import (
    BaseGraphStore,
    GraphEdge,
    GraphIngestPayload,
    GraphNode,
    GraphSearchResult,
)

logger = logging.getLogger(__name__)


def _escape_cypher(value: str) -> str:
    """Escape a string for safe use inside Cypher single quotes."""
    return value.replace('\\', '\\\\').replace("'", "\\'")


def _props_to_cypher(props: dict) -> str:
    """Convert a flat dict to a Cypher property map string.

    Handles str, int, float, bool, None. Complex values (dict, list)
    are JSON-serialized to strings so they survive round-tripping.
    """
    parts: list[str] = []
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f'{k}: {v}')
        elif isinstance(v, str):
            escaped = _escape_cypher(v)
            parts.append(f"{k}: '{escaped}'")
        elif isinstance(v, (dict, list)):
            serialized = json.dumps(v, separators=(',', ':'))
            parts.append(f"{k}: '{_escape_cypher(serialized)}'")
        else:
            parts.append(f"{k}: '{_escape_cypher(str(v))}'")
    return ', '.join(parts)


class FalkorGraphStore(BaseGraphStore):
    """Graph store backed by FalkorDB (Redis Graph).

    Uses Cypher queries for full CRUD. Node and edge properties are
    stored as real graph properties — no semantic fact conversion.
    """

    def __init__(
        self,
        agent_name: str,
        settings: FalkorSettings | None = None,
        ontology: OntologyDefinition | None = None,
    ) -> None:
        super().__init__(ontology=ontology)
        self.agent_name = agent_name
        self.settings = settings or get_falkor_settings()
        self._db = None

    @property
    def graph(self):
        """Lazy-import and initialize the FalkorDB graph."""
        if self._db is None:
            try:
                from falkordb import FalkorDB
            except ImportError as exc:
                raise ImportError(
                    'FalkorDB SDK is required. '
                    'Install with: pip install falkordb'
                ) from exc

            db = FalkorDB(
                host=self.settings.host,
                port=self.settings.port,
                password=self.settings.password or None,
            )
            graph_name = self.settings.graph_name_template.format(
                agent_name=self.agent_name,
            )
            self._db = db.select_graph(graph_name)
            logger.info('Connected to FalkorDB graph: %s', graph_name)
        return self._db

    # ------------------------------------------------------------------
    # BaseGraphStore interface
    # ------------------------------------------------------------------

    def ingest(self, payload: GraphIngestPayload) -> None:
        """Ingest edges into FalkorDB.

        Each edge dict is expected to have the same structure produced by
        ``AthenaRulePipeline._rule_to_ingest_payload``:

        - ``source``: Source node name
        - ``source_type``: Source node label (e.g. "RiskFactor")
        - ``target``: Target node name
        - ``target_type``: Target node label (e.g. "Rule")
        - ``relation``: Edge relationship type (e.g. "TRIGGERS")
        - ``source_properties``: Optional dict of source node properties
        - ``target_properties``: Optional dict of target node properties
        - ``properties``: Optional dict of edge properties
        """
        if not payload.edges:
            return

        for edge_dict in payload.edges:
            self._ingest_edge(edge_dict)

        logger.info(
            'Ingested %d edge(s) into FalkorDB for %s',
            len(payload.edges),
            self.agent_name,
        )

    def _ingest_edge(self, edge_dict: dict) -> None:
        """MERGE a single edge (and its endpoint nodes) into the graph."""
        source_name = edge_dict.get('source', '')
        source_label = edge_dict.get('source_type', 'Entity')
        target_name = edge_dict.get('target', '')
        target_label = edge_dict.get('target_type', 'Entity')
        relation = edge_dict.get('relation', 'RELATED_TO')

        # Build node properties — always include name
        source_props = {'name': source_name}
        source_props.update(edge_dict.get('source_properties', {}))

        target_props = {'name': target_name}
        target_props.update(edge_dict.get('target_properties', {}))

        edge_props = dict(edge_dict.get('properties', {}))
        # Add a UUID so edges are individually addressable
        if 'uuid' not in edge_props:
            edge_props['uuid'] = str(uuid.uuid4())

        src_cypher = _props_to_cypher(source_props)
        tgt_cypher = _props_to_cypher(target_props)
        edge_cypher = _props_to_cypher(edge_props)

        # MERGE nodes by (label, name), then CREATE the edge.
        # Using CREATE for edges so duplicate ingestions produce multiple
        # edges (matching Zep semantics). Switch to MERGE if dedup wanted.
        query = (
            f"MERGE (s:{source_label} {{name: '{_escape_cypher(source_name)}'}}) "
            f'SET s += {{{src_cypher}}} '
            f"MERGE (t:{target_label} {{name: '{_escape_cypher(target_name)}'}}) "
            f'SET t += {{{tgt_cypher}}} '
            f'CREATE (s)-[r:{relation} {{{edge_cypher}}}]->(t) '
            f'RETURN id(r)'
        )
        self.graph.query(query)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
    ) -> GraphSearchResult:
        """Search the FalkorDB graph.

        Supports several query modes:

        - ``"*"`` — return all edges
        - ``scope="relation:<REL>"`` — return edges of a specific relation type
        - ``scope="label:<LABEL>"`` — return nodes of a specific label
        - Otherwise, substring match on node names (case-insensitive)
        """
        if query == '*':
            return self._search_all(limit=limit)

        if scope and scope.startswith('relation:'):
            rel_type = scope.split(':', 1)[1]
            return self._search_by_relation(rel_type, limit=limit)

        if scope and scope.startswith('label:'):
            label = scope.split(':', 1)[1]
            return self._search_by_label(label, limit=limit)

        return self._search_by_name(query, limit=limit)

    def delete_node(self, node_uuid: str) -> None:
        """Delete a node by its internal graph ID or name.

        FalkorDB doesn't use external UUIDs by default. This method
        tries to match by name first.
        """
        query = (
            f"MATCH (n) WHERE n.name = '{_escape_cypher(node_uuid)}' "
            'DETACH DELETE n'
        )
        self.graph.query(query)
        logger.info('Deleted node %s from FalkorDB', node_uuid)

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self.graph.query('MATCH (n) DETACH DELETE n')
        logger.info('Cleared FalkorDB graph for %s', self.agent_name)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def add_mitigant(self, rule_name: str, mitigant_name: str) -> None:
        """Add an OVERRIDES edge between a mitigant and a rule."""
        payload = GraphIngestPayload(
            edges=[
                {
                    'source': mitigant_name,
                    'source_type': 'Mitigant',
                    'relation': 'OVERRIDES',
                    'target': rule_name,
                    'target_type': 'Rule',
                }
            ]
        )
        self.ingest(payload)

    # ------------------------------------------------------------------
    # Cypher query helpers (exposed for analytics)
    # ------------------------------------------------------------------

    def cypher(self, query: str, params: dict | None = None) -> Any:
        """Execute a raw Cypher query and return the result set.

        This is the escape hatch for structured queries that go beyond
        the BaseGraphStore interface — e.g. filtering by action, product
        type, or threshold fields directly.
        """
        return self.graph.query(query)

    # ------------------------------------------------------------------
    # Internal search implementations
    # ------------------------------------------------------------------

    def _search_all(self, *, limit: int) -> GraphSearchResult:
        """Return all edges in the graph."""
        query = (
            'MATCH (s)-[r]->(t) '
            f'RETURN s, r, t, type(r) AS rel, id(s) AS sid, id(t) AS tid, id(r) AS rid '
            f'LIMIT {limit}'
        )
        return self._result_from_query(query, '*')

    def _search_by_relation(self, rel_type: str, *, limit: int) -> GraphSearchResult:
        """Return edges of a specific relation type."""
        query = (
            f'MATCH (s)-[r:{rel_type}]->(t) '
            f'RETURN s, r, t, type(r) AS rel, id(s) AS sid, id(t) AS tid, id(r) AS rid '
            f'LIMIT {limit}'
        )
        return self._result_from_query(query, rel_type)

    def _search_by_label(self, label: str, *, limit: int) -> GraphSearchResult:
        """Return all edges connected to nodes of a given label."""
        query = (
            f'MATCH (s:{label})-[r]->(t) '
            f'RETURN s, r, t, type(r) AS rel, id(s) AS sid, id(t) AS tid, id(r) AS rid '
            f'LIMIT {limit}'
        )
        return self._result_from_query(query, label)

    def _search_by_name(self, name: str, *, limit: int) -> GraphSearchResult:
        """Substring search on node names, returning connected edges."""
        escaped = _escape_cypher(name.lower())
        query = (
            'MATCH (s)-[r]->(t) '
            f"WHERE toLower(s.name) CONTAINS '{escaped}' "
            f"OR toLower(t.name) CONTAINS '{escaped}' "
            f'RETURN s, r, t, type(r) AS rel, id(s) AS sid, id(t) AS tid, id(r) AS rid '
            f'LIMIT {limit}'
        )
        return self._result_from_query(query, name)

    def _result_from_query(self, cypher_query: str, original_query: str) -> GraphSearchResult:
        """Execute a Cypher query and convert to GraphSearchResult."""
        try:
            result = self.graph.query(cypher_query)
        except Exception as exc:
            logger.warning('FalkorDB query failed: %s', exc)
            return GraphSearchResult(query=original_query)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_nodes: set[str] = set()

        for row in result.result_set:
            s_node, r_edge, t_node, rel_type, s_id, t_id, r_id = row

            s_uuid = str(s_id)
            t_uuid = str(t_id)
            r_uuid = str(r_id)

            # Extract node properties from FalkorDB Node objects
            s_props = _extract_node_props(s_node)
            t_props = _extract_node_props(t_node)
            r_props = _extract_edge_props(r_edge)

            s_name = s_props.pop('name', s_uuid)
            t_name = t_props.pop('name', t_uuid)
            s_label = _extract_label(s_node)
            t_label = _extract_label(t_node)

            if s_uuid not in seen_nodes:
                seen_nodes.add(s_uuid)
                nodes.append(
                    GraphNode(
                        uuid=s_uuid,
                        name=s_name,
                        label=s_label,
                        properties=s_props,
                    )
                )
            if t_uuid not in seen_nodes:
                seen_nodes.add(t_uuid)
                nodes.append(
                    GraphNode(
                        uuid=t_uuid,
                        name=t_name,
                        label=t_label,
                        properties=t_props,
                    )
                )

            edges.append(
                GraphEdge(
                    uuid=r_uuid,
                    source_node_uuid=s_uuid,
                    target_node_uuid=t_uuid,
                    relation=rel_type,
                    properties=r_props,
                )
            )

        return GraphSearchResult(nodes=nodes, edges=edges, query=original_query)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        agent_name: str,
        ontology: OntologyDefinition | None = None,
    ) -> FalkorGraphStore:
        """Create a FalkorGraphStore from a YAML config file.

        Expected YAML structure::

            memory:
              falkor:
                host: "localhost"
                port: 6379
                graph_name_template: "{agent_name}_rules"
        """
        from eval_workbench.shared.config import load_config

        cfg = load_config(path)
        falkor_cfg = cfg.get('memory', {}).get('falkor', {})

        settings = FalkorSettings(
            host=falkor_cfg.get('host', 'localhost'),
            port=falkor_cfg.get('port', 6379),
            password=falkor_cfg.get('password'),
            graph_name_template=falkor_cfg.get(
                'graph_name_template', '{agent_name}_rules',
            ),
        )
        return cls(agent_name=agent_name, settings=settings, ontology=ontology)


# ---------------------------------------------------------------------------
# Helpers for extracting properties from FalkorDB SDK objects
# ---------------------------------------------------------------------------

def _extract_node_props(node: Any) -> dict:
    """Extract properties from a FalkorDB Node object."""
    props: dict = {}
    if hasattr(node, 'properties'):
        raw = node.properties
        if isinstance(raw, dict):
            props = dict(raw)
        elif hasattr(raw, 'items'):
            props = dict(raw.items())
    return props


def _extract_edge_props(edge: Any) -> dict:
    """Extract properties from a FalkorDB Edge object."""
    props: dict = {}
    if hasattr(edge, 'properties'):
        raw = edge.properties
        if isinstance(raw, dict):
            props = dict(raw)
        elif hasattr(raw, 'items'):
            props = dict(raw.items())
    return props


def _extract_label(node: Any) -> str:
    """Extract the label from a FalkorDB Node object."""
    if hasattr(node, 'alias'):
        return str(node.alias)
    if hasattr(node, 'labels'):
        labels = node.labels
        if isinstance(labels, (list, tuple)) and labels:
            return str(labels[0])
    if hasattr(node, 'label'):
        return str(node.label)
    return ''
