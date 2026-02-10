from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from eval_workbench.shared.memory.ontology import OntologyDefinition
from eval_workbench.shared.memory.settings import ZepSettings, get_zep_settings
from eval_workbench.shared.memory.store import (
    BaseGraphStore,
    GraphEdge,
    GraphIngestPayload,
    GraphNode,
    GraphSearchResult,
)

logger = logging.getLogger(__name__)


class ZepGraphStore(BaseGraphStore):
    """Graph store backed by Zep Cloud's knowledge graph API.

    Each agent gets its own Zep user for multi-tenancy isolation.
    """

    def __init__(
        self,
        agent_name: str,
        settings: ZepSettings | None = None,
        ontology: OntologyDefinition | None = None,
    ) -> None:
        super().__init__(ontology=ontology)
        self.agent_name = agent_name
        self.settings = settings or get_zep_settings()
        self.user_id = self.settings.resolve_user_id(agent_name)
        self._client = None

    @property
    def client(self):
        """Lazy-import and initialize the Zep client."""
        if self._client is None:
            try:
                from zep_cloud.client import Zep
            except ImportError as exc:
                raise ImportError(
                    'Zep Cloud SDK is required. '
                    'Install with: pip install eval-workbench[memory]'
                ) from exc

            if not self.settings.api_key:
                raise ValueError('ZEP_API_KEY is required but not set.')

            kwargs: dict[str, Any] = {'api_key': self.settings.api_key}
            if self.settings.base_url:
                kwargs['base_url'] = self.settings.base_url
            self._client = Zep(**kwargs)
            self._ensure_user_exists(self._client)
        return self._client

    def _ensure_user_exists(self, client: Any) -> None:
        """Create the Zep user for this agent if it doesn't already exist."""
        try:
            client.user.get(self.user_id)
            logger.debug('Zep user %s already exists.', self.user_id)
        except Exception:
            logger.info('Creating Zep user %s', self.user_id)
            client.user.add(
                user_id=self.user_id,
                email=self.settings.admin_email,
            )

    def ingest(self, payload: GraphIngestPayload) -> None:
        """Ingest edges into the Zep knowledge graph."""
        if not payload.edges:
            return

        data = json.dumps(payload.edges)
        self.client.graph.add(
            user_id=self.user_id,
            type='json',
            data=data,
        )
        logger.info('Ingested %d edge(s) for user %s', len(payload.edges), self.user_id)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
    ) -> GraphSearchResult:
        """Search the Zep knowledge graph."""
        kwargs = {
            'user_id': self.user_id,
            'query': query,
            'limit': limit,
        }
        if scope:
            kwargs['scope'] = scope

        raw = self.client.graph.search(**kwargs)

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_nodes: set[str] = set()

        for edge in getattr(raw, 'edges', []) or []:
            edge_uuid = getattr(edge, 'uuid', '') or ''
            source_uuid = getattr(edge, 'source_node_uuid', '') or ''
            target_uuid = getattr(edge, 'target_node_uuid', '') or ''
            relation = getattr(edge, 'relation', '') or ''
            fact = getattr(edge, 'fact', '') or ''

            edges.append(
                GraphEdge(
                    uuid=edge_uuid,
                    source_node_uuid=source_uuid,
                    target_node_uuid=target_uuid,
                    relation=relation,
                    properties={'fact': fact},
                )
            )

            # Build node stubs from edge endpoints
            for node_uuid, node_name_attr in [
                (source_uuid, 'source_node_name'),
                (target_uuid, 'target_node_name'),
            ]:
                if node_uuid and node_uuid not in seen_nodes:
                    seen_nodes.add(node_uuid)
                    nodes.append(
                        GraphNode(
                            uuid=node_uuid,
                            name=getattr(edge, node_name_attr, '') or node_uuid,
                        )
                    )

        return GraphSearchResult(nodes=nodes, edges=edges, query=query)

    def delete_node(self, node_uuid: str) -> None:
        """Delete a node from the Zep graph."""
        graph = cast(Any, self.client.graph)
        graph.delete_node(user_id=self.user_id, node_uuid=node_uuid)
        logger.info('Deleted node %s for user %s', node_uuid, self.user_id)

    def clear(self) -> None:
        """Delete and recreate the Zep user to clear all graph data."""
        try:
            self.client.user.delete(self.user_id)
            logger.info('Deleted Zep user %s', self.user_id)
        except Exception:
            logger.debug('User %s did not exist, nothing to clear.', self.user_id)

        self._client = None
        # Force re-creation on next access
        _ = self.client

    def add_mitigant(self, rule_name: str, mitigant_name: str) -> None:
        """Convenience method to add an OVERRIDES edge between a mitigant and a rule."""
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

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        agent_name: str,
        ontology: OntologyDefinition | None = None,
    ) -> ZepGraphStore:
        """Create a ZepGraphStore from a YAML config file.

        Expected YAML structure::

            memory:
              zep:
                api_key: "${ZEP_API_KEY}"
                admin_email: "..."
                user_id_template: "..."
        """
        from eval_workbench.shared.config import load_config

        cfg = load_config(path)
        zep_cfg = cfg.get('memory', {}).get('zep', {})

        settings = ZepSettings(
            api_key=zep_cfg.get('api_key'),
            base_url=zep_cfg.get('base_url'),
            admin_email=zep_cfg.get('admin_email', 'system@eval-workbench.local'),
            user_id_template=zep_cfg.get(
                'user_id_template', '{agent_name}_global_rules'
            ),
        )
        return cls(agent_name=agent_name, settings=settings, ontology=ontology)
