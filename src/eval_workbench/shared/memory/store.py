from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from eval_workbench.shared.memory.ontology import OntologyDefinition


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    uuid: str = Field(description='Unique identifier for the node.')
    name: str = Field(description='Display name of the node.')
    label: str = Field(default='', description='Ontology label (e.g. RiskFactor).')
    properties: dict = Field(
        default_factory=dict,
        description='Arbitrary key-value properties.',
    )


class GraphEdge(BaseModel):
    """An edge (relation) in the knowledge graph."""

    uuid: str = Field(description='Unique identifier for the edge.')
    source_node_uuid: str = Field(description='UUID of the source node.')
    target_node_uuid: str = Field(description='UUID of the target node.')
    relation: str = Field(description='Relation type (e.g. TRIGGERS).')
    properties: dict = Field(
        default_factory=dict,
        description='Arbitrary key-value properties.',
    )


class GraphSearchResult(BaseModel):
    """Result of a graph search operation."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    query: str = Field(default='', description='Original search query.')

    @property
    def node_map(self) -> dict[str, GraphNode]:
        """Return a UUID -> GraphNode lookup dict."""
        return {n.uuid: n for n in self.nodes}

    def edges_by_relation(self, relation: str) -> list[GraphEdge]:
        """Filter edges to only those matching the given relation."""
        return [e for e in self.edges if e.relation == relation]

    @property
    def is_empty(self) -> bool:
        """True if the result contains no nodes or edges."""
        return len(self.nodes) == 0 and len(self.edges) == 0


class GraphIngestPayload(BaseModel):
    """Payload for ingesting edges into the graph store."""

    edges: list[dict] = Field(
        default_factory=list,
        description='List of edge dicts to ingest.',
    )


class BaseGraphStore(ABC):
    """Abstract base class for knowledge graph storage backends."""

    def __init__(self, ontology: OntologyDefinition | None = None) -> None:
        self.ontology = ontology

    @abstractmethod
    def ingest(self, payload: GraphIngestPayload) -> None:
        """Ingest edges into the graph store."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
    ) -> GraphSearchResult:
        """Search the graph store and return matching nodes/edges."""

    @abstractmethod
    def delete_node(self, node_uuid: str) -> None:
        """Delete a node by UUID."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the graph store."""
