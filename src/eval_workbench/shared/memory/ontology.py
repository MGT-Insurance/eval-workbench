from __future__ import annotations

from pydantic import BaseModel, Field


class NodeTypeDefinition(BaseModel, frozen=True):
    """Defines a node type in the knowledge graph ontology."""

    label: str = Field(description='Unique label for this node type.')
    description: str = Field(description='Human-readable description of the node type.')
    properties: dict[str, str] = Field(
        default_factory=dict,
        description='Property name -> description mapping.',
    )
    required_properties: list[str] = Field(
        default_factory=list,
        description='Properties that must be present on every node of this type.',
    )


class EdgeTypeDefinition(BaseModel, frozen=True):
    """Defines an edge type (relation) in the knowledge graph ontology."""

    relation: str = Field(description='Unique relation name (e.g. TRIGGERS).')
    source_label: str = Field(description='Label of the source node type.')
    target_label: str = Field(description='Label of the target node type.')
    description: str = Field(description='Human-readable description of the relation.')
    properties: dict[str, str] = Field(
        default_factory=dict,
        description='Property name -> description mapping.',
    )


class OntologyDefinition(BaseModel, frozen=True):
    """Complete ontology schema for a knowledge graph domain."""

    name: str = Field(description='Unique ontology name (e.g. athena_underwriting).')
    version: str = Field(default='1.0.0', description='Semantic version.')
    description: str = Field(description='Human-readable description of the ontology.')
    node_types: list[NodeTypeDefinition] = Field(
        default_factory=list,
        description='Node type definitions.',
    )
    edge_types: list[EdgeTypeDefinition] = Field(
        default_factory=list,
        description='Edge type definitions.',
    )

    def get_node_type(self, label: str) -> NodeTypeDefinition | None:
        """Look up a node type by label."""
        for nt in self.node_types:
            if nt.label == label:
                return nt
        return None

    def get_edge_type(self, relation: str) -> EdgeTypeDefinition | None:
        """Look up an edge type by relation name."""
        for et in self.edge_types:
            if et.relation == relation:
                return et
        return None

    def to_prompt_context(self) -> str:
        """Serialize the ontology for injection into LLM system prompts."""
        lines = [
            f'# Knowledge Graph Ontology: {self.name} (v{self.version})',
            f'{self.description}',
            '',
            '## Node Types',
        ]
        for nt in self.node_types:
            lines.append(f'- **{nt.label}**: {nt.description}')
            if nt.properties:
                for prop, desc in nt.properties.items():
                    req = ' (required)' if prop in nt.required_properties else ''
                    lines.append(f'  - `{prop}`: {desc}{req}')

        lines.append('')
        lines.append('## Edge Types (Relations)')
        for et in self.edge_types:
            lines.append(
                f'- **{et.relation}**: {et.source_label} -> {et.target_label} '
                f'-- {et.description}'
            )
            if et.properties:
                for prop, desc in et.properties.items():
                    lines.append(f'  - `{prop}`: {desc}')

        return '\n'.join(lines)


class OntologyRegistry:
    """Singleton registry for ontology definitions.

    Mirrors the metric_registry pattern from Axion.
    """

    def __init__(self) -> None:
        self._ontologies: dict[str, OntologyDefinition] = {}
        self._finalized: bool = False

    def register(self, ontology: OntologyDefinition) -> None:
        """Register an ontology definition."""
        if self._finalized:
            raise RuntimeError(
                f'Cannot register ontology {ontology.name!r}: registry is finalized.'
            )
        if ontology.name in self._ontologies:
            raise ValueError(f'Ontology {ontology.name!r} is already registered.')
        self._ontologies[ontology.name] = ontology

    def get(self, name: str) -> OntologyDefinition:
        """Retrieve an ontology by name."""
        if name not in self._ontologies:
            raise KeyError(
                f'Ontology {name!r} not found. '
                f'Available: {list(self._ontologies.keys())}'
            )
        return self._ontologies[name]

    def list_ontologies(self) -> list[str]:
        """Return a list of registered ontology names."""
        return list(self._ontologies.keys())

    def all(self) -> dict[str, OntologyDefinition]:
        """Return all registered ontologies."""
        return dict(self._ontologies)

    def finalize(self) -> None:
        """Lock the registry to prevent further registration."""
        self._finalized = True


ontology_registry = OntologyRegistry()
