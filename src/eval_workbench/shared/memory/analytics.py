from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from eval_workbench.shared.memory.store import BaseGraphStore

logger = logging.getLogger(__name__)


class BaseGraphAnalytics(ABC):
    """Abstract base class for domain-specific graph analytics."""

    def __init__(self, store: BaseGraphStore) -> None:
        self.store = store

    @abstractmethod
    def query(self, input: str, **kwargs) -> list[dict]:
        """Primary runtime search: find relevant knowledge for a given input."""

    @abstractmethod
    def trace(self, input: str, **kwargs) -> list[dict]:
        """Full decision-path trace for a given input."""

    @abstractmethod
    def export(self, **kwargs) -> list[dict]:
        """Export all knowledge from the graph."""

    def export_to_json(self, file_path: str | Path, **kwargs) -> int:
        """Export all knowledge to a JSON file.

        Returns the number of items exported.
        """
        items = self.export(**kwargs)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(items, f, indent=2, default=str)
        logger.info('Exported %d items to %s', len(items), path)
        return len(items)

    def visualize(self, **kwargs) -> None:
        """Render a networkx/matplotlib visualization of the graph.

        Reconstructs a graph from search results and plots it. Requires
        ``networkx`` and ``matplotlib`` to be installed (available via the
        ``memory`` optional dependency group).
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                'Graph visualization requires networkx and matplotlib. '
                'Install with: pip install eval-workbench[memory]'
            ) from exc

        result = self.store.search('*', limit=200)
        if result.is_empty:
            logger.info('Graph is empty, nothing to visualize.')
            return

        G = nx.DiGraph()

        for node in result.nodes:
            G.add_node(node.uuid, label=node.name, node_type=node.label)

        for edge in result.edges:
            G.add_edge(
                edge.source_node_uuid,
                edge.target_node_uuid,
                relation=edge.relation,
            )

        pos = nx.spring_layout(G, seed=42)
        labels = {n: d.get('label', n[:8]) for n, d in G.nodes(data=True)}
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_size=1200,
            node_color='lightblue',
            font_size=8,
            arrows=True,
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        ax.set_title(kwargs.get('title', 'Knowledge Graph'))
        plt.tight_layout()
        plt.show()
