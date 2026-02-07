from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from eval_workbench.shared.memory.store import BaseGraphStore, GraphIngestPayload

logger = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    """Outcome of a pipeline run."""

    items_processed: int = Field(default=0, description='Total items processed.')
    items_ingested: int = Field(default=0, description='Items successfully ingested.')
    items_failed: int = Field(default=0, description='Items that failed to ingest.')
    errors: list[str] = Field(
        default_factory=list,
        description='Error messages from failed items.',
    )


class BasePipeline(ABC):
    """Abstract base class for knowledge-graph ingestion pipelines.

    Subclasses implement ``extract`` to transform raw data into ingestable dicts.
    The concrete ``run`` method handles batching, ingestion, and error tracking.
    """

    def __init__(self, store: BaseGraphStore) -> None:
        self.store = store

    @abstractmethod
    def extract(self, raw_data: str | list | dict, **kwargs) -> list[dict]:
        """Extract structured items from raw data.

        Returns a list of dicts, each representing a knowledge unit to ingest.
        """

    def run(
        self,
        raw_data: str | list | dict,
        *,
        batch_size: int = 10,
        **kwargs,
    ) -> PipelineResult:
        """Execute the full extract-ingest pipeline.

        Parameters
        ----------
        raw_data:
            Input data to extract knowledge from.
        batch_size:
            Number of items to ingest per batch.
        """
        result = PipelineResult()

        items = self.extract(raw_data, **kwargs)
        result.items_processed = len(items)

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            for item in batch:
                try:
                    payload = GraphIngestPayload(edges=[item])
                    self.store.ingest(payload)
                    result.items_ingested += 1
                except Exception as exc:
                    result.items_failed += 1
                    result.errors.append(str(exc))
                    logger.warning('Ingestion failed for item: %s', exc)

        return result
