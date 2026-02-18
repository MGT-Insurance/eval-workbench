from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from axion.dataset import DatasetItem

from eval_workbench.shared.extractors.utils import (
    parse_json_like,
    safe_get,
    select_step_generation,
    select_step_span,
    to_plain_dict,
)

TInput = TypeVar('TInput')


class BaseExtractor(ABC, Generic[TInput]):
    """Abstract extractor contract: source input -> DatasetItem."""

    @abstractmethod
    def extract(self, source: TInput) -> DatasetItem:
        """Extract a DatasetItem from a source object."""

    def __call__(self, source: TInput) -> DatasetItem:
        return self.extract(source)


class ExtractorHelpers(BaseExtractor[TInput], ABC):
    """Shared helper methods for concrete extractors."""

    def safe_get(
        self,
        obj: Any,
        path: str,
        default: Any = None,
        *,
        fuzzy_dict_match: bool = False,
    ) -> Any:
        return safe_get(
            obj,
            path,
            default,
            fuzzy_dict_match=fuzzy_dict_match,
        )

    def to_plain_dict(self, value: Any) -> Any:
        return to_plain_dict(value)

    def parse_json_like(self, value: Any, *, use_json_repair: bool = True) -> Any:
        return parse_json_like(value, use_json_repair=use_json_repair)

    def select_step_generation(self, step: Any) -> Any:
        return select_step_generation(step)

    def select_step_span(self, step: Any) -> Any:
        return select_step_span(step)
