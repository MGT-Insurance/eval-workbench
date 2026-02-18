from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from axion.dataset import DatasetItem

TInput = TypeVar('TInput')


class BaseExtractor(ABC, Generic[TInput]):
    """Abstract extractor contract: source input -> DatasetItem."""

    @abstractmethod
    def extract(self, source: TInput) -> DatasetItem:
        """Extract a DatasetItem from a source object."""

    def __call__(self, source: TInput) -> DatasetItem:
        return self.extract(source)


class ExtractorMixin(BaseExtractor[TInput], ABC):
    """Mixin-style shared helper methods for concrete extractors."""

    def safe_get(
        self,
        obj: Any,
        path: str,
        default: Any = None,
        *,
        fuzzy_dict_match: bool = False,
    ) -> Any:
        if obj is None:
            return default

        current = obj
        for part in path.split('.'):
            try:
                if isinstance(current, dict):
                    if part in current:
                        current = current[part]
                    elif fuzzy_dict_match:
                        matched = self._fuzzy_dict_get(current, part)
                        if matched is None:
                            return default
                        current = matched
                    else:
                        return default
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    current = current[part]
            except Exception:
                return default

            if current is None:
                return default

        return current

    def to_plain_dict(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: self.to_plain_dict(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.to_plain_dict(v) for v in value]
        if hasattr(value, 'to_dict'):
            try:
                return self.to_plain_dict(value.to_dict())
            except Exception:
                pass
        if hasattr(value, 'model_dump'):
            try:
                return self.to_plain_dict(value.model_dump())
            except Exception:
                pass
        if hasattr(value, '__dict__'):
            try:
                return self.to_plain_dict(vars(value))
            except Exception:
                pass
        return str(value)

    def parse_json_like(self, value: Any, *, use_json_repair: bool = True) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list, int, float, bool)):
            return value
        if not isinstance(value, str):
            return self.to_plain_dict(value)

        blob = value.strip()
        if not blob:
            return None

        try:
            return json.loads(blob)
        except Exception:
            pass

        if use_json_repair:
            try:
                from json_repair import repair_json

                repaired = repair_json(blob)
                return json.loads(repaired)
            except Exception:
                pass

        try:
            return ast.literal_eval(blob)
        except Exception:
            return None

    def select_step_generation(self, step: Any) -> Any:
        if step is None:
            return None
        try:
            for obs in list(getattr(step, 'observations', [])):
                if getattr(obs, 'type', '').upper() == 'GENERATION':
                    return obs
        except Exception:
            return None
        return self.safe_get(step, 'GENERATION', None) or self.safe_get(
            step, 'generation', None
        )

    def select_step_span(self, step: Any) -> Any:
        if step is None:
            return None
        try:
            for obs in list(getattr(step, 'observations', [])):
                if getattr(obs, 'type', '').upper() == 'SPAN':
                    return obs
        except Exception:
            return None
        return None

    @staticmethod
    def _fuzzy_dict_get(data: dict[str, Any], key: str) -> Any:
        target = key.lower().replace('_', '')
        for candidate_key, value in data.items():
            if candidate_key.lower().replace('_', '') == target:
                return value
        return None
