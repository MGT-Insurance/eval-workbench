from __future__ import annotations

import importlib
from enum import Enum
from typing import Any, Callable

from axion.dataset import DatasetItem

# Generic extractor callable used by monitoring sources.
# Langfuse extractors receive Trace-like inputs; Neon extractors receive row dicts.
ExtractorFn = Callable[[Any], DatasetItem]


class ExtractorKind(str, Enum):
    """Canonical keys for built-in extractor functions."""

    ATHENA_RECOMMENDATION = 'athena.recommendation'
    ATHENA_RECOMMENDATION_ROW = 'athena.recommendation.row'
    ATHENA_LOCATION_EXTRACTION = 'athena.location_extraction'
    MAGIC_DUST_GROUNDING = 'magic_dust.grounding'


EXTRACTOR_REGISTRY: dict[ExtractorKind, str] = {
    ExtractorKind.ATHENA_RECOMMENDATION: (
        'eval_workbench.implementations.athena.extractors.extract_recommendation'
    ),
    ExtractorKind.ATHENA_RECOMMENDATION_ROW: (
        'eval_workbench.implementations.athena.extractors.extract_recommendation_from_row'
    ),
    ExtractorKind.ATHENA_LOCATION_EXTRACTION: (
        'eval_workbench.implementations.athena.extractors.extract_location_extraction'
    ),
    ExtractorKind.MAGIC_DUST_GROUNDING: (
        'eval_workbench.implementations.magic_dust.extractors.extract_grounding'
    ),
}


def _import_callable(path: str) -> ExtractorFn:
    module_path, symbol_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, symbol_name)
    if not callable(fn):
        raise TypeError(f'Resolved symbol is not callable: {path}')
    return fn


def resolve_extractor(kind_or_key: ExtractorKind | str) -> ExtractorFn | None:
    """Resolve a built-in extractor by enum value or canonical key.

    Returns ``None`` if ``kind_or_key`` is not a known registry key.
    """
    if isinstance(kind_or_key, ExtractorKind):
        return _import_callable(EXTRACTOR_REGISTRY[kind_or_key])

    normalized = str(kind_or_key).strip().lower()
    if not normalized:
        return None

    try:
        kind = ExtractorKind(normalized)
    except ValueError:
        return None

    return _import_callable(EXTRACTOR_REGISTRY[kind])

