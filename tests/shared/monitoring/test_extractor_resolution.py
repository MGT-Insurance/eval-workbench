from __future__ import annotations

import pytest


def test_registry_resolves_enum_and_string_keys() -> None:
    from eval_workbench.shared.extractors import ExtractorKind, resolve_extractor

    fn_from_enum = resolve_extractor(ExtractorKind.ATHENA_RECOMMENDATION)
    fn_from_key = resolve_extractor('athena.recommendation')

    assert fn_from_enum is not None
    assert fn_from_key is not None
    assert fn_from_enum.__name__ == 'extract_recommendation'
    assert fn_from_key.__name__ == 'extract_recommendation'


def test_registry_unknown_key_returns_none() -> None:
    from eval_workbench.shared.extractors import resolve_extractor

    assert resolve_extractor('not.a.registered.extractor') is None


def test_monitor_resolve_extractor_supports_registry_and_dotted_path() -> None:
    from eval_workbench.shared.monitoring import monitor as monitor_mod

    registry_resolved = monitor_mod._resolve_extractor('athena.recommendation')
    dotted_resolved = monitor_mod._resolve_extractor(
        'eval_workbench.implementations.athena.extractors.extract_recommendation'
    )

    assert registry_resolved.__name__ == 'extract_recommendation'
    assert dotted_resolved.__name__ == 'extract_recommendation'


def test_monitor_resolve_extractor_raises_for_unknown_identifier() -> None:
    from eval_workbench.shared.monitoring import monitor as monitor_mod

    with pytest.raises((ModuleNotFoundError, AttributeError, ValueError)):
        monitor_mod._resolve_extractor('totally.invalid.extractor.identifier')

