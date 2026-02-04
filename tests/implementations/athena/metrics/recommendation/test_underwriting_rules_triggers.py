import importlib.util
import re
from pathlib import Path


def _load_underwriting_rules_module():
    """
    Load `underwriting_rules.py` directly by file path to avoid executing
    `implementations.athena.metrics.recommendation.__init__`, which may import
    optional/alias modules not present in the unit-test environment.
    """
    here = Path(__file__).resolve()
    repo_root = next(parent for parent in here.parents if (parent / 'src').exists())
    module_path = (
        repo_root
        / 'src'
        / 'implementations'
        / 'athena'
        / 'metrics'
        / 'recommendation'
        / 'underwriting_rules.py'
    )

    spec = importlib.util.spec_from_file_location(
        'test_underwriting_rules_module', module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


uw_rules = _load_underwriting_rules_module()
TRIGGER_SPECS = uw_rules.TRIGGER_SPECS
TriggerName = uw_rules.TriggerName
UnderwritingRules = uw_rules.UnderwritingRules


def _spec(name):
    return next(s for s in TRIGGER_SPECS if s.name == name)


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def test_conv_store_temp_matches_common_synonyms():
    spec = _spec(TriggerName.CONV_STORE_TEMP)
    assert _matches_any(spec.patterns, 'Neighborhood liquor store - package store')
    assert _matches_any(spec.patterns, 'Corner store / bodega with lottery sales')
    assert _matches_any(spec.patterns, 'Convenience store open 24/7')


def test_non_owned_building_coverage_matches_lease_language():
    spec = _spec(TriggerName.NON_OWNED_BLDG)
    assert _matches_any(
        spec.patterns,
        'Applicant is a tenant requesting building coverage; need NNN (triple-net) lease',
    )


def test_number_of_employees_matches_common_phrasing():
    spec = _spec(TriggerName.NUM_EMPLOYEES)
    assert _matches_any(spec.patterns, 'Headcount is 35 employees')
    assert _matches_any(spec.patterns, 'Staff of 25 reported on the application')


def test_ghost_referral_classifier_instruction_is_synced_to_trigger_specs():
    uw = UnderwritingRules()
    instruction = uw.classifier.instruction
    assert 'Triggers (ordered by priority):' in instruction
    # Spot-check that we include the updated phrasing we expect to keep in sync.
    assert '- convStoreTemp:' in instruction
    assert 'liquor/package' in instruction
