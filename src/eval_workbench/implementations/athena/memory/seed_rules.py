"""Manually curated seed rules for the Athena knowledge graph.

These are pre-defined referral trigger rules that can be loaded into
Neon and optionally ingested into Zep without running LLM extraction.

Usage::

    from eval_workbench.implementations.athena.memory.seed_rules import (
        REFERRAL_TRIGGER_RULES,
        seed_to_neon,
        seed_to_graph,
    )

    # Insert into Neon only
    with NeonConnection() as db:
        ids = seed_to_neon(db)

    # Insert into Neon + ingest into Zep graph
    with NeonConnection() as db:
        ids = seed_to_graph(db, store)
"""

from __future__ import annotations

import logging
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Referral trigger rules — manually curated from underwriting guidelines
# ---------------------------------------------------------------------------

REFERRAL_TRIGGER_RULES: list[dict] = [
    # --- convStoreTemp ---
    {
        'risk_factor': 'Convenience Store / Liquor Store',
        'risk_category': 'occupancy',
        'rule_name': 'convStoreTemp',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'All convenience stores and liquor stores automatically refer due to '
            'inherent business class risk. Required checks: (1) Crime/Burglary Score '
            'from Confianza data — DECLINE if total crime score = 5 OR burglary/theft '
            'score = 5, PASS if all scores 1-4. (2) Building Conditions via Google '
            'Maps/Street View — check roof, parking lot, multi-tenant issues, cooking '
            'equipment, propane storage. DECLINE if significant unaddressed hazard. '
            '(3) Prohibited Products — DECLINE if cannabis, THC, Delta-8/9/10, kratom, '
            'hookahs, bongs, or psychoactive items are sold. (4) Hours of Operation — '
            'DECLINE if 24/7 operations confirmed.'
        ),
        'mitigants': [
            'Crime/burglary scores 1-4',
            'Building reasonably maintained',
            'No prohibited products sold',
            'Not 24/7 operations',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': {'field': 'crime_score', 'operator': 'gte', 'value': 5, 'unit': 'score'},
        'threshold_type': 'hard',
        'historical_exceptions': None,
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': [
            'magicdust.confianza.crime_score',
            'magicdust.confianza.burglary_theft_score',
        ],
    },
    # --- bppToSalesRatio ---
    {
        'risk_factor': 'Low BPP-to-Sales Ratio',
        'risk_category': 'eligibility',
        'rule_name': 'bppToSalesRatio',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'Contents-to-sales ratio is below 10%, suggesting potentially inadequate '
            'coverage or unusual business model. Review business model: service businesses, '
            'consultancies, professional services operating from commercial space may '
            'legitimately have low ratios. APPROVE if low ratio is explainable by business '
            'model. REQUEST_INFO if cannot determine reason for low ratio — request '
            'inventory valuation or recommend agent increase contents limit.'
        ),
        'mitigants': [
            'Service business or consultancy (minimal physical inventory)',
            'Professional services operating from commercial space',
            'Large retail footprint with low-margin goods',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': {'field': 'bpp_to_sales_ratio', 'operator': 'lt', 'value': 0.10, 'unit': 'ratio'},
        'threshold_type': 'soft',
        'historical_exceptions': 'Low ratio is valid for service businesses, consultancies, and professional services',
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': ['auxData.rateData.input.bpp_to_sales_ratio'],
    },
    # --- orgEstYear ---
    {
        'risk_factor': 'New Business (< 3 Years) with Building Coverage',
        'risk_category': 'eligibility',
        'rule_name': 'orgEstYear',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'Business established within 3 years and seeking building coverage. New '
            'business + building ownership = elevated financial risk. Check owner '
            'experience: 3+ years similar experience is a significant mitigating factor. '
            'Check building condition via Google Maps/Street View. Restaurant exception: '
            'manager with 3+ years experience is acceptable. APPROVE if owner/manager has '
            '3+ years similar experience AND no building concerns (subject to inspection, '
            '100% ITV). REQUEST_INFO if cannot verify experience — request business plan '
            'showing 1.5% of building value in reserves or confirmation of prior experience. '
            'DECLINE if prior bankruptcy discovered.'
        ),
        'mitigants': [
            'Owner has 3+ years experience in similar business',
            'Restaurant manager with 3+ years experience',
            'Building in good condition (no remodel needed)',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': {'field': 'business_age_years', 'operator': 'lt', 'value': 3, 'unit': 'years'},
        'threshold_type': 'soft',
        'historical_exceptions': 'Approved when owner demonstrates 3+ years similar experience despite new business entity',
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': ['auxData.rateData.input.org_est_year'],
    },
    # --- nonOwnedBuildingCoverage ---
    {
        'risk_factor': 'Non-Owned Building Coverage Request',
        'risk_category': 'eligibility',
        'rule_name': 'nonOwnedBuildingCoverage',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'Agent requested building coverage but applicant does not own the building. '
            'Building coverage for non-owners requires triple-net lease showing contractual '
            'obligation. Always REQUEST_INFO: request copy of lease from agent. If lease '
            'supports building coverage = approve. If lease does not support = offer '
            'contents-only policy.'
        ),
        'mitigants': [
            'Triple-net lease on file showing contractual obligation',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': None,
        'threshold_type': None,
        'historical_exceptions': None,
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': [],
    },
    # --- businessNOC ---
    {
        'risk_factor': 'Not Otherwise Classified Business',
        'risk_category': 'eligibility',
        'rule_name': 'businessNOC',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            '"Not Otherwise Classified" was selected, meaning no standard class code fit. '
            'Research actual business operations via Google, website, social media. Determine '
            'if business can fit an existing MGT class code. If no class fits, assess whether '
            'this business type is something MGT would reasonably write based on similarity '
            'to existing appetite. APPROVE if can identify appropriate existing class (note '
            'which class). APPROVE if no class fits but business is clearly reasonable for '
            'MGT appetite. REQUEST_INFO if cannot determine business operations. DECLINE if '
            'business operations are clearly outside MGT small business appetite.'
        ),
        'mitigants': [
            'Business fits an existing MGT class code',
            'Business is clearly reasonable for MGT small business appetite',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': None,
        'threshold_type': None,
        'historical_exceptions': None,
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': [],
    },
    # --- bppValue ---
    {
        'risk_factor': 'High BPP Value (> $250k)',
        'risk_category': 'eligibility',
        'rule_name': 'bppValue',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'BPP limits exceed $250k, requiring reasonableness review. Some classes '
            'legitimately have high BPP: restaurants (equipment, fixtures), jewelry stores, '
            'medical offices (equipment), high-end retail, liquor stores (inventory), food '
            'retail. Check security measures via Google Maps/Street View: cameras, alarms, '
            'window bars/gates, sprinkler systems. Cross-reference with crime score — high '
            'BPP + high crime area = elevated concern. APPROVE if BPP reasonable for class '
            'AND adequate security measures. REQUEST_INFO if BPP seems high for class OR '
            'security unclear. DECLINE if BPP grossly unreasonable AND no security AND '
            'high crime area.'
        ),
        'mitigants': [
            'Business class legitimately requires high BPP',
            'Security cameras and alarm system installed',
            'Sprinkler system present',
            'Low crime area',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': {'field': 'bpp_value', 'operator': 'gt', 'value': 250000, 'unit': 'usd'},
        'threshold_type': 'soft',
        'historical_exceptions': 'High BPP approved for restaurants, jewelry stores, medical offices with adequate security',
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': ['auxData.rateData.input.bpp_value'],
    },
    # --- homeBasedBPP ---
    {
        'risk_factor': 'Home-Based Business with Contents Coverage',
        'risk_category': 'occupancy',
        'rule_name': 'homeBasedBPP',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'Home-based business seeking contents coverage. Check commercial readiness '
            'via Google Maps/Street View: ADA compliance, exit signage, dedicated/appropriate '
            'parking area, separation from residential use. Check if business involves 1:1 '
            'access to children, elderly, disabled, or other vulnerable populations — if '
            'yes, are there adequate controls? APPROVE if adequate commercial setup AND no '
            'vulnerable population concerns. REQUEST_INFO if cannot assess commercial readiness. '
            'DECLINE if clear lack of commercial controls AND/OR uncontrolled access to '
            'vulnerable populations.'
        ),
        'mitigants': [
            'ADA compliant location',
            'Proper exit signage',
            'Dedicated parking area',
            'Clear separation from residential use',
            'No vulnerable population access concerns',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': None,
        'threshold_type': None,
        'historical_exceptions': None,
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': [],
    },
    # --- claimsHistory ---
    {
        'risk_factor': 'Prior Claims History',
        'risk_category': 'claims',
        'rule_name': 'claimsHistory',
        'product_type': 'ALL',
        'action': 'refer',
        'outcome_description': (
            'Prior claim(s) reported on the risk. Classify claim as property or liability. '
            'For property: what was damaged, have repairs been made, what prevention measures '
            'are now in place? For liability: what were circumstances, what steps taken to '
            'prevent recurrence? Multiple claims may indicate moral hazard — look for pattern '
            'or frequency concerns. APPROVE if single claim with clear remediation demonstrated. '
            'REQUEST_INFO if claim details unclear or remediation status unknown. DECLINE if '
            'multiple claims suggesting pattern OR clear moral hazard indicators OR no '
            'remediation for significant prior loss.'
        ),
        'mitigants': [
            'Single claim with clear remediation',
            'Repairs completed and documented',
            'Prevention measures implemented',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': None,
        'threshold_type': 'soft',
        'historical_exceptions': 'Single claims with documented remediation are routinely approved',
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': ['auxData.rateData.input.claims_history'],
    },
    # --- numberOfEmployees ---
    {
        'risk_factor': 'Employee Count Exceeds 20',
        'risk_category': 'eligibility',
        'rule_name': 'numberOfEmployees',
        'product_type': 'BOP',
        'action': 'refer',
        'outcome_description': (
            'Employee count exceeds 20, requiring small business eligibility confirmation. '
            'Check revenue eligibility: max $5M for professional/medical office classes, $3M '
            'for all other classes — DECLINE if revenue exceeds limits. Check staffing '
            'reasonableness: payroll-per-employee typically $25k-$75k (varies by class), '
            'higher for professional services, lower for retail/food service. Verify business '
            'scale via Google/Street View — does location support stated employee count? '
            'APPROVE if within revenue limits AND staffing ratio reasonable. REQUEST_INFO if '
            'employee count seems misaligned. DECLINE if revenue exceeds limits OR staffing '
            'numbers clearly unreasonable.'
        ),
        'mitigants': [
            'Revenue within limits ($5M professional, $3M other)',
            'Payroll-per-employee ratio is reasonable for class',
            'Physical location supports stated employee count',
        ],
        'source': 'Referral Trigger Guidelines',
        'source_type': 'manual',
        'confidence': 'high',
        'threshold': {'field': 'employee_count', 'operator': 'gt', 'value': 20, 'unit': 'employees'},
        'threshold_type': 'soft',
        'historical_exceptions': 'Approved when revenue within limits and staffing ratio reasonable despite count > 20',
        'decision_quality': None,
        'compound_trigger': None,
        'data_fields': [
            'auxData.rateData.input.employee_count',
            'auxData.rateData.input.annual_revenue',
            'auxData.rateData.input.annual_payroll',
        ],
    },
]


# ---------------------------------------------------------------------------
# Seed functions
# ---------------------------------------------------------------------------


def seed_to_neon(
    db,
    *,
    rules: list[dict] | None = None,
    agent_name: str = 'athena',
    source_label: str = 'manual_seed',
) -> list[str]:
    """Insert seed rules into rule_extractions table.

    Returns list of generated IDs. Uses ``source_label`` as the
    batch raw_text so seeded rules are identifiable.

    Parameters
    ----------
    db:
        NeonConnection instance.
    rules:
        Rules to seed. Defaults to ``REFERRAL_TRIGGER_RULES``.
    agent_name:
        Agent name for the extractions.
    source_label:
        Label stored in ``raw_text`` to identify seeded rules.
    """
    from eval_workbench.shared.memory.persistence import save_extractions

    rules = rules or REFERRAL_TRIGGER_RULES
    batch_id = str(uuid.uuid4())

    ids = save_extractions(
        db,
        rules,
        batch_id=batch_id,
        agent_name=agent_name,
        raw_text=source_label,
    )
    logger.info(
        'Seeded %d rules into Neon (batch_id=%s, source=%s)',
        len(ids), batch_id, source_label,
    )
    return ids


def seed_to_graph(
    db,
    store,
    *,
    rules: list[dict] | None = None,
    agent_name: str = 'athena',
    source_label: str = 'manual_seed',
) -> list[str]:
    """Insert seed rules into Neon AND ingest into Zep graph.

    Saves to Neon first, then ingests each rule into the graph store,
    updating ingestion status on success or failure.

    Returns list of generated IDs.
    """
    from eval_workbench.implementations.athena.memory.pipeline import (
        AthenaRulePipeline,
    )
    from eval_workbench.shared.memory.persistence import mark_failed, mark_ingested

    rules = rules or REFERRAL_TRIGGER_RULES
    ids = seed_to_neon(db, rules=rules, agent_name=agent_name, source_label=source_label)

    for i, rule in enumerate(rules):
        try:
            payload = AthenaRulePipeline._rule_to_ingest_payload(rule)
            store.ingest(payload)
            mark_ingested(db, ids[i])
        except Exception as exc:
            logger.warning('Failed to ingest seed rule %s: %s', rule.get('rule_name'), exc)
            mark_failed(db, ids[i], str(exc))

    logger.info('Seeded %d rules into Neon + Zep graph.', len(ids))
    return ids
