from __future__ import annotations

from eval_workbench.shared.memory.ontology import (
    EdgeTypeDefinition,
    NodeTypeDefinition,
    OntologyDefinition,
)

ATHENA_ONTOLOGY = OntologyDefinition(
    name='athena_underwriting',
    version='1.1.0',
    description=(
        'Underwriting knowledge graph for Athena. Models risk factors, rules, '
        'outcomes, mitigants, and their relationships for insurance underwriting. '
        'Supports both hard deterministic rules and soft guidelines with historical '
        'precedent and threshold flexibility.'
    ),
    node_types=[
        NodeTypeDefinition(
            label='RiskFactor',
            description='An observable characteristic that affects underwriting. Includes traditional risk factors (Gas Station, Frame Construction), data quality patterns (Magic Dust discrepancy), geographic appetite (Tier 1 County), and pricing signals (below-minimum rate).',
            properties={
                'name': 'Canonical name of the risk factor.',
                'category': 'Risk category: occupancy, construction, location, operations, eligibility, pricing, data_quality, geographic_appetite, catastrophe, classification, process.',
                'severity': 'Severity level (low, medium, high, critical).',
            },
            required_properties=['name'],
        ),
        NodeTypeDefinition(
            label='Rule',
            description='An underwriting rule or guideline that dictates how to handle a risk factor. May be hard (absolute) or soft (guideline with known flexibility).',
            properties={
                'name': 'Canonical name of the rule.',
                'product_type': 'Insurance product type the rule applies to (e.g. LRO, BOP).',
                'action': 'Prescribed action (refer, decline, approve_with_conditions).',
                'description': 'Human-readable description of the rule.',
                'threshold_type': 'Whether the rule boundary is "hard" (absolute) or "soft" (flexible/discretionary).',
            },
            required_properties=['name'],
        ),
        NodeTypeDefinition(
            label='Outcome',
            description='The result of applying a rule (e.g. Refer, Decline, Approve with Conditions).',
            properties={
                'name': 'Outcome label.',
                'description': 'Details about the outcome.',
            },
            required_properties=['name'],
        ),
        NodeTypeDefinition(
            label='Mitigant',
            description="A condition or factor that can override or reduce a rule's impact.",
            properties={
                'name': 'Name of the mitigating factor.',
                'description': 'How this mitigant affects the rule.',
            },
            required_properties=['name'],
        ),
        NodeTypeDefinition(
            label='Source',
            description='Origin of the knowledge (e.g. underwriting manual, training session, SME interview, production decision log).',
            properties={
                'name': 'Source identifier.',
                'type': 'Source type (manual, training, sme, production, compliance).',
                'date': 'Date of the source material.',
            },
            required_properties=['name'],
        ),
    ],
    edge_types=[
        EdgeTypeDefinition(
            relation='TRIGGERS',
            source_label='RiskFactor',
            target_label='Rule',
            description='A risk factor triggers a specific underwriting rule.',
            properties={
                'confidence': 'Confidence level of the trigger relationship.',
                'product_type': 'Product type context for this trigger.',
                'threshold_type': 'Whether the threshold is "hard" or "soft".',
                'threshold': 'Numeric threshold object: {field, operator, value, unit}.',
                'historical_exceptions': 'Known cases where this rule was overridden in practice.',
                'decision_quality': 'From production cases: "aligned", "divergent", or "partial".',
                'compound_trigger': 'If multiple conditions must combine (e.g. "loss AND low_pricing AND prior_rejection").',
                'data_fields': 'System field paths referenced (e.g. auxData.rateData.input.property_rate).',
            },
        ),
        EdgeTypeDefinition(
            relation='RESULTS_IN',
            source_label='Rule',
            target_label='Outcome',
            description='A rule results in a specific underwriting outcome.',
            properties={
                'conditions': 'Conditions under which this outcome applies.',
            },
        ),
        EdgeTypeDefinition(
            relation='OVERRIDES',
            source_label='Mitigant',
            target_label='Rule',
            description='A mitigant can override or reduce the impact of a rule.',
            properties={
                'override_type': 'Type of override (full, partial, conditional).',
                'description': 'How the mitigant modifies the rule outcome.',
            },
        ),
        EdgeTypeDefinition(
            relation='DERIVED_FROM',
            source_label='Rule',
            target_label='Source',
            description='A rule is derived from a specific knowledge source.',
            properties={
                'extraction_date': 'When the rule was extracted from the source.',
            },
        ),
    ],
)
