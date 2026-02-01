"""
Test script for FeedbackConversationAnalyzer with sample data.

Run with: python tests/test_feedback_analyzer.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation

from implementations.athena.metrics.feedback import FeedbackConversationAnalyzer


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_CONVERSATIONS = [
    # Factual Correction (Hard Escalation)
    DatasetItem(
        id="test-001-factual-correction",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-12345 - Joe's Roofing LLC**

Base Score: 35/100
Recommendation: **Decline**

Risk Factors:
- Roof age: 25 years (exceeds 20-year threshold)
- No recent inspection on file
- High-risk zone for hail damage"""
                ),
                HumanMessage(
                    content="This is incorrect. The roof was replaced in 2022 - I have the inspection report right here. The data in the system is outdated."
                ),
                AIMessage(
                    content="I apologize for the error. Let me recalculate with the updated roof information. With a 2022 roof replacement, the risk profile changes significantly."
                ),
                HumanMessage(content="Yes, please update. The new roof is architectural shingles with 30-year warranty."),
                AIMessage(
                    content="""**Updated Assessment**

Base Score: 72/100
Recommendation: **Approve**

The 2022 roof replacement with architectural shingles significantly reduces the risk profile."""
                ),
                HumanMessage(content="Approved. Thanks for the correction."),
            ]
        ),
        additional_input={
            "thread_id": "T001",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-12345",
            "sender": "john.underwriter",
        },
    ),
    # No Intervention - STP (Straight Through Processing)
    DatasetItem(
        id="test-002-stp",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-67890 - Smith's Bakery**

Base Score: 85/100
Recommendation: **Approve**

Risk Assessment:
- Established business (15 years)
- Clean claims history
- Modern fire suppression system
- Recent electrical inspection passed

Premium: $4,250/year"""
                ),
            ]
        ),
        additional_input={
            "thread_id": "T002",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-67890",
            "sender": None,  # No human interaction
        },
    ),
    # Frustrated User
    DatasetItem(
        id="test-003-frustrated",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-11111 - Downtown Cafe**

Base Score: 45/100
Recommendation: **Review Required**

Flagged Issues:
- Mixed-use building classification unclear
- Alcohol percentage needs verification"""
                ),
                HumanMessage(
                    content="I already submitted this information TWICE!! Why isn't it showing up in the system???"
                ),
                AIMessage(
                    content="I understand your frustration. Let me check the submission history for this case."
                ),
                HumanMessage(
                    content="This is ridiculous. The same issue happened last week with another case. The system STILL doesn't work properly!"
                ),
                AIMessage(
                    content="I apologize for the continued issues. I can see there may be a sync delay. Let me escalate this to the technical team."
                ),
                HumanMessage(content="Fine. Just get it fixed. I'm done dealing with this today."),
            ]
        ),
        additional_input={
            "thread_id": "T003",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-11111",
            "sender": "frustrated.agent",
        },
    ),
    # Missing Context (Soft Escalation)
    DatasetItem(
        id="test-004-missing-context",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-22222 - Harbor Marina**

Base Score: 55/100
Recommendation: **Hold**

Missing Information:
- Unable to determine watercraft storage capacity
- Fuel storage details not found"""
                ),
                HumanMessage(
                    content="The marina doesn't store any watercraft - it's a dry dock facility only. They do minor repairs but no fuel on premises."
                ),
                AIMessage(
                    content="Thank you for that clarification. A dry dock facility with no fuel storage has a much better risk profile. Let me update the assessment."
                ),
                HumanMessage(content="Correct. They have a separate contract with the fuel dock next door."),
                AIMessage(
                    content="""**Updated Assessment**

Base Score: 78/100
Recommendation: **Approve**

Dry dock facility without fuel storage. Repair operations only."""
                ),
                HumanMessage(content="Looks good. Approved."),
            ]
        ),
        additional_input={
            "thread_id": "T004",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-22222",
            "sender": "mary.underwriter",
        },
    ),
    # Authority/Approval Required
    DatasetItem(
        id="test-005-authority",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-33333 - City Convention Center**

Base Score: 62/100
Recommendation: **Approve with Conditions**

This is a large account ($500K+ premium) requiring senior underwriter approval.

Risk Notes:
- High foot traffic venue
- Multiple event types
- Adequate safety measures in place"""
                ),
                HumanMessage(
                    content="I've reviewed the file. As senior underwriter, I'm approving this account. The safety protocols are comprehensive."
                ),
                AIMessage(content="Thank you for the approval. I'll update the case status to Approved."),
                HumanMessage(content="Confirmed. Please proceed with binding."),
            ]
        ),
        additional_input={
            "thread_id": "T005",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-33333",
            "sender": "senior.underwriter",
        },
    ),
    # Technical Issue
    DatasetItem(
        id="test-006-tech-issue",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-44444 - Tech Startup Inc**

Base Score: Error
Recommendation: Unable to calculate

System Error: Premium calculation returned $43.03 which appears incorrect for a $2M coverage limit."""
                ),
                HumanMessage(
                    content="Yeah, that's definitely wrong. Looks like the premium got divided by 100 somewhere. Should be around $4,300."
                ),
                AIMessage(
                    content="You're correct. I've identified a calculation error in the premium module. The base rate wasn't being multiplied correctly."
                ),
                HumanMessage(content="Can you flag this for the dev team? This is the third time this month."),
                AIMessage(content="I've logged the issue. Ticket #DEV-789 created for the engineering team."),
                HumanMessage(content="Thanks. I'll manually calculate the premium for now. $4,303 is correct."),
            ]
        ),
        additional_input={
            "thread_id": "T006",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-44444",
            "sender": "tech.savvy.agent",
        },
    ),
    # Clarification Request
    DatasetItem(
        id="test-007-clarification",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-55555 - Mountain Lodge Resort**

Base Score: 68/100
Recommendation: **Approve**

Coverage includes:
- Main lodge building
- 12 guest cabins
- Ski equipment rental shop"""
                ),
                HumanMessage(content="Quick question - does this include the outdoor hot tub area? That's usually a separate liability consideration."),
                AIMessage(
                    content="Good catch. The outdoor amenities including the hot tub area are covered under the general liability portion. The premium accounts for this with a 5% surcharge already applied."
                ),
                HumanMessage(content="Perfect, thanks for clarifying. Approved as-is."),
            ]
        ),
        additional_input={
            "thread_id": "T007",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-55555",
            "sender": "detail.oriented.uw",
        },
    ),
    # Declined Case
    DatasetItem(
        id="test-008-declined",
        conversation=MultiTurnConversation(
            messages=[
                AIMessage(
                    content="""**Case: MGT-BOP-66666 - Fireworks Warehouse**

Base Score: 12/100
Recommendation: **Decline**

Critical Risk Factors:
- Class 1.3G explosive storage
- Previous fire incident (2021)
- Location within 500ft of residential area
- Inadequate sprinkler system"""
                ),
                HumanMessage(
                    content="Agreed. This is well outside our risk appetite. Declining."
                ),
                AIMessage(content="Confirmed. Case marked as Declined."),
            ]
        ),
        additional_input={
            "thread_id": "T008",
            "channel_id": "C09MAP9HR9D",
            "case_id": "MGT-BOP-66666",
            "sender": "risk.manager",
        },
    ),
]


# =============================================================================
# Test Runner
# =============================================================================


async def run_tests():
    """Run FeedbackConversationAnalyzer on all sample conversations."""
    analyzer = FeedbackConversationAnalyzer()

    print("=" * 80)
    print("FeedbackConversationAnalyzer Test Results")
    print("=" * 80)

    all_results = []

    for item in SAMPLE_CONVERSATIONS:
        print(f"\n{'─' * 80}")
        print(f"Test: {item.id}")
        print(f"Case: {item.additional_input.get('case_id', 'N/A')}")
        print(f"{'─' * 80}")

        result = await analyzer.execute(item)
        all_results.append((item, result))

        signals = result.signals

        print(f"\n📊 Analysis Results:")
        print(f"   Explanation: {result.explanation}")

        print(f"\n📈 Interaction Signals:")
        print(f"   • Total messages: {signals.interaction.total_messages}")
        print(f"   • Human messages: {signals.interaction.human_messages}")
        print(f"   • Bot messages: {signals.interaction.bot_messages}")
        print(f"   • Exchanges: {signals.interaction.exchange_count}")
        print(f"   • Ping-pong ratio: {signals.interaction.ping_pong_ratio:.2f}")

        print(f"\n🔧 Intervention Signals:")
        print(f"   • Has intervention: {signals.intervention.has_intervention}")
        print(f"   • Type: {signals.intervention.intervention_type.value}")
        print(f"   • Friction point: {signals.intervention.friction_point or 'N/A'}")
        if signals.intervention.issue_details:
            print(f"   • Issue details: {signals.intervention.issue_details[:100]}...")

        print(f"\n⬆️ Escalation Signals:")
        print(f"   • Escalation type: {signals.escalation.escalation_type.value}")
        print(f"   • Is STP: {signals.escalation.is_stp}")
        print(f"   • Hard: {signals.escalation.is_hard_escalation}")
        print(f"   • Soft: {signals.escalation.is_soft_escalation}")
        print(f"   • Authority: {signals.escalation.is_authority_escalation}")

        print(f"\n😊 Sentiment Signals:")
        print(f"   • Sentiment: {signals.sentiment.sentiment.value}")
        print(f"   • Score: {signals.sentiment.sentiment_score:.2f}")
        print(f"   • Is frustrated: {signals.sentiment.is_frustrated}")
        if signals.sentiment.frustration_indicators:
            print(f"   • Indicators: {signals.sentiment.frustration_indicators[:3]}")

        print(f"\n✅ Resolution Signals:")
        print(f"   • Final status: {signals.resolution.final_status.value}")
        print(f"   • Is resolved: {signals.resolution.is_resolved}")
        print(f"   • Resolution type: {signals.resolution.resolution_type or 'N/A'}")
        print(f"   • Is stalemate: {signals.resolution.is_stalemate}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    intervention_count = sum(
        1 for _, r in all_results if r.signals.intervention.has_intervention
    )
    stp_count = sum(1 for _, r in all_results if r.signals.escalation.is_stp)
    frustrated_count = sum(1 for _, r in all_results if r.signals.sentiment.is_frustrated)
    resolved_count = sum(1 for _, r in all_results if r.signals.resolution.is_resolved)

    print(f"\nTotal conversations: {len(all_results)}")
    print(f"Interventions: {intervention_count} ({100*intervention_count/len(all_results):.0f}%)")
    print(f"STP (no escalation): {stp_count} ({100*stp_count/len(all_results):.0f}%)")
    print(f"Frustrated users: {frustrated_count} ({100*frustrated_count/len(all_results):.0f}%)")
    print(f"Resolved: {resolved_count} ({100*resolved_count/len(all_results):.0f}%)")

    # Escalation breakdown
    escalation_types = {}
    for _, r in all_results:
        etype = r.signals.escalation.escalation_type.value
        escalation_types[etype] = escalation_types.get(etype, 0) + 1

    print(f"\nEscalation breakdown:")
    for etype, count in sorted(escalation_types.items()):
        print(f"   • {etype}: {count}")

    # Test to_rows() and to_kpi_summary()
    print(f"\n{'─' * 80}")
    print("Testing output methods...")

    sample_result = all_results[0][1].signals
    rows = sample_result.to_rows()
    print(f"✓ to_rows() returned {len(rows)} rows")
    print(f"  Row metrics: {[r['metric'] for r in rows]}")

    kpi = sample_result.to_kpi_summary()
    print(f"✓ to_kpi_summary() returned {len(kpi)} fields")
    print(f"  Sample fields: {list(kpi.keys())[:5]}...")

    print(f"\n{'=' * 80}")
    print("All tests completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(run_tests())
