import re
from typing import Literal, Pattern, Tuple

OutcomePatternSet = Tuple[Pattern, Pattern, Pattern]


def get_outcome_patterns(
    variant: Literal['referral_reason', 'underwriting_rules'] = 'referral_reason',
) -> OutcomePatternSet:
    """
    Return (decline, referral, approval) regex patterns for outcome detection.
    """
    if variant == 'underwriting_rules':
        return (
            re.compile(r'\b(decline[ds]?|declin(?:e|ing))\b', re.IGNORECASE),
            re.compile(
                r'\b(refer(?:ral|red)?|block(?:ed)?|stop(?:ped)?)\b', re.IGNORECASE
            ),
            re.compile(
                r'\b(approv(?:e[ds]?|al)|auto[- ]?approv|clear(?:ed)?|accept(?:ed)?)\b',
                re.IGNORECASE,
            ),
        )

    return (
        re.compile(r'\b(decline[ds]?|reject(?:ed)?)\b', re.IGNORECASE),
        re.compile(
            r'\b(refer(?:ral|red)?|review\s+required|manual\s+review)\b', re.IGNORECASE
        ),
        re.compile(
            r'\b(approv(?:e[ds]?|al)|accept(?:ed)?|clear(?:ed)?|bind)\b', re.IGNORECASE
        ),
    )


def detect_outcome(
    text: str,
    *,
    variant: Literal['referral_reason', 'underwriting_rules'] = 'referral_reason',
) -> Tuple[bool, str]:
    """
    Detect if the outcome is negative (referral/decline).

    Returns:
        Tuple of (is_negative: bool, outcome_label: str)
    """
    decline_pattern, referral_pattern, approval_pattern = get_outcome_patterns(variant)
    has_decline = bool(decline_pattern.search(text))
    has_referral = bool(referral_pattern.search(text))
    has_approval = bool(approval_pattern.search(text))

    if has_decline:
        return True, 'Decline'
    if has_referral:
        return True, 'Referral'
    if has_approval:
        return False, 'Approved'
    return False, 'Unknown'
