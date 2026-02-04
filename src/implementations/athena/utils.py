import re
from typing import Literal, Pattern, Tuple

OutcomePatternSet = Tuple[Pattern, Pattern, Pattern]


def get_outcome_patterns(
    variant: Literal[
        'refer_to_underwriter_reason', 'underwriting_rules', 'refer_reason'
    ] = 'refer_to_underwriter_reason',
) -> OutcomePatternSet:
    """
    Return (decline, refer, approval) regex patterns for outcome detection.
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

    # Default patterns for 'refer_to_underwriter_reason' and 'refer_reason'
    return (
        re.compile(r'\b(decline[ds]?|reject(?:ed)?)\b', re.IGNORECASE),
        re.compile(
            r'\b(refer(?:ral|red)?'
            r'|refer\s+to\s+(?:underwriter|uw)'
            r'|referral\s+to\s+uw'
            r'|review\s+required'
            r'|manual\s+review)\b',
            re.IGNORECASE,
        ),
        re.compile(
            r'\b(approv(?:e[ds]?|al)|accept(?:ed)?|clear(?:ed)?|bind)\b', re.IGNORECASE
        ),
    )


def detect_outcome(
    text: str,
    *,
    variant: Literal[
        'refer_to_underwriter_reason', 'underwriting_rules', 'refer_reason'
    ] = 'refer_to_underwriter_reason',
) -> Tuple[bool, str]:
    """
    Detect if the outcome is negative (referral/decline).

    Returns:
        Tuple of (is_negative: bool, outcome_label: str)
    """
    decline_pattern, refer_to_underwriter_pattern, approval_pattern = (
        get_outcome_patterns(variant)
    )
    has_decline = bool(decline_pattern.search(text))
    has_refer_to_underwriter = bool(refer_to_underwriter_pattern.search(text))
    has_approval = bool(approval_pattern.search(text))

    if has_decline:
        return True, 'Decline'
    if has_refer_to_underwriter:
        return True, 'Refer'
    if has_approval:
        return False, 'Approved'
    return False, 'Unknown'
