import re
from typing import Any, Dict, List, Optional

from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion.dataset_schema import MultiTurnConversation
from pydantic import Field


class SlackMetadata(RichBaseModel):
    """Parsed metadata from Slack thread additional_input."""

    thread_ts: Optional[str] = Field(default=None, description='Thread timestamp ID')
    channel_id: Optional[str] = Field(default=None, description='Slack channel ID')
    reply_count: Optional[int] = Field(
        default=None, description='Number of replies in thread'
    )
    sender: Optional[str] = Field(default=None, description='Original message sender')
    team_id: Optional[str] = Field(default=None, description='Slack team/workspace ID')


def parse_slack_metadata(additional_input: Optional[Dict[str, Any]]) -> SlackMetadata:
    """
    Parse Slack-specific metadata from additional_input dictionary.

    Args:
        additional_input: Dictionary containing Slack metadata fields

    Returns:
        SlackMetadata with parsed values
    """
    if not additional_input:
        return SlackMetadata()

    return SlackMetadata(
        thread_ts=additional_input.get('thread_ts'),
        channel_id=additional_input.get('channel_id'),
        reply_count=additional_input.get('reply_count'),
        sender=additional_input.get('sender'),
        team_id=additional_input.get('team_id'),
    )


def extract_mentions(text: str) -> List[str]:
    """
    Extract @mentions from Slack message text.

    Handles both Slack's encoded format (<@U123ABC>) and display format (@username).

    Args:
        text: Message text to parse

    Returns:
        List of mentioned user IDs or usernames
    """
    if not text:
        return []

    # Slack encoded format: <@U123ABC> or <@U123ABC|username>
    encoded_mentions = re.findall(r'<@([A-Z0-9]+)(?:\|[^>]+)?>', text)

    # Display format: @username (alphanumeric with underscores/hyphens)
    display_mentions = re.findall(r'(?<!\S)@([a-zA-Z0-9_-]+)', text)

    return list(set(encoded_mentions + display_mentions))


def detect_ai_sender(
    sender: Optional[str], ai_names: Optional[List[str]] = None
) -> bool:
    """
    Determine if a sender is an AI assistant.

    Args:
        sender: Sender name or ID
        ai_names: List of known AI assistant names (default: ['Athena'])

    Returns:
        True if sender appears to be an AI assistant
    """
    if not sender:
        return False

    ai_names = ai_names or ['Athena']
    sender_lower = sender.lower()

    return any(name.lower() in sender_lower for name in ai_names)


def get_human_messages(
    conversation: Optional[MultiTurnConversation],
) -> List[HumanMessage]:
    """
    Extract all human messages from a conversation.

    Args:
        conversation: MultiTurnConversation object

    Returns:
        List of HumanMessage objects
    """
    if not conversation or not conversation.messages:
        return []

    return [msg for msg in conversation.messages if isinstance(msg, HumanMessage)]


def get_ai_messages(conversation: Optional[MultiTurnConversation]) -> List[AIMessage]:
    """
    Extract all AI messages from a conversation.

    Args:
        conversation: MultiTurnConversation object

    Returns:
        List of AIMessage objects
    """
    if not conversation or not conversation.messages:
        return []

    return [msg for msg in conversation.messages if isinstance(msg, AIMessage)]


def find_recommendation_turn(
    conversation: Optional[MultiTurnConversation],
) -> Optional[int]:
    """
    Find the turn index containing an AI recommendation.

    Searches for recommendation patterns in AI messages.

    Args:
        conversation: MultiTurnConversation object

    Returns:
        Turn index (0-based) of first recommendation, or None if not found
    """
    if not conversation or not conversation.messages:
        return None

    for idx, msg in enumerate(conversation.messages):
        if isinstance(msg, AIMessage) and msg.content:
            if has_recommendation_pattern(msg.content):
                return idx

    return None


def has_recommendation_pattern(text: str) -> bool:
    """
    Check if text contains a recommendation pattern.

    Args:
        text: Text to check

    Returns:
        True if text contains a recommendation pattern
    """
    if not text:
        return False

    for pattern in RECOMMENDATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def extract_recommendation_type(text: str) -> Optional[str]:
    """
    Extract the recommendation type from text.

    Args:
        text: Text containing recommendation

    Returns:
        Recommendation type: 'approve', 'decline', 'review', 'hold', or None
    """
    if not text:
        return None

    # Primary recommendation pattern
    match = re.search(r'Recommend\s+(Approve|Decline|Review|Hold)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Secondary patterns
    text_lower = text.lower()
    if 'recommend approval' in text_lower or 'recommended for approval' in text_lower:
        return 'approve'
    if 'recommend decline' in text_lower or 'recommended to decline' in text_lower:
        return 'decline'
    if 'recommend review' in text_lower or 'needs review' in text_lower:
        return 'review'
    if 'recommend hold' in text_lower or 'place on hold' in text_lower:
        return 'hold'

    return None


def extract_case_id(text: str) -> Optional[str]:
    """
    Extract case identifier from text.

    Supports MGT-BOP format and similar patterns.

    Args:
        text: Text to parse

    Returns:
        Case ID if found, None otherwise
    """
    if not text:
        return None

    # MGT-BOP format: MGT-BOP-1234567
    match = re.search(r'(MGT-BOP-\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Generic case ID format: CASE-12345 or similar
    match = re.search(r'(CASE[-_]?\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def extract_priority_score(text: str) -> Optional[int]:
    """
    Extract priority/base score from text.

    Args:
        text: Text to parse

    Returns:
        Priority score (0-100) if found, None otherwise
    """
    if not text:
        return None

    # Pattern: "Base Score: 47/100" or "Priority Score: 85"
    match = re.search(
        r'(?:Base|Priority)\s*Score[:\s]+(\d+)(?:/100)?', text, re.IGNORECASE
    )
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 100)  # Clamp to 0-100

    return None


def count_questions(text: str) -> int:
    """
    Count the number of questions in text.

    Args:
        text: Text to analyze

    Returns:
        Number of questions detected
    """
    if not text:
        return 0

    # Count question marks
    question_count = text.count('?')

    # Also detect question patterns without question marks
    question_patterns = [
        r'\b(?:can|could|would|should|will|do|does|did|is|are|was|were|have|has|had)\s+(?:you|we|i|they)\b',
        r'\bwhat\s+(?:is|are|was|were|about)\b',
        r'\bhow\s+(?:do|does|did|can|could|would|should)\b',
        r'\bwhy\s+(?:is|are|was|were|do|does|did)\b',
    ]

    for pattern in question_patterns:
        question_count += len(re.findall(pattern, text, re.IGNORECASE))

    return question_count


def build_transcript(conversation: Optional[MultiTurnConversation]) -> str:
    """
    Build a plain text transcript from conversation messages.

    Args:
        conversation: MultiTurnConversation object

    Returns:
        Plain text transcript with role prefixes
    """
    if not conversation or not conversation.messages:
        return ''

    lines = []
    for msg in conversation.messages:
        role = 'User' if isinstance(msg, HumanMessage) else 'AI'
        content = msg.content or ''
        lines.append(f'{role}: {content}')

    return '\n'.join(lines)


# Common recommendation patterns to detect
RECOMMENDATION_PATTERNS = [
    r'Recommend\s+(Approve|Decline|Review|Hold)',
    r'Base\s*Score[:\s]+\d+/100',
    r'Priority\s*Score[:\s]+\d+',
    r'MGT-BOP-\d+',
    r'(?:recommended?|suggestion|advice)[:\s]+(?:approve|decline|review|hold)',
]

# Escalation indicator patterns
ESCALATION_PATTERNS = [
    r'<@[A-Z0-9]+>',  # Slack user mention
    r'@[a-zA-Z0-9_-]+',  # Display mention
    r'(?:please|can|could)\s+(?:review|check|look\s+at|help)',
    r'need(?:s)?\s+(?:human|manual|someone|help)',
    r'(?:error|issue|problem)\s+(?:occurred|happened|with)',
    r'apologize.*(?:error|issue|unable)',
    r'(?:escalat|handoff|transfer)',
]

# Frustration indicator patterns
FRUSTRATION_PATTERNS = [
    r'\?\?+',  # Multiple question marks
    r'!!+',  # Multiple exclamation marks
    r'[A-Z]{3,}',  # All caps words (3+ chars)
    r'(?:frustrated|annoyed|confused|wrong|incorrect|bad)',
    r'(?:doesn\'t|does not|didn\'t|did not)\s+(?:work|help|make sense)',
    r'(?:still|again)\s+(?:not|doesn\'t|wrong)',
    r'(?:what|how)\s+(?:is|are)\s+(?:wrong|going on)',
]
