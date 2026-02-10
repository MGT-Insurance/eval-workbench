import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from axion._core.schema import AIMessage, BaseMessage, HumanMessage, RichBaseModel
from axion.dataset_schema import MultiTurnConversation as Conversation
from pydantic import Field

# Recommendation type keywords
RECOMMENDATION_KEYWORDS = {
    'approve': ['recommend approval', 'approve', 'approved', 'recommend accepting'],
    'decline': [
        'recommend decline',
        'decline',
        'declined',
        'recommend rejecting',
        'reject',
    ],
    'review': ['recommend review', 'review required', 'needs review', 'further review'],
    'hold': ['recommend hold', 'on hold', 'holding', 'pending review'],
}

# Reaction sentiment mappings
POSITIVE_REACTIONS = {
    'thumbsup',
    '+1',
    'heart',
    'tada',
    'white_check_mark',
    'heavy_check_mark',
    'star',
    'rocket',
    'fire',
    'clap',
    'raised_hands',
    'ok_hand',
    '100',
    'muscle',
    'pray',
}

NEGATIVE_REACTIONS = {
    'thumbsdown',
    '-1',
    'x',
    'warning',
    'no_entry',
    'no_entry_sign',
    'confused',
    'angry',
    'rage',
    'disappointed',
    'cry',
    'sob',
    'facepalm',
    'skull',
}


@dataclass
class SlackMetadata:
    """Parsed Slack metadata from additional_input and/or dataset_metadata."""

    thread_ts: Optional[str] = None
    channel_id: Optional[str] = None
    sender: Optional[str] = None
    reply_count: Optional[int] = None
    reactions: Optional[Dict[str, int]] = None
    participants: Optional[List[str]] = None


class ReactionSignals(RichBaseModel):
    """Slack emoji reaction analysis (heuristic)."""

    thumbs_up_count: int = Field(
        default=0, description='Count of positive thumb reactions'
    )
    thumbs_down_count: int = Field(
        default=0, description='Count of negative thumb reactions'
    )
    has_positive_reaction: bool = Field(
        default=False, description='Whether any positive reaction exists'
    )
    has_negative_reaction: bool = Field(
        default=False, description='Whether any negative reaction exists'
    )
    reaction_sentiment_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description='Overall sentiment from reactions (0=negative, 0.5=neutral, 1=positive)',
    )
    all_reactions: Dict[str, int] = Field(
        default_factory=dict, description='All reactions with counts'
    )


class StalemateSignals(RichBaseModel):
    """Detect bot repeating same error/message (heuristic)."""

    is_stalemate: bool = Field(
        default=False, description='Whether a stalemate was detected'
    )
    repeated_message_count: int = Field(
        default=0, description='Number of times the same message was repeated'
    )
    stalemate_turn_index: Optional[int] = Field(
        default=None, description='Turn where stalemate began'
    )
    repeated_content: Optional[str] = Field(
        default=None, description='The content that was repeated'
    )


def _load_dataset_metadata(dataset_metadata: Optional[str]) -> Dict[str, Any]:
    """Parse DatasetItem.dataset_metadata JSON string into a dict."""
    if not dataset_metadata:
        return {}
    try:
        value = json.loads(dataset_metadata)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def parse_slack_metadata(
    additional_input: Optional[Dict[str, Any]],
    dataset_metadata: Optional[str] = None,
) -> SlackMetadata:
    """
    Parse Slack metadata from DatasetItem additional_input and/or dataset_metadata.

    Args:
        additional_input: Dictionary containing Slack metadata
        dataset_metadata: DatasetItem.dataset_metadata JSON string (optional)

    Returns:
        SlackMetadata dataclass with parsed values
    """
    meta_dict = _load_dataset_metadata(dataset_metadata)
    merged: Dict[str, Any] = {**(additional_input or {}), **meta_dict}
    if not merged:
        return SlackMetadata()

    return SlackMetadata(
        thread_ts=merged.get('thread_ts'),
        channel_id=merged.get('channel_id') or merged.get('channel'),
        sender=merged.get('sender') or merged.get('user'),
        reply_count=merged.get('reply_count'),
        reactions=merged.get('reactions'),
        participants=merged.get('participants'),
    )


def get_ai_messages(conversation: Conversation) -> List[AIMessage]:
    """Extract AI messages from a conversation."""
    if not conversation or not conversation.messages:
        return []
    return [m for m in conversation.messages if isinstance(m, AIMessage)]


def get_human_messages(conversation: Conversation) -> List[HumanMessage]:
    """Extract human messages from a conversation."""
    if not conversation or not conversation.messages:
        return []
    return [m for m in conversation.messages if isinstance(m, HumanMessage)]


def build_transcript(
    conversation: Conversation,
    include_turn_numbers: bool = True,
    max_turns: Optional[int] = None,
) -> str:
    """
    Build a formatted transcript string from a conversation.
    """
    if not conversation or not conversation.messages:
        return ''

    messages = conversation.messages
    if max_turns:
        messages = messages[:max_turns]

    lines = []
    for i, msg in enumerate(messages):
        role = 'Human' if isinstance(msg, HumanMessage) else 'AI'
        content = msg.content or '[empty]'

        if include_turn_numbers:
            lines.append(f'[Turn {i + 1}] {role}: {content}')
        else:
            lines.append(f'{role}: {content}')

    return '\n'.join(lines)


def count_questions(text: str) -> int:
    """Count the number of questions in a text."""
    if not text:
        return 0

    question_marks = text.count('?')

    question_patterns = [
        r'\b(what|when|where|why|how|who|which|can you|could you|would you|do you|is there|are there)\b',
    ]

    pattern_matches = 0
    for pattern in question_patterns:
        matches = re.findall(pattern, text.lower())
        pattern_matches += len(matches)

    return max(question_marks, pattern_matches)


def extract_case_id(text: str) -> Optional[str]:
    """Extract a case ID from text."""
    if not text:
        return None

    patterns = [
        r'[Cc]ase\s*(?:[Ii][Dd])?[:# ]\s*([A-Za-z0-9-]*\d[A-Za-z0-9-]*)',
        r'[Rr]eference[:# ]\s*([A-Za-z0-9-]+)',
        r'#([A-Z]{2,}-\d+)',
        r'\b([A-Z]{2,}\d{4,})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text."""
    if not text:
        return []

    mentions = re.findall(r'@([a-zA-Z0-9_.-]+)', text)
    slack_ids = re.findall(r'<@([A-Z0-9]+)>', text)

    return list(set(mentions + slack_ids))


def extract_priority_score(text: str) -> Optional[int]:
    """Extract priority score from text."""
    if not text:
        return None

    match = re.search(r'[Pp]riority(?:\s*[Ss]core)?[:= ]\s*(\d)', text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score

    match = re.search(r'\b[Pp]([1-5])\b', text)
    if match:
        return int(match.group(1))

    return None


def extract_recommendation_type(text: str) -> Optional[str]:
    """Extract recommendation type from text."""
    if not text:
        return None

    text_lower = text.lower()

    for rec_type, keywords in RECOMMENDATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return rec_type

    return None


def find_recommendation_turn(conversation: Conversation) -> Optional[int]:
    """Find the turn index where a recommendation was made."""
    if not conversation or not conversation.messages:
        return None

    for i, msg in enumerate(conversation.messages):
        if isinstance(msg, AIMessage) and msg.content:
            rec_type = extract_recommendation_type(msg.content)
            if rec_type:
                return i

    return None


def analyze_reactions(reactions: Optional[Dict[str, int]]) -> ReactionSignals:
    """Analyze Slack reactions to extract sentiment signals."""
    if not reactions:
        return ReactionSignals()

    thumbs_up = 0
    thumbs_down = 0
    positive_count = 0
    negative_count = 0

    for reaction, count in reactions.items():
        normalized = reaction.strip(':').lower()

        if normalized in ('thumbsup', '+1', 'thumbs_up'):
            thumbs_up += count
            positive_count += count
        elif normalized in ('thumbsdown', '-1', 'thumbs_down'):
            thumbs_down += count
            negative_count += count
        elif normalized in POSITIVE_REACTIONS:
            positive_count += count
        elif normalized in NEGATIVE_REACTIONS:
            negative_count += count

    total = positive_count + negative_count
    sentiment_score = positive_count / total if total > 0 else 0.5

    return ReactionSignals(
        thumbs_up_count=thumbs_up,
        thumbs_down_count=thumbs_down,
        has_positive_reaction=positive_count > 0,
        has_negative_reaction=negative_count > 0,
        reaction_sentiment_score=round(sentiment_score, 2),
        all_reactions=reactions,
    )


def detect_stalemate(
    messages: List[BaseMessage],
    similarity_threshold: float = 0.9,
    min_repeats: int = 2,
) -> StalemateSignals:
    """Detect if the bot is stuck in a stalemate (repeating same message)."""
    if not messages:
        return StalemateSignals()

    ai_messages = [
        (i, m) for i, m in enumerate(messages) if isinstance(m, AIMessage) and m.content
    ]

    if len(ai_messages) < min_repeats:
        return StalemateSignals()

    content_counts: Dict[str, List[int]] = {}

    for idx, msg in ai_messages:
        normalized = _normalize_for_comparison(msg.content)
        content_counts.setdefault(normalized, []).append(idx)

    for content, indices in content_counts.items():
        if len(indices) >= min_repeats and _are_indices_clustered(indices, max_gap=3):
            return StalemateSignals(
                is_stalemate=True,
                repeated_message_count=len(indices),
                stalemate_turn_index=indices[0],
                repeated_content=content[:200],
            )

    return StalemateSignals()


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ''
    normalized = ' '.join(text.lower().split())
    normalized = re.sub(
        r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b', '[TIMESTAMP]', normalized
    )
    normalized = re.sub(
        r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b',
        '[UUID]',
        normalized,
    )
    return normalized


def _are_indices_clustered(indices: List[int], max_gap: int = 3) -> bool:
    """Check if indices are clustered together."""
    if len(indices) < 2:
        return True
    sorted_indices = sorted(indices)
    return all(
        sorted_indices[i] - sorted_indices[i - 1] <= max_gap
        for i in range(1, len(sorted_indices))
    )


def calculate_time_to_resolution(messages: List[BaseMessage]) -> Optional[int]:
    """Calculate time to resolution from message timestamps."""
    if not messages or len(messages) < 2:
        return None

    first_ts = _extract_timestamp(messages[0])
    last_ts = _extract_timestamp(messages[-1])

    if first_ts is None or last_ts is None:
        return None

    return int((last_ts - first_ts).total_seconds())


def _extract_timestamp(message: BaseMessage) -> Optional[datetime]:
    """Extract timestamp from message metadata."""
    if not message.metadata:
        return None

    ts_value = message.metadata.get('ts') or message.metadata.get('timestamp')
    if ts_value is None:
        return None

    if isinstance(ts_value, str):
        try:
            if '.' in ts_value:
                return datetime.fromtimestamp(float(ts_value))
            return datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
        except (ValueError, OSError):
            return None

    if isinstance(ts_value, (int, float)):
        try:
            return datetime.fromtimestamp(ts_value)
        except (ValueError, OSError):
            return None

    return None


__all__ = [
    'SlackMetadata',
    'ReactionSignals',
    'StalemateSignals',
    'parse_slack_metadata',
    'get_ai_messages',
    'get_human_messages',
    'build_transcript',
    'count_questions',
    'extract_case_id',
    'extract_mentions',
    'extract_priority_score',
    'extract_recommendation_type',
    'find_recommendation_turn',
    'analyze_reactions',
    'detect_stalemate',
    'calculate_time_to_resolution',
    'POSITIVE_REACTIONS',
    'NEGATIVE_REACTIONS',
]
