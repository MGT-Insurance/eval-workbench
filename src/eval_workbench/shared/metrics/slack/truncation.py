"""Token counting and conversation truncation utilities."""

from typing import List, Optional, Tuple

from axion._core.schema import BaseMessage, HumanMessage

from eval_workbench.shared.metrics.slack.config import TruncationConfig

# Try to import tiktoken for accurate token counting, fallback to estimation
try:
    import tiktoken

    _encoder = tiktoken.get_encoding('cl100k_base')
    HAS_TIKTOKEN = True
except ImportError:
    _encoder = None
    HAS_TIKTOKEN = False


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses tiktoken if available (accurate), otherwise falls back to
    character-based estimation (~4 characters per token).
    """
    if not text:
        return 0

    if HAS_TIKTOKEN and _encoder is not None:
        return len(_encoder.encode(text))

    return len(text) // 4


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate token count for a message including role overhead."""
    content = message.content or ''
    return estimate_tokens(content) + 4


def truncate_conversation(
    messages: List[BaseMessage],
    config: TruncationConfig,
    recommendation_turn: Optional[int] = None,
) -> Tuple[List[BaseMessage], str]:
    """
    Apply Head-Tail-Target truncation strategy to a conversation.
    """
    if not messages:
        return [], 'No messages to truncate.'

    total_messages = len(messages)

    total_tokens = sum(estimate_message_tokens(m) for m in messages)
    if total_tokens <= config.max_tokens:
        return messages, ''

    keep_indices = set()

    head_end = min(config.head_turns, total_messages)
    keep_indices.update(range(head_end))

    tail_start = max(0, total_messages - config.tail_turns)
    keep_indices.update(range(tail_start, total_messages))

    if recommendation_turn is not None and 0 <= recommendation_turn < total_messages:
        target_start = max(0, recommendation_turn - config.target_context)
        target_end = min(
            total_messages, recommendation_turn + config.target_context + 1
        )
        keep_indices.update(range(target_start, target_end))

    sorted_indices = sorted(keep_indices)
    truncated = [messages[i] for i in sorted_indices]
    truncated_tokens = sum(estimate_message_tokens(m) for m in truncated)

    while (
        truncated_tokens > config.max_tokens
        and len(truncated) > config.head_turns + config.tail_turns
    ):
        middle_start = config.head_turns
        middle_end = len(truncated) - config.tail_turns

        if middle_start < middle_end:
            remove_idx = (middle_start + middle_end) // 2
            removed_msg = truncated.pop(remove_idx)
            sorted_indices.pop(remove_idx)
            truncated_tokens -= estimate_message_tokens(removed_msg)
        else:
            break

    omitted_count = total_messages - len(truncated)
    if omitted_count > 0:
        gaps = []
        prev_idx = -1
        for idx in sorted_indices:
            if prev_idx >= 0 and idx > prev_idx + 1:
                gap_start = prev_idx + 1
                gap_end = idx - 1
                gap_count = gap_end - gap_start + 1
                gaps.append(
                    f'turns {gap_start + 1}-{gap_end + 1} ({gap_count} messages)'
                )
            prev_idx = idx

        summary_parts = [
            f'[TRUNCATED: {omitted_count} of {total_messages} messages omitted to fit context window]'
        ]
        if gaps:
            summary_parts.append(f'Omitted sections: {"; ".join(gaps)}')
        summary_parts.append('Head, tail, and recommendation context preserved.')
        summary = '\n'.join(summary_parts)
    else:
        summary = ''

    return truncated, summary


def get_truncation_markers(original_count: int, truncated_indices: List[int]):
    """Identify gaps in the truncated conversation for marker insertion."""
    gaps = []
    prev_idx = -1
    for idx in sorted(truncated_indices):
        if prev_idx >= 0 and idx > prev_idx + 1:
            gaps.append((prev_idx + 1, idx - 1))
        prev_idx = idx
    return gaps


def format_truncated_transcript(
    messages: List[BaseMessage],
    truncation_summary: str,
    include_turn_numbers: bool = True,
) -> str:
    """Format truncated messages into a transcript string."""
    lines: List[str] = []
    if truncation_summary:
        lines.append(truncation_summary)
        lines.append('')

    for i, msg in enumerate(messages):
        role = 'Human' if isinstance(msg, HumanMessage) else 'AI'
        content = msg.content or '[empty]'
        if include_turn_numbers:
            lines.append(f'[Turn {i + 1}] {role}: {content}')
        else:
            lines.append(f'{role}: {content}')

    return '\n'.join(lines)


__all__ = [
    'estimate_tokens',
    'estimate_message_tokens',
    'truncate_conversation',
    'get_truncation_markers',
    'format_truncated_transcript',
    'HAS_TIKTOKEN',
]
