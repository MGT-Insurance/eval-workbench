import re


def parse_chat_transcript(text, agent_name: str = 'Athena'):
    """
    Parses a raw string transcript into a list of dictionaries with
    'role' and 'content' keys.
    """

    text = text.replace('---', '').strip()

    pattern = rf'(?:^|\n+)({agent_name}|User):\s*'

    # re.split including a capture group returns:
    # [pre-text, delimiter, content, delimiter, content...]
    parts = re.split(pattern, text)

    messages = []
    for i in range(1, len(parts), 2):
        speaker = parts[i]  # e.g., "Athena" or "User"
        content = parts[i + 1]  # The text following the speaker

        role = 'assistant' if speaker == agent_name else 'user'

        clean_content = content.strip()

        messages.append({'role': role, 'content': clean_content})

    return messages
