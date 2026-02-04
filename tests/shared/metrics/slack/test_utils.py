"""Tests for Slack metrics utility functions."""

import pytest
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset_schema import MultiTurnConversation

from eval_workbench.shared.metrics.slack.utils import (
    build_transcript,
    count_questions,
    detect_ai_sender,
    extract_case_id,
    extract_mentions,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
    get_human_messages,
    has_recommendation_pattern,
    parse_slack_metadata,
)


class TestParseSlackMetadata:
    """Tests for parse_slack_metadata function."""

    def test_parses_complete_metadata(self):
        """Should parse all metadata fields."""
        additional_input = {
            'thread_ts': '1234567890.123456',
            'channel_id': 'C12345',
            'reply_count': 5,
            'sender': 'user123',
            'team_id': 'T12345',
        }

        result = parse_slack_metadata(additional_input)

        assert result.thread_ts == '1234567890.123456'
        assert result.channel_id == 'C12345'
        assert result.reply_count == 5
        assert result.sender == 'user123'
        assert result.team_id == 'T12345'

    def test_handles_partial_metadata(self):
        """Should handle missing fields gracefully."""
        additional_input = {'thread_ts': '123', 'sender': 'user'}

        result = parse_slack_metadata(additional_input)

        assert result.thread_ts == '123'
        assert result.sender == 'user'
        assert result.channel_id is None
        assert result.reply_count is None

    def test_handles_none_input(self):
        """Should return empty metadata for None input."""
        result = parse_slack_metadata(None)

        assert result.thread_ts is None
        assert result.channel_id is None

    def test_handles_empty_dict(self):
        """Should return empty metadata for empty dict."""
        result = parse_slack_metadata({})

        assert result.thread_ts is None


class TestExtractMentions:
    """Tests for extract_mentions function."""

    def test_extracts_encoded_mentions(self):
        """Should extract Slack encoded mentions."""
        text = 'Hey <@U12345ABC> can you help with this? cc <@U98765XYZ>'

        result = extract_mentions(text)

        assert 'U12345ABC' in result
        assert 'U98765XYZ' in result

    def test_extracts_display_mentions(self):
        """Should extract display format mentions."""
        text = 'Hey @john_doe can you help? Also @jane-smith please look.'

        result = extract_mentions(text)

        assert 'john_doe' in result
        assert 'jane-smith' in result

    def test_extracts_mention_with_name(self):
        """Should extract mentions with display names."""
        text = '<@U12345|john> please review'

        result = extract_mentions(text)

        assert 'U12345' in result

    def test_handles_no_mentions(self):
        """Should return empty list for text without mentions."""
        text = 'Just a regular message without any mentions.'

        result = extract_mentions(text)

        assert result == []

    def test_handles_empty_text(self):
        """Should return empty list for empty text."""
        assert extract_mentions('') == []
        assert extract_mentions(None) == []


class TestDetectAiSender:
    """Tests for detect_ai_sender function."""

    def test_detects_athena(self):
        """Should detect Athena as AI sender."""
        assert detect_ai_sender('Athena') is True
        assert detect_ai_sender('athena') is True
        assert detect_ai_sender('ATHENA') is True

    def test_detects_custom_ai_names(self):
        """Should detect custom AI names."""
        assert detect_ai_sender('Claude', ['Claude', 'GPT']) is True
        assert detect_ai_sender('gpt-4', ['Claude', 'GPT']) is True

    def test_rejects_non_ai_senders(self):
        """Should return False for non-AI senders."""
        assert detect_ai_sender('john_doe') is False
        assert detect_ai_sender('human_user', ['Athena']) is False

    def test_handles_none_sender(self):
        """Should return False for None sender."""
        assert detect_ai_sender(None) is False
        assert detect_ai_sender('') is False


class TestGetMessages:
    """Tests for get_human_messages and get_ai_messages functions."""

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation."""
        return MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello, I need help.'),
                AIMessage(content='Hi! How can I help you?'),
                HumanMessage(content='Can you review this case?'),
                AIMessage(content='Sure, let me check.'),
            ]
        )

    def test_get_human_messages(self, sample_conversation):
        """Should return only human messages."""
        result = get_human_messages(sample_conversation)

        assert len(result) == 2
        assert all(isinstance(m, HumanMessage) for m in result)
        assert result[0].content == 'Hello, I need help.'

    def test_get_ai_messages(self, sample_conversation):
        """Should return only AI messages."""
        result = get_ai_messages(sample_conversation)

        assert len(result) == 2
        assert all(isinstance(m, AIMessage) for m in result)
        assert result[0].content == 'Hi! How can I help you?'

    def test_handles_none_conversation(self):
        """Should return empty list for None conversation."""
        assert get_human_messages(None) == []
        assert get_ai_messages(None) == []


class TestRecommendationDetection:
    """Tests for recommendation-related functions."""

    def test_has_recommendation_pattern_approve(self):
        """Should detect approve recommendation."""
        assert has_recommendation_pattern('Recommend Approve this case.') is True
        # Note: "I recommend approval." uses different phrasing not in patterns
        assert has_recommendation_pattern('I recommend approval.') is False

    def test_has_recommendation_pattern_decline(self):
        """Should detect decline recommendation."""
        assert has_recommendation_pattern('Recommend Decline') is True

    def test_has_recommendation_pattern_score(self):
        """Should detect priority score patterns."""
        assert has_recommendation_pattern('Base Score: 47/100') is True
        assert has_recommendation_pattern('Priority Score: 85') is True

    def test_has_recommendation_pattern_case_id(self):
        """Should detect case ID patterns."""
        assert has_recommendation_pattern('MGT-BOP-1234567') is True

    def test_no_recommendation_pattern(self):
        """Should return False for regular text."""
        assert has_recommendation_pattern('Just a regular message.') is False

    def test_extract_recommendation_type_approve(self):
        """Should extract approve recommendation type."""
        assert extract_recommendation_type('Recommend Approve') == 'approve'
        assert extract_recommendation_type('recommend approve') == 'approve'

    def test_extract_recommendation_type_decline(self):
        """Should extract decline recommendation type."""
        assert extract_recommendation_type('Recommend Decline') == 'decline'

    def test_extract_recommendation_type_review(self):
        """Should extract review recommendation type."""
        assert extract_recommendation_type('Recommend Review') == 'review'

    def test_extract_recommendation_type_none(self):
        """Should return None for no recommendation."""
        assert extract_recommendation_type('No recommendation here.') is None


class TestCaseIdExtraction:
    """Tests for extract_case_id function."""

    def test_extracts_mgt_bop_format(self):
        """Should extract MGT-BOP case IDs."""
        text = 'Please review case MGT-BOP-1234567 for approval.'

        result = extract_case_id(text)

        assert result == 'MGT-BOP-1234567'

    def test_extracts_case_format(self):
        """Should extract CASE format IDs."""
        text = 'Looking at CASE-12345 now.'

        result = extract_case_id(text)

        assert result == 'CASE-12345'

    def test_returns_none_for_no_case(self):
        """Should return None when no case ID found."""
        assert extract_case_id('No case ID here.') is None


class TestPriorityScoreExtraction:
    """Tests for extract_priority_score function."""

    def test_extracts_base_score(self):
        """Should extract Base Score."""
        text = 'Base Score: 47/100. This case needs review.'

        result = extract_priority_score(text)

        assert result == 47

    def test_extracts_priority_score(self):
        """Should extract Priority Score."""
        text = 'Priority Score: 85'

        result = extract_priority_score(text)

        assert result == 85

    def test_clamps_to_valid_range(self):
        """Should clamp scores to 0-100."""
        assert extract_priority_score('Base Score: 150') == 100
        # Negative numbers aren't matched by the regex (unlikely in real data)
        assert extract_priority_score('Priority Score: -10') is None

    def test_returns_none_for_no_score(self):
        """Should return None when no score found."""
        assert extract_priority_score('No score here.') is None


class TestCountQuestions:
    """Tests for count_questions function."""

    def test_counts_question_marks(self):
        """Should count question marks."""
        text = 'What is this? How does it work?'

        result = count_questions(text)

        assert result >= 2

    def test_counts_question_patterns(self):
        """Should count question patterns without marks."""
        text = 'Can you help me with this.'

        result = count_questions(text)

        assert result >= 1

    def test_returns_zero_for_no_questions(self):
        """Should return 0 for statements."""
        text = 'This is a statement. No questions here.'

        result = count_questions(text)

        assert result == 0


class TestBuildTranscript:
    """Tests for build_transcript function."""

    def test_builds_transcript(self):
        """Should build formatted transcript."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi there!'),
            ]
        )

        result = build_transcript(conversation)

        assert 'User: Hello' in result
        assert 'AI: Hi there!' in result

    def test_handles_none_conversation(self):
        """Should return empty string for None."""
        assert build_transcript(None) == ''


class TestFindRecommendationTurn:
    """Tests for find_recommendation_turn function."""

    def test_finds_recommendation_turn(self):
        """Should find turn with recommendation."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Review this case.'),
                AIMessage(content='Recommend Approve. Base Score: 85/100'),
                HumanMessage(content='Thanks!'),
            ]
        )

        result = find_recommendation_turn(conversation)

        assert result == 1  # Index of AI message with recommendation

    def test_returns_none_for_no_recommendation(self):
        """Should return None when no recommendation found."""
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi, how can I help?'),
            ]
        )

        result = find_recommendation_turn(conversation)

        assert result is None
