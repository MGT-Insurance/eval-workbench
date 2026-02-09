import json

import pytest
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset_schema import MultiTurnConversation

from eval_workbench.shared.metrics.slack.utils import (
    analyze_reactions,
    build_transcript,
    calculate_time_to_resolution,
    count_questions,
    detect_stalemate,
    extract_case_id,
    extract_mentions,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
    get_human_messages,
    parse_slack_metadata,
)


class TestParseSlackMetadata:
    def test_parses_additional_input(self):
        additional_input = {
            'thread_ts': '123.456',
            'channel_id': 'C123',
            'reply_count': 3,
            'sender': 'user123',
            'reactions': {'thumbsup': 2},
        }

        result = parse_slack_metadata(additional_input)

        assert result.thread_ts == '123.456'
        assert result.channel_id == 'C123'
        assert result.reply_count == 3
        assert result.sender == 'user123'
        assert result.reactions == {'thumbsup': 2}

    def test_merges_dataset_metadata(self):
        additional_input = {'sender': 'user123'}
        dataset_metadata = json.dumps(
            {'thread_ts': '999.000', 'channel_id': 'C999'}
        )

        result = parse_slack_metadata(additional_input, dataset_metadata)

        assert result.thread_ts == '999.000'
        assert result.channel_id == 'C999'
        assert result.sender == 'user123'

    def test_handles_none(self):
        result = parse_slack_metadata(None, None)

        assert result.thread_ts is None
        assert result.channel_id is None


class TestExtractMentions:
    def test_extracts_mentions(self):
        text = 'Hey @john_doe and <@U12345> please review.'
        result = extract_mentions(text)

        assert 'john_doe' in result
        assert 'U12345' in result

    def test_empty_text(self):
        assert extract_mentions('') == []
        assert extract_mentions(None) == []


class TestGetMessages:
    @pytest.fixture
    def sample_conversation(self):
        return MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi there'),
                HumanMessage(content='Thanks'),
            ]
        )

    def test_get_human_messages(self, sample_conversation):
        result = get_human_messages(sample_conversation)
        assert len(result) == 2
        assert all(isinstance(m, HumanMessage) for m in result)

    def test_get_ai_messages(self, sample_conversation):
        result = get_ai_messages(sample_conversation)
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)

    def test_handles_none(self):
        assert get_human_messages(None) == []
        assert get_ai_messages(None) == []


class TestRecommendationDetection:
    def test_extract_recommendation_type(self):
        assert extract_recommendation_type('Recommend Decline') == 'decline'
        assert extract_recommendation_type('No recommendation') is None

    def test_find_recommendation_turn(self):
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Review this.'),
                AIMessage(content='Recommend Approve.'),
                HumanMessage(content='Ok'),
            ]
        )
        assert find_recommendation_turn(conversation) == 1


class TestCaseIdExtraction:
    def test_extracts_case_id(self):
        assert extract_case_id('Case ID: ABC123') == 'ABC123'
        assert extract_case_id('No case here') is None


class TestPriorityScoreExtraction:
    def test_extracts_priority_score(self):
        assert extract_priority_score('Priority Score: 3') == 3
        assert extract_priority_score('P2') == 2
        assert extract_priority_score('No score') is None


class TestCountQuestions:
    def test_counts_questions(self):
        assert count_questions('What is this? How does it work?') >= 2
        assert count_questions('No questions.') == 0


class TestBuildTranscript:
    def test_builds_transcript(self):
        conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content='Hello'),
                AIMessage(content='Hi'),
            ]
        )
        transcript = build_transcript(conversation)
        assert '[Turn 1] Human: Hello' in transcript
        assert '[Turn 2] AI: Hi' in transcript

    def test_handles_none(self):
        assert build_transcript(None) == ''


class TestAnalyzeReactions:
    def test_analyzes_reactions(self):
        signals = analyze_reactions({'thumbsup': 2, 'thumbsdown': 1, 'heart': 1})
        assert signals.thumbs_up_count == 2
        assert signals.thumbs_down_count == 1
        assert signals.has_positive_reaction is True
        assert signals.has_negative_reaction is True
        assert 0.0 <= signals.reaction_sentiment_score <= 1.0

    def test_handles_none(self):
        signals = analyze_reactions(None)
        assert signals.thumbs_up_count == 0
        assert signals.reaction_sentiment_score == 0.5


class TestStalemateDetection:
    def test_detects_stalemate(self):
        messages = [
            AIMessage(content='Error occurred'),
            HumanMessage(content='Please try again'),
            AIMessage(content='Error occurred'),
            AIMessage(content='Error occurred'),
        ]
        signals = detect_stalemate(messages)
        assert signals.is_stalemate is True


class TestTimeToResolution:
    def test_calculates_time_to_resolution(self):
        messages = [
            HumanMessage(content='Start', metadata={'ts': '1234567890.0'}),
            AIMessage(content='End', metadata={'ts': '1234567895.0'}),
        ]
        assert calculate_time_to_resolution(messages) == 5
