# Slack Metrics

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Slack KPIs and conversation analytics</strong><br>
<span class="badge" style="margin-top: 0.5rem;">13 Metrics</span>
<span class="badge" style="background: #667eea;">Multi-turn</span>
</div>

This module provides metrics for analyzing Slack conversations between users and
AI assistants (e.g., Athena). These metrics support KPI computation for
measuring AI assistant effectiveness in underwriting workflows.

---

## Slack KPI Metrics

<div class="grid-container">

<div class="grid-item">
<strong><a href="composite/">Slack Conversation Analyzer</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">All KPI signals in one pass</p>
<code>conversation</code> <code>additional_input</code>
</div>

<div class="grid-item">
<strong><a href="interaction/">Slack Interaction Analyzer</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Interaction rate and MAU signals</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="engagement/">Thread Engagement Analyzer</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Engagement depth and response signals</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="recommendation/">Recommendation Analyzer</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Recommendation extraction for KPIs</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="acceptance/">Acceptance Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Accepted vs rejected recommendations</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="override/">Override Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Human overrides of AI recommendations</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="override_satisfaction/">Override Satisfaction Analyzer</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Quality of override explanations</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="escalation/">Escalation Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Detect escalations to humans</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="frustration/">Frustration Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Score user frustration levels</p>
<code>conversation</code>
</div>

<div class="grid-item">
<strong><a href="intervention/">Intervention Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Detect human intervention and escalation type</p>
<code>conversation</code> <code>additional_input</code>
</div>

<div class="grid-item">
<strong><a href="sentiment/">Sentiment Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Positive/neutral/frustrated/confused sentiment</p>
<code>conversation</code> <code>additional_input</code>
</div>

<div class="grid-item">
<strong><a href="resolution/">Resolution Detector</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Outcome: approved/declined/stalemate/pending</p>
<code>conversation</code> <code>additional_input</code>
</div>

<div class="grid-item">
<strong><a href="slack_compliance/">Slack Formatting Compliance</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Slack mrkdwn formatting checks</p>
<code>actual_output</code>
</div>

</div>

## Overview

| Metric | Type | Description | KPI |
|--------|------|-------------|-----|
| `SlackConversationAnalyzer` | Composite | All-in-one analysis combining all metrics below | All KPIs |
| `SlackInteractionAnalyzer` | Heuristic | Message counts and interaction detection | interaction_rate, MAU |
| `ThreadEngagementAnalyzer` | Heuristic | Engagement depth and quality signals | engagement_rate |
| `RecommendationAnalyzer` | Heuristic | AI recommendation extraction | acceptance_rate, override_rate |
| `EscalationDetector` | LLM | Detects escalation to human team members | escalation_rate |
| `FrustrationDetector` | LLM | Scores user frustration level | frustration_rate |
| `AcceptanceDetector` | LLM | Determines if recommendations were accepted | acceptance_rate |
| `OverrideDetector` | LLM | Detects when humans override AI recommendations | override_rate |
| `OverrideSatisfactionAnalyzer` | LLM | Scores quality of override explanations | override_satisfaction |
| `InterventionDetector` | LLM | Detects human intervention and escalation type | intervention_rate, stp_rate |
| `SentimentDetector` | LLM | Detects user sentiment (positive/neutral/frustrated/confused) | sentiment distribution, frustration_rate |
| `ResolutionDetector` | LLM | Detects thread outcome (approved/declined/blocked/stalemate) | resolution_rate, stalemate_rate |
| `SlackFormattingCompliance` | Heuristic | Validates Slack mrkdwn formatting | - |

## Metrics Detail

### SlackConversationAnalyzer (Composite)

**File:** `composite.py`

**Purpose:** Perform comprehensive analysis of a Slack conversation in a single
pass, extracting all KPI-relevant signals efficiently.

**What it computes:**
- Counts AI and human messages to determine if the thread is interactive
- Calculates engagement depth (number of back-and-forth exchanges)
- Detects AI recommendations and extracts case metadata (ID, priority score)
- Uses LLM to classify escalation, frustration, acceptance, override, satisfaction, intervention, sentiment, and resolution
- Combines heuristic preprocessing with a single LLM call for efficiency
- Produces a nested result structure that can be expanded into 11 separate metric rows for reporting

**Signals produced:**
- `interaction` - Message counts, AI/human participation
- `engagement` - Interaction depth, response lengths, questions
- `recommendation` - AI recommendation type, case ID, priority
- `escalation` - Whether escalated, escalation type/reason
- `frustration` - Frustration score (0-1), indicators, cause
- `acceptance` - Whether recommendation was accepted
- `override` - Whether recommendation was overridden
- `satisfaction` - Quality of override explanation
- `intervention` - Human intervention type, escalation class, STP
- `sentiment` - Overall user sentiment and score
- `resolution` - Final outcome and stale threads

**Usage:**
```python
from shared.metrics.slack.composite import SlackConversationAnalyzer

analyzer = SlackConversationAnalyzer(
    frustration_threshold=0.6,
    satisfaction_threshold=0.7
)
result = await analyzer.execute(dataset_item)

# Convert to rows for reporting
rows = result.signals.to_rows()

# Get KPI summary
kpi = result.signals.to_kpi_summary()
```

---

### SlackInteractionAnalyzer

**File:** `interaction.py`

**Purpose:** Determine whether a Slack thread represents a meaningful AI
interaction and track unique users for MAU calculation.

**What it computes:**
- Counts the number of AI messages and human messages in a thread
- Determines if the thread is "interactive" (has both AI and human participation)
- Identifies who initiated the conversation (AI or human)
- Extracts thread metadata (thread_id, channel_id, sender) for aggregation
- Does NOT produce a numeric score - outputs structured signals for KPI computation

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `is_interactive` | bool | Has both AI and human participation |
| `ai_message_count` | int | Number of AI messages |
| `human_message_count` | int | Number of human messages |
| `total_turn_count` | int | Total messages in thread |
| `is_ai_initiated` | bool | Whether AI sent first message |
| `has_human_response` | bool | Whether humans responded to AI |
| `thread_id` | str | Slack thread timestamp |
| `channel_id` | str | Slack channel ID |
| `sender` | str | Original sender (for MAU tracking) |

**KPIs supported:**

- `interaction_rate` = Interactive threads / Total eligible cases
- `MAU` = Unique senders in 30 days

---

### ThreadEngagementAnalyzer

**File:** `engagement.py`

**Purpose:** Measure how deeply users engage with the AI assistant beyond simple
one-off interactions.

**What it computes:**
- Calculates interaction depth by counting back-and-forth exchanges (human→AI→human = 1 exchange)
- Determines if the thread has multiple human interactions (more than one human message)
- Measures average response lengths for both human and AI messages
- Counts total questions asked in the thread (indicates information-seeking behavior)
- Counts @mentions (indicates collaboration or escalation)
- Estimates unique human participants in the thread

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `interaction_depth` | int | Number of back-and-forth exchanges |
| `has_multiple_interactions` | bool | More than one human message |
| `avg_human_response_length` | float | Average human message length (chars) |
| `avg_ai_response_length` | float | Average AI message length (chars) |
| `question_count` | int | Total questions asked |
| `mention_count` | int | Total @mentions |
| `unique_participants` | int | Count of unique human participants |

**KPIs supported:**

- `engagement_rate` = Avg interactions per case or % with multiple interactions

---

### RecommendationAnalyzer

**File:** `recommendation.py`

**Purpose:** Extract and classify AI recommendations from conversations to enable
acceptance and override analysis.

**What it computes:**
- Scans AI messages for recommendation patterns ("Recommend Approve", "Recommend Decline", etc.)
- Classifies recommendation type: approve, decline, review, hold, or none
- Extracts the turn index where the recommendation was made
- Parses confidence levels if present ("high confidence", "confidence: 85%")
- Extracts case identifiers (MGT-BOP-XXXXXXX format)
- Extracts priority/base scores from "Base Score: XX/100" patterns
- Provides the foundation for AcceptanceDetector and OverrideDetector

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `has_recommendation` | bool | Whether AI made a recommendation |
| `recommendation_type` | str | Type: approve, decline, review, hold, none |
| `recommendation_turn_index` | int | Turn where recommendation was made |
| `recommendation_confidence` | float | Extracted confidence level (0-1) |
| `case_id` | str | Case identifier (e.g., MGT-BOP-123456) |
| `case_priority` | int | Priority/base score (0-100) |

**Patterns detected:**
- "Recommend Approve/Decline/Review/Hold"
- "Base Score: XX/100"
- "Priority Score: XX"
- Case IDs: MGT-BOP-XXXXXXX

---

### EscalationDetector

**File:** `escalation.py`

**Purpose:** Identify when a conversation moves beyond AI assistance to require
human team member involvement.

**What it computes:**
- Uses LLM to analyze conversation context and detect escalation patterns
- Classifies the type of escalation (team mention, explicit handoff, error, complexity)
- Identifies the turn where escalation occurred
- Extracts @mentioned users who were brought into the conversation
- Determines the reason for escalation
- Falls back to heuristic pattern matching if LLM fails (detects @mentions after AI interaction, AI error messages)
- Returns score of 1.0 if escalated, 0.0 if not

**Escalation Types:**

| Type | Description |
|------|-------------|
| `no_escalation` | Normal AI-handled conversation |
| `team_mention` | User @mentioned team members |
| `explicit_handoff` | AI explicitly handed off to human |
| `error_escalation` | Escalation due to AI error |
| `complexity_escalation` | Escalation due to case complexity |

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `is_escalated` | bool | Whether conversation was escalated |
| `escalation_type` | str | Type of escalation |
| `escalation_turn_index` | int | Turn where escalation occurred |
| `escalation_targets` | list | @mentioned users during escalation |
| `escalation_reason` | str | Reason for escalation |

**KPIs supported:**

- `escalation_rate` = Escalated cases / Total AI cases

---

### FrustrationDetector

**File:** `frustration.py`

**Purpose:** Measure user satisfaction by detecting frustration signals in their
messages.

**What it computes:**
- Uses LLM to analyze human messages for frustration indicators
- Produces a continuous frustration score from 0.0 (calm) to 1.0 (very frustrated)
- Identifies specific frustration indicators (repeated questions, ALL CAPS, multiple punctuation)
- Determines the primary cause of frustration (AI error, wrong answer, poor understanding, etc.)
- Finds the turn where frustration peaked
- Classifies as "frustrated" if score >= threshold (default 0.6)
- Falls back to heuristic pattern matching if LLM fails

**Score interpretation:**

| Score | Level |
|-------|-------|
| 0.0-0.2 | Calm, positive interaction |
| 0.2-0.4 | Mild impatience |
| 0.4-0.6 | Moderate frustration |
| 0.6-0.8 | Significant frustration |
| 0.8-1.0 | Severe frustration |

**Frustration causes:**
- `ai_error` - AI made a mistake
- `slow_response` - Response time complaints
- `wrong_answer` - Incorrect information
- `repeated_questions` - User had to repeat themselves
- `poor_understanding` - AI didn't understand
- `system_issue` - Technical problems

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `frustration_score` | float | Overall score (0-1) |
| `is_frustrated` | bool | Score >= threshold (default 0.6) |
| `frustration_indicators` | list | Detected signals |
| `peak_frustration_turn` | int | Turn with highest frustration |
| `frustration_cause` | str | Primary cause |

**KPIs supported:**

- `frustration_rate` = Frustrated interactions / Total interactions

---

### AcceptanceDetector

**File:** `acceptance.py`

**Purpose:** Determine whether users accepted or rejected AI recommendations to
measure recommendation quality.

**What it computes:**
- First checks if a recommendation exists in the conversation (uses RecommendationAnalyzer patterns)
- Uses LLM to analyze post-recommendation messages for acceptance signals
- Classifies acceptance status: accepted, accepted_with_discussion, pending, rejected, modified
- Identifies who made the final decision (if detectable)
- Calculates turns between recommendation and decision
- Maps acceptance status to a score: accepted=1.0, accepted_with_discussion=0.8, modified=0.5, pending/rejected=0.0
- Only runs analysis if a recommendation is found

**Acceptance statuses:**

| Status | Description |
|--------|-------------|
| `accepted` | Accepted without change |
| `accepted_with_discussion` | Accepted after team discussion |
| `pending` | No clear resolution |
| `rejected` | Explicitly rejected |
| `modified` | Accepted with modifications |

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `acceptance_status` | str | Status from above |
| `is_accepted` | bool | accepted or accepted_with_discussion |
| `acceptance_turn_index` | int | Turn where decision was made |
| `decision_maker` | str | Who made the final decision |
| `turns_to_decision` | int | Turns between recommendation and decision |

**KPIs supported:**

- `acceptance_rate` = Recommendations accepted / Total recommendations

---

### OverrideDetector

**File:** `override.py`

**Purpose:** Detect when human underwriters make decisions that contradict AI
recommendations, indicating potential AI improvement opportunities.

**What it computes:**
- First checks if a recommendation exists in the conversation
- Uses LLM to analyze if the human's final decision contradicts the AI recommendation
- Classifies override type: no_override, full_override (opposite decision), partial_override (modified), pending_override
- Captures the original AI recommendation and the actual final decision
- Extracts the stated reason for the override from the conversation
- Categorizes override reasons (additional_info, risk_assessment, policy_exception, etc.)
- Returns score of 1.0 if overridden, 0.0 if not
- Only runs analysis if a recommendation is found

**Override types:**

| Type | Description |
|------|-------------|
| `no_override` | AI recommendation followed |
| `full_override` | Completely different decision |
| `partial_override` | Modified recommendation |
| `pending_override` | Discussion suggests override, not confirmed |

**Override reason categories:**
- `additional_info` - Based on info AI didn't have
- `risk_assessment` - Different risk evaluation
- `policy_exception` - Policy exception applied
- `class_code_issue` - Class code concerns
- `rate_issue` - Rating/pricing concerns
- `experience_judgment` - Human judgment differs

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `is_overridden` | bool | Whether recommendation was overridden |
| `override_type` | str | Type of override |
| `original_recommendation` | str | What AI recommended |
| `final_decision` | str | What was actually decided |
| `override_reason` | str | Stated reason for override |
| `override_reason_category` | str | Category from above |

**KPIs supported:**

- `override_rate` = Recommendations overridden / Total recommendations

---

### OverrideSatisfactionAnalyzer

**File:** `override.py`

**Purpose:** Evaluate whether override explanations are high-quality and
actionable for improving AI recommendations.

**What it computes:**
- First runs OverrideDetector to check if an override occurred
- Only analyzes threads where an override was detected
- Uses LLM to evaluate the quality of the override explanation on three dimensions:
  - **Clear reason**: Is there a specific, understandable justification?
  - **Supporting evidence**: Does the explanation cite specific data or information?
  - **Actionable**: Could this feedback help improve future AI recommendations?
- Produces a satisfaction score from 0.0 (poor explanation) to 1.0 (excellent explanation)
- Classifies as "satisfactory" if score >= threshold (default 0.7)
- Extracts improvement suggestions that could be used to retrain or improve the AI

**Quality criteria:**
- `has_clear_reason` - Specific, understandable justification
- `has_supporting_evidence` - Cites specific information/data
- `is_actionable` - Could help improve AI recommendations

**Score interpretation:**

| Score | Quality |
|-------|---------|
| 0.0-0.3 | Vague or no reason given |
| 0.3-0.5 | Basic reason, lacks detail |
| 0.5-0.7 | Good reason with context |
| 0.7-0.9 | Clear reason with evidence |
| 0.9-1.0 | Excellent, comprehensive |

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `satisfaction_score` | float | Quality score (0-1) |
| `is_satisfactory` | bool | Score >= threshold (default 0.7) |
| `has_clear_reason` | bool | Has stated reason |
| `has_supporting_evidence` | bool | Has evidence |
| `is_actionable` | bool | Provides actionable feedback |
| `improvement_suggestions` | list | Suggestions for AI improvement |

**KPIs supported:**

- `override_satisfaction` = Satisfactory overrides / Total overrides

---

### InterventionDetector

**File:** `intervention.py`

**Purpose:** Detect whether a human intervened in the thread and categorize the type of intervention and escalation.

**What it computes:**
- Uses LLM to classify intervention category (e.g., correction, missing context, approval)
- Maps intervention into escalation type: `hard` / `soft` / `authority` / `none`
- Flags STP (straight-through processing) when there is no human intervention
- Extracts friction point and issue details when available

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `has_intervention` | bool | Human intervened |
| `intervention_type` | str | Category of intervention |
| `escalation_type` | str | hard/soft/authority/none |
| `is_stp` | bool | Straight-through (no intervention/escalation) |
| `friction_point` | str | Concept causing friction (optional) |
| `issue_details` | str | Technical details (optional) |

**KPIs supported:**
- `intervention_rate` = Threads with intervention / Total threads
- `stp_rate` = Threads with no intervention / Total threads

---

### SentimentDetector

**File:** `sentiment.py`

**Purpose:** Measure user sentiment in the thread and provide a score (negative → positive).

**What it computes:**
- Classifies sentiment: positive / neutral / frustrated / confused
- Produces a calibrated `sentiment_score` (0.0-1.0)
- Extracts frustration indicators and peak sentiment turn when available

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `sentiment` | str | Sentiment label |
| `sentiment_score` | float | 0=negative, 1=positive |
| `is_frustrated` | bool | Frustrated flag |
| `frustration_indicators` | list | Detected negative signals |

**KPIs supported:**
- Sentiment distribution (positive/neutral/frustrated/confused)
- `frustration_rate` (via sentiment/frustrated flag)

---

### ResolutionDetector

**File:** `resolution.py`

**Purpose:** Determine how the thread ended (approved/declined/blocked/needs_info/stalemate/pending) and whether it is resolved.

**What it computes:**
- Classifies final status and resolution type
- Detects stalemate based on inactivity (when timestamps are provided in `additional_input`)
- Optionally computes time-to-resolution

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `final_status` | str | approved/declined/blocked/needs_info/stalemate/pending |
| `is_resolved` | bool | Resolved flag |
| `is_stalemate` | bool | Inactive beyond threshold |
| `time_to_resolution_seconds` | float | Time from first to last message (optional) |

**KPIs supported:**
- `resolution_rate` = Resolved / Total
- `stalemate_rate` = Stalemates / Total

---

### SlackFormattingCompliance

**File:** `slack_compliance.py`

**Purpose:** Ensure AI-generated messages use correct Slack mrkdwn formatting that
renders properly in Slack.

**What it computes:**
- Scans the AI's output text for formatting patterns that break in Slack
- Detects `**bold**` (Slack uses single `*bold*` instead)
- Detects `# Headers` (not supported in Slack mrkdwn)
- Detects unwrapped money/percentages that should be in backticks for clarity
- Returns a compliance score where 1.0 = fully compliant, lower scores indicate issues
- Lists specific formatting issues found with context snippets

**Checks performed:**
- No `**bold**` (Slack uses single `*bold*`)
- No `# Headers` (not supported in Slack)
- Money/percentages should be in backticks

**Signals:**

| Signal | Type | Description |
|--------|------|-------------|
| `score` | float | Compliance score (1.0 = fully compliant) |
| `issues` | list | List of formatting issues found |

---

## Utility Functions

**File:** `utils.py`

Common utilities used across metrics:

| Function | Description |
|----------|-------------|
| `parse_slack_metadata()` | Extract thread_ts, channel_id, sender from additional_input |
| `extract_mentions()` | Extract @mentions from message text |
| `get_human_messages()` | Filter human messages from conversation |
| `get_ai_messages()` | Filter AI messages from conversation |
| `find_recommendation_turn()` | Find turn containing AI recommendation |
| `extract_recommendation_type()` | Extract approve/decline/review/hold |
| `extract_case_id()` | Extract case ID (MGT-BOP-XXXXXXX) |
| `extract_priority_score()` | Extract base/priority score |
| `count_questions()` | Count questions in text |
| `build_transcript()` | Build plain text transcript from conversation |
