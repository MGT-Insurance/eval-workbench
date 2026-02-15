# Heuristic Analyzer

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Zero-cost pattern matching — no LLM required.</strong> Computes interaction metrics, engagement depth, recommendation detection, emoji reaction sentiment, and bot message repetition (stalemate).
</p>
</div>

<div class="at-a-glance">
<div class="at-a-glance__header">
<span class="at-a-glance__title">At a Glance</span>
<span class="at-a-glance__badge at-a-glance__badge--heuristic">No LLM</span>
</div>
<div class="at-a-glance__body">
<div class="at-a-glance__item">
<span class="at-a-glance__label">Class</span>
<span class="at-a-glance__value"><code>SlackHeuristicAnalyzer</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Base</span>
<span class="at-a-glance__value"><code>BaseMetric</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Source</span>
<span class="at-a-glance__value"><code>shared/metrics/slack/heuristic.py</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Temperature</span>
<span class="at-a-glance__value">N/A</span>
</div>
<div class="at-a-glance__item at-a-glance__item--wide">
<span class="at-a-glance__label">Signals</span>
<span class="at-a-glance__value"><code>InteractionSignals</code> · <code>EngagementSignals</code> · <code>RecommendationSignals</code> · <code>ReactionSignals</code> · <code>StalemateSignals</code></span>
</div>
</div>
</div>

---

## Constructor

```python
from eval_workbench.shared.metrics.slack.heuristic import SlackHeuristicAnalyzer

analyzer = SlackHeuristicAnalyzer(
    config=None,                # Optional[AnalyzerConfig]
    include_reactions=True,     # Analyze emoji reactions
    include_stalemate=True,     # Detect repeated bot messages
)
```

---

## Signals

### InteractionSignals

| Signal | Type | Description |
|--------|------|-------------|
| `ai_message_count` | int | Number of AI messages |
| `human_message_count` | int | Number of human messages |
| `total_turn_count` | int | Total messages in thread |
| `reply_count` | int | Number of replies |
| `is_ai_initiated` | bool | Whether AI sent first message |
| `has_human_response` | bool | Whether humans responded to AI |
| `is_interactive` | bool | Has both AI and human participation |

### EngagementSignals

| Signal | Type | Description |
|--------|------|-------------|
| `interaction_depth` | int | Number of back-and-forth exchanges |
| `has_multiple_interactions` | bool | More than one human message |
| `avg_human_response_length` | float | Average human message length (chars) |
| `avg_ai_response_length` | float | Average AI message length (chars) |
| `question_count` | int | Total questions asked |
| `mention_count` | int | Total @mentions |
| `unique_participants` | int | Count of unique human participants |

### RecommendationSignals

| Signal | Type | Description |
|--------|------|-------------|
| `has_recommendation` | bool | Whether AI made a recommendation |
| `recommendation_type` | str | Type: approve, decline, review, hold, none |
| `recommendation_turn_index` | int | Turn where recommendation was made |
| `recommendation_confidence` | float | Extracted confidence level (0-1) |
| `case_id` | str | Case identifier (e.g., MGT-BOP-123456) |
| `case_priority` | int | Priority/base score (0-100) |

### ReactionSignals (optional)

Computed when `include_reactions=True`. Analyzes emoji reactions on messages.

### StalemateSignals (optional)

Computed when `include_stalemate=True`. Detects repeated/identical bot messages that may indicate a stuck conversation loop.

---

## Sub-Metrics

<div class="sub-metrics">
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--score">0–1</span>
<span class="sub-metric__text">
<span class="sub-metric__name">interaction</span>
<span class="sub-metric__desc">Interaction score from message counts</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--score">0–1</span>
<span class="sub-metric__text">
<span class="sub-metric__name">engagement</span>
<span class="sub-metric__desc">Engagement depth score</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">recommendation_type</span>
<span class="sub-metric__desc">Detected recommendation type</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--score">0–1</span>
<span class="sub-metric__text">
<span class="sub-metric__name">reaction_sentiment</span>
<span class="sub-metric__desc">Reaction sentiment (if enabled)</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">is_stalemate</span>
<span class="sub-metric__desc">Bot repeating (if enabled)</span>
</span>
</div>
</div>

---

## Usage

```python
from eval_workbench.shared.metrics.slack.heuristic import SlackHeuristicAnalyzer

analyzer = SlackHeuristicAnalyzer()
result = await analyzer.execute(dataset_item)

# Access signals
signals = result.signals
print(f"Interactive: {signals.interaction.is_interactive}")
print(f"Depth: {signals.engagement.interaction_depth}")
print(f"Recommendation: {signals.recommendation.recommendation_type}")

# Get sub-metrics for reporting
sub_metrics = analyzer.get_sub_metrics(result)
```

---

## KPIs Supported

- `interaction_rate` — Interactive threads / Total eligible cases
- `MAU` — Unique senders in 30 days
- `engagement_rate` — Avg interactions per case or % with multiple interactions
- `acceptance_rate` / `override_rate` — Via recommendation detection (foundation for downstream analyzers)
