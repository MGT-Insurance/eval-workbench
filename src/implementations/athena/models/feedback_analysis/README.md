# Feedback Analysis Pipeline

The Feedback Analysis Pipeline processes AI-Human underwriting conversations from Slack, extracting structured insights, computing per-conversation metrics, and aggregating them for dashboard analytics.

## Architecture

```
Raw Slack JSON Data
        │
        ▼
┌─────────────────────────────────────────────┐
│  FeedbackAnalysisPipeline._transform_to_contexts()  │
└─────────────────────────────────────────────┘
        │
        ▼
ConversationContext Objects
(parsed timestamps, human participants)
        │
        ▼
┌─────────────────────────────────────────────┐
│  UnderwritingAuditHandler.execute()         │
│  (LLM Analysis)                             │
└─────────────────────────────────────────────┘
        │
        ▼
AuditResult Objects
(intervention type, friction points, sentiment)
        │
        ▼
┌─────────────────────────────────────────────┐
│  ConversationMetricCalculator.compute()     │
└─────────────────────────────────────────────┘
        │
        ▼
ConversationMetrics Objects
(message counts, timings, ping-pong, resolution)
        │
        ▼
PipelineResult
```

### Aggregation Flow

```
PipelineResult (Multiple periods)
        │
        ▼
┌─────────────────────────────────────────────┐
│  AggregationJobRunner._fetch_period_data()  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  MetricAggregationService.aggregate_*()     │
│  (daily / weekly / monthly)                 │
└─────────────────────────────────────────────┘
        │
        ▼
AggregatedMetrics
(Trust, Escalation, Efficiency, Business Impact)
```

---

## Schema Classes

### Message

Represents a single message in a conversation thread.

```python
class Message(RichBaseModel):
    ts: str                              # Slack timestamp
    timestamp_utc: Optional[datetime]    # Parsed UTC datetime
    sender: str                          # Display name
    user_id: Optional[str]               # Slack user ID
    is_bot: bool = False                 # True if from bot
    content: str                         # Message text
    messageUrl: Optional[str]            # Direct URL to message
    reply_count: int = 0                 # Number of replies
    is_thread_reply: bool = False        # True if reply in thread
```

### ConversationContext

Represents a single underwriting thread/case to be analyzed.

```python
class ConversationContext(RichBaseModel):
    thread_id: str                           # Unique thread ID
    channel_id: Optional[str]                # Slack channel ID
    case_id: Optional[str]                   # Case ID (e.g., MGT-BOP-...)
    slack_url: Optional[str]                 # Thread URL
    title: Optional[str]                     # Business name
    messages: List[Message] = []             # Chronological messages
    thread_created_at: Optional[datetime]    # Thread creation time
    thread_last_activity_at: Optional[datetime]
    human_participants: List[str] = []       # Human user IDs for MAU
```

### Enumerations

#### InterventionType

```python
class InterventionType(str, RichEnum):
    NONE = "no_intervention"
    CORRECTION_FACTUAL = "correction_factual"
    MISSING_CONTEXT = "missing_context"
    RISK_APPETITE = "risk_appetite"
    TECH_ISSUE = "tech_issue"
    DATA_QUALITY = "data_quality"
    CLARIFICATION = "clarification"
    SUPPORT = "support"
    APPROVAL = "approval"
```

#### EscalationType

Dashboard categorization for escalation spectrum.

```python
class EscalationType(str, RichEnum):
    HARD = "hard"         # CORRECTION_FACTUAL, TECH_ISSUE, DATA_QUALITY
    SOFT = "soft"         # MISSING_CONTEXT, RISK_APPETITE, CLARIFICATION, SUPPORT
    AUTHORITY = "authority"  # APPROVAL
    NONE = "none"         # No intervention (STP)
```

#### Sentiment

```python
class Sentiment(str, RichEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
```

### AuditResult

The structured analysis extracted from the conversation by the LLM.

```python
class AuditResult(RichBaseModel):
    has_human_intervention: bool              # True if human wrote a message
    intervention_category: InterventionType   # Why human intervened
    escalation_type: Optional[EscalationType] # Computed from intervention_category
    summary_of_human_input: Optional[str]     # 1-sentence summary
    friction_point: Optional[str]             # Specific concept causing friction
    issue_details: Optional[str]              # Technical description with values
    human_sentiment: Sentiment                # Sentiment classification
    final_status: str                         # Final quote state
```

### ConversationMetrics

Computed metrics for a single conversation.

```python
class ConversationMetrics(RichBaseModel):
    thread_id: str
    computed_at: datetime

    # Message counts
    total_messages: int
    human_messages: int
    bot_messages: int

    # Time metrics
    duration_seconds: float
    first_response_time_seconds: Optional[float]
    avg_response_time_seconds: Optional[float]

    # Ping-pong analysis
    exchange_count: int                    # Bot<->human transitions
    ping_pong_ratio: float                 # Exchanges / total messages

    # Resolution
    is_resolved: bool
    resolution_type: Optional[str]         # "approved", "declined", "stalemate"
    time_to_resolution_seconds: Optional[float]

    # Sentiment & stalemate
    has_frustrated_message: bool
    is_stalemate: bool                     # Inactive >72 hours without resolution
```

### AggregatedMetrics

Aggregated metrics for a time period (dashboard data).

```python
class AggregatedMetrics(RichBaseModel):
    # Period metadata
    period_start: datetime
    period_end: datetime
    period_type: str                       # "daily", "weekly", "monthly"

    # Trust metrics
    mau: int                               # Monthly Active Users
    total_conversations: int
    shift_left_rate: float                 # STP rate

    # Escalation spectrum
    hard_count: int
    soft_count: int
    authority_count: int
    stp_count: int

    # Operational efficiency
    stp_rate: float
    avg_messages: float
    avg_ping_pong: float
    avg_turnaround_seconds: float

    # Conversation hygiene
    clarification_rate: float
    mttr_seconds: float                    # Mean Time to Resolution
    stalemate_rate: float
    frustrated_rate: float

    # Business impact
    approval_rate: float
    decline_rate: float
```

---

## Handler

### UnderwritingAuditHandler

Specialized LLM handler to audit AI-Human underwriting conversations.

```python
class UnderwritingAuditHandler(LLMHandler[ConversationContext, AuditResult]):
    input_model = ConversationContext
    output_model = AuditResult
    as_structured_llm = False
```

The handler prompts the LLM as a "Senior Underwriting Auditor" to:
1. Check if a human posted a message
2. Categorize the intervention type
3. Extract friction points
4. Capture issue details with specific values
5. Determine the final quote status

---

## Pipeline

### PipelineResult

Result container for pipeline execution.

```python
@dataclass
class PipelineResult:
    contexts: List[ConversationContext] = field(default_factory=list)
    audit_results: List[AuditResult] = field(default_factory=list)
    metrics: List[ConversationMetrics] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    human_user_ids: Set[str] = field(default_factory=set)

    @property
    def success_count(self) -> int: ...

    @property
    def error_count(self) -> int: ...

    @property
    def total_count(self) -> int: ...
```

### FeedbackAnalysisPipeline

Main pipeline for processing Slack conversations and computing analytics.

```python
class FeedbackAnalysisPipeline:
    def __init__(self, llm=None, db_manager=None):
        """
        Args:
            llm: LLM instance for analysis. Defaults to registry LLM.
            db_manager: Optional database manager for persistence.
        """

    async def process(self, raw_data: List[Dict[str, Any]]) -> PipelineResult:
        """Process raw Slack data through full pipeline."""
```

**Process Flow:**
1. Transform raw JSON to typed ConversationContext objects
2. Run LLM analysis concurrently
3. Separate successes from errors
4. Classify escalation types
5. Compute per-conversation metrics
6. Collect human user IDs for MAU
7. Persist if db configured
8. Return PipelineResult

---

## Metric Computations

### classify_escalation

Maps InterventionType to EscalationType for dashboard categorization.

```python
def classify_escalation(intervention: InterventionType) -> EscalationType:
    """
    Mapping:
    - HARD: CORRECTION_FACTUAL, TECH_ISSUE, DATA_QUALITY
    - SOFT: MISSING_CONTEXT, RISK_APPETITE, CLARIFICATION, SUPPORT
    - AUTHORITY: APPROVAL
    - NONE: no_intervention
    """
```

### ConversationMetricCalculator

Computes per-conversation metrics at ingestion time.

```python
class ConversationMetricCalculator:
    STALEMATE_THRESHOLD_SECONDS = 72 * 60 * 60  # 259200 seconds

    def compute(
        self,
        context: ConversationContext,
        audit_result: AuditResult,
        current_time: Optional[datetime] = None,
    ) -> ConversationMetrics:
        """Compute all metrics for a single conversation."""
```

**Computed Metrics:**
- Message counts (total, human, bot)
- Time metrics (duration, response times)
- Ping-pong analysis (exchange count, ratio)
- Resolution detection (resolved, type, time to resolution)
- Sentiment analysis (frustrated message detection)
- Stalemate detection (>72 hours inactive)

---

## Aggregation Service

### MetricAggregationService

Service for aggregating per-conversation metrics into period-based dashboard data.

```python
class MetricAggregationService:
    def aggregate_period(
        self,
        metrics: List[ConversationMetrics],
        audit_results: List[AuditResult],
        user_ids: Set[str],
        period_start: datetime,
        period_end: datetime,
        period_type: str = "daily",
    ) -> AggregatedMetrics:
        """Aggregate metrics for a time period."""

    def aggregate_daily(self, metrics, audit_results, user_ids, date) -> AggregatedMetrics: ...
    def aggregate_weekly(self, metrics, audit_results, user_ids, week_start) -> AggregatedMetrics: ...
    def aggregate_monthly(self, metrics, audit_results, user_ids, year, month) -> AggregatedMetrics: ...
```

**Aggregation Categories:**
- **Trust Metrics:** MAU, STP rate, shift-left rate
- **Escalation Spectrum:** hard/soft/authority/STP counts
- **Operational Efficiency:** avg messages, ping-pong, turnaround time
- **Conversation Hygiene:** clarification rate, MTTR, stalemate rate, frustrated rate
- **Business Impact:** approval rate, decline rate

---

## Jobs

### AggregationJobRunner

Runner for aggregation jobs. Coordinates data fetching, aggregation, and storage.

```python
class AggregationJobRunner:
    def __init__(self, db_manager=None): ...

    async def run_daily_aggregation(
        self, date: Optional[datetime] = None, store_result: bool = True
    ) -> AggregatedMetrics: ...

    async def run_weekly_aggregation(
        self, week_start: Optional[datetime] = None, store_result: bool = True
    ) -> AggregatedMetrics: ...

    async def run_monthly_aggregation(
        self, year: Optional[int] = None, month: Optional[int] = None, store_result: bool = True
    ) -> AggregatedMetrics: ...
```

### run_all_pending_aggregations

Utility function to run all pending aggregations since a given date.

```python
async def run_all_pending_aggregations(
    runner: AggregationJobRunner,
    since: Optional[datetime] = None
) -> Dict[str, List[AggregatedMetrics]]:
    """
    Returns: Dict with 'daily', 'weekly', 'monthly' keys
    """
```

---

## Code Examples

### High-Level: Pipeline Usage

```python
from implementations.athena.models.feedback_analysis import (
    FeedbackAnalysisPipeline,
    PipelineResult,
)

# Initialize pipeline
pipeline = FeedbackAnalysisPipeline(llm=my_llm, db_manager=db)

# Load raw Slack data
raw_conversations = [
    {"id": "thread_1", "threadReplies": [...], "messageUrl": "..."},
    {"id": "thread_2", "threadReplies": [...], "messageUrl": "..."},
]

# Process through pipeline
result: PipelineResult = await pipeline.process(raw_conversations)

# Access results
print(f"Processed: {result.success_count}/{result.total_count}")
print(f"Errors: {result.error_count}")
print(f"Unique human users: {len(result.human_user_ids)}")

# Iterate results
for ctx, audit, metric in zip(result.contexts, result.audit_results, result.metrics):
    print(f"Thread {ctx.thread_id}:")
    print(f"  Case: {ctx.case_id}")
    print(f"  Intervention: {audit.intervention_category.value}")
    print(f"  Escalation: {audit.escalation_type.value}")
    print(f"  Messages: {metric.total_messages}")
    print(f"  Duration: {metric.duration_seconds:.0f}s")
```

### Mid-Level: Aggregation Jobs

```python
from implementations.athena.models.feedback_analysis import (
    AggregationJobRunner,
    run_all_pending_aggregations,
)
from datetime import datetime, timedelta

# Initialize runner
runner = AggregationJobRunner(db_manager=db)

# Run daily aggregation for yesterday
daily = await runner.run_daily_aggregation()
print(f"Daily STP Rate: {daily.stp_rate:.1%}")
print(f"Hard Escalations: {daily.hard_count}")

# Run weekly aggregation
weekly = await runner.run_weekly_aggregation()
print(f"Weekly Conversations: {weekly.total_conversations}")

# Backfill missing aggregations
since = datetime.now() - timedelta(days=30)
results = await run_all_pending_aggregations(runner, since=since)
print(f"Created {len(results['daily'])} daily aggregations")
print(f"Created {len(results['weekly'])} weekly aggregations")
```

### Low-Level: Direct Handler Usage

```python
from implementations.athena.models.feedback_analysis import (
    UnderwritingAuditHandler,
    ConversationContext,
    Message,
    AuditResult,
    ConversationMetricCalculator,
    classify_escalation,
)
from shared.llm.registry import LLMRegistry

# Create handler
llm = LLMRegistry().get_llm()
handler = UnderwritingAuditHandler(llm=llm)

# Build context manually
context = ConversationContext(
    thread_id="thread_123",
    case_id="MGT-BOP-2024-001",
    messages=[
        Message(ts="1700000000.000000", sender="Athena", is_bot=True, content="New case..."),
        Message(ts="1700000060.000000", sender="John", is_bot=False, content="Override approved"),
    ],
)

# Run audit
audit_result: AuditResult = await handler.execute(context)
print(f"Has intervention: {audit_result.has_human_intervention}")
print(f"Category: {audit_result.intervention_category}")

# Classify escalation
escalation = classify_escalation(audit_result.intervention_category)
print(f"Escalation type: {escalation}")

# Compute metrics
calculator = ConversationMetricCalculator()
metrics = calculator.compute(context, audit_result)
print(f"Exchange count: {metrics.exchange_count}")
print(f"Ping-pong ratio: {metrics.ping_pong_ratio:.2f}")
```

---

## Exports

```python
from implementations.athena.models.feedback_analysis import (
    # Schema
    Message,
    ConversationContext,
    InterventionType,
    EscalationType,
    Sentiment,
    AuditResult,
    ConversationMetrics,
    AggregatedMetrics,

    # Pipeline
    FeedbackAnalysisPipeline,
    PipelineResult,
    run_pipeline,  # Legacy

    # Handler
    UnderwritingAuditHandler,

    # Metric computation
    ConversationMetricCalculator,
    classify_escalation,
    parse_slack_timestamp,
    STALEMATE_THRESHOLD_SECONDS,

    # Aggregation
    MetricAggregationService,

    # Jobs
    AggregationJobRunner,
    run_all_pending_aggregations,
)
```
