# Langfuse Prompt Management

The Athena Langfuse integration provides a join system that bridges Neon database records with Langfuse distributed traces, enabling rich analysis of AI workflows through combined case and trace data.

## Overview

```
┌─────────────────────────────┐
│  AthenaNeonLangfuseJoiner   │
└──────────────┬──────────────┘
               │
    ┌──────────┴───────────────────────────────┐
    │                                          │
    ▼                                          ▼
┌──────────────────┐              ┌──────────────────────┐
│ NeonDatabaseMgr  │              │ LangfuseTraceLoader  │
│  fetch_dataframe │              │   fetch_traces()     │
│  fetch_cases()   │              │  fetch_trace_by_id() │
└────────┬─────────┘              └──────────┬───────────┘
         │                                   │
         ▼                                   ▼
    ┌─────────┐                    ┌──────────────────┐
    │ cases   │                    │ TraceCollection  │
    │DataFrame│                    │  List[Trace]     │
    └────┬────┘                    └──────┬───────────┘
         │                                │
         │         join_cases_with_traces()
         │                 │
         └────────────┬────┘
                      ▼
           ┌──────────────────────┐
           │ Joined DataFrame     │
           │ - case columns       │
           │ + langfuse_trace col │
           └──────────────────────┘
```

### Trace Access (SmartAccess)

```
Trace.recommendation
   └─ TraceStep
      ├─ .generation (GENERATION observation)
      ├─ .context (SPAN observation)
      └─ .variables (extracted from patterns)
         ├─ caseAssessment
         ├─ contextData
         ├─ underwritingFlags
         └─ swallowDebugData
```

---

## JoinSettings Configuration

Frozen dataclass for configuring the join operation.

```python
@dataclass(frozen=True)
class JoinSettings:
    case_table: str = "athena_cases"
    case_columns: tuple[str, ...] = (
        "id",
        "workflow_id",
        "quote_locator",
        "slack_thread_ts",
        "slack_channel_id",
        "langfuse_trace_id",
    )
    trace_name: str = "athena"
    trace_tags: tuple[str, ...] = ("production",)
```

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `case_table` | `str` | `"athena_cases"` | Target table in Neon database |
| `case_columns` | `tuple[str, ...]` | See above | Columns to fetch from case_table |
| `trace_name` | `str` | `"athena"` | Langfuse trace name filter |
| `trace_tags` | `tuple[str, ...]` | `("production",)` | Langfuse trace tags to filter by |

**Usage:**
```python
# Default settings
settings = JoinSettings()

# Custom settings
settings = JoinSettings(
    case_table="custom_cases",
    case_columns=("id", "workflow_id", "status"),
    trace_name="custom_workflow",
    trace_tags=("staging", "v2"),
)
```

---

## AthenaNeonLangfuseJoiner

Main helper class to coordinate fetching and joining case data from Neon with trace data from Langfuse.

### Constructor

```python
def __init__(
    self,
    neon_db: NeonConnection,
    trace_loader: LangfuseTraceLoader,
    *,
    settings: JoinSettings | None = None,
    prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neon_db` | `NeonConnection` | Required | Database connection manager |
| `trace_loader` | `LangfuseTraceLoader` | Required | Langfuse trace fetching client |
| `settings` | `JoinSettings \| None` | `JoinSettings()` | Configuration for table/column/trace names |
| `prompt_patterns` | `PromptPatternsBase` | `WorkflowPromptPatterns` | Pattern registry for prompt extraction |

### Methods

#### fetch_cases()

Fetch case records from Neon database.

```python
def fetch_cases(
    self,
    *,
    limit: Optional[int] = None,
    where: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame
```

**Example:**
```python
cases = joiner.fetch_cases(
    limit=50,
    where="status = 'active'",
    columns=("id", "workflow_id", "quote_locator"),
)
```

#### fetch_traces()

Fetch traces from Langfuse by name and tags.

```python
def fetch_traces(
    self,
    *,
    limit: int = 200,
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    fetch_full_traces: bool = True,
    show_progress: bool = False,
) -> TraceCollection
```

**Example:**
```python
traces = joiner.fetch_traces(
    limit=100,
    name="custom_workflow",
    tags=["staging"],
    fetch_full_traces=True,
)
```

#### fetch_traces_by_ids()

Fetch specific traces by their IDs.

```python
def fetch_traces_by_ids(
    self,
    trace_ids: Sequence[str],
    *,
    fetch_full_traces: bool = True,
    show_progress: bool = False,
    trace_fetcher: Callable[[str], object] | None = None,
) -> TraceCollection
```

**Example:**
```python
trace_ids = ["trace-001", "trace-002", "trace-003"]
traces = joiner.fetch_traces_by_ids(trace_ids, fetch_full_traces=True)
```

#### join_cases_with_traces()

Join case records with corresponding trace objects via trace ID.

```python
def join_cases_with_traces(
    self,
    cases: pd.DataFrame,
    traces: TraceCollection,
    *,
    trace_id_column: str = "langfuse_trace_id",
    trace_output_column: str = "langfuse_trace",
) -> pd.DataFrame
```

**Example:**
```python
joined = joiner.join_cases_with_traces(
    cases,
    traces,
    trace_id_column="langfuse_trace_id",
    trace_output_column="trace_data",
)
# Result: cases DataFrame with new 'trace_data' column containing Trace objects
```

---

## Prompt Extraction Patterns

Pattern classes define regex patterns for extracting structured data from LLM prompts. They inherit from `PromptPatternsBase`.

### PromptPatternsBase

Base registry for regex extraction patterns.

```python
class PromptPatternsBase:
    @classmethod
    def get_for(cls, step_name: str) -> Dict[str, str]:
        """
        Dynamically looks for methods named _patterns_{step_name}().
        Returns dict of {field_name: regex_pattern}.
        """
```

### WorkflowPromptPatterns

Extract structured fields from "recommendation" workflow prompts.

```python
class WorkflowPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_recommendation() -> Dict[str, str]:
        """Returns patterns for recommendation step."""
```

**Extracted Fields:**

| Field | Header Delimiter | Description |
|-------|-----------------|-------------|
| `caseAssessment` | "CASE ASSESSMENT..." → "FULL CONTEXT DATA" | Previous case analysis |
| `contextData` | "FULL CONTEXT DATA" → "UNDERWRITING FLAGS..." | Complete context info |
| `underwritingFlags` | "UNDERWRITING FLAGS..." → "SWALLOW DEBUG DATA..." | Active underwriting rules |
| `swallowDebugData` | "SWALLOW DEBUG DATA..." → "PREVIOUS RECOMMENDATIONS..." or EOF | Debug trace data |

### ChatPromptPatterns

Extract structured fields from chat-based prompts.

```python
class ChatPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_chat() -> Dict[str, str]:
        """Returns patterns for chat step."""
```

**Extracted Fields:**

| Field | Delimiter Pattern | Description |
|-------|------------------|-------------|
| `conversationHistory` | `**Conversation History:**` → `**Quote Context:**` | Chat message history |
| `quoteContext` | `**Quote Context:**` → `**Current User Message:**` | Quote/reference data |
| `currentUserMessage` | `**Current User Message:**` → `**Context:**` | Latest user input |
| `context` | `**Context:**` → `**Instructions:**` | System/background context |
| `instructions` | `**Instructions:**` → EOF | System instructions |

---

## TraceCollection and SmartAccess Utilities

### TraceCollection

Wraps a list of trace data items, providing collection-level operations.

```python
class TraceCollection:
    def __init__(
        self,
        data: List[Any],
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    ): ...

    def __getitem__(self, index: int) -> Trace: ...
    def __iter__(self) -> Iterator[Trace]: ...
    def __len__(self) -> int: ...
    def filter_by(self, **kwargs) -> TraceCollection: ...
```

**Example:**
```python
traces = TraceCollection(trace_data_list, prompt_patterns=WorkflowPromptPatterns)

# Access individual traces
first_trace = traces[0]

# Iterate
for trace in traces:
    print(trace.id)

# Filter
production_traces = traces.filter_by(name="athena")

# Get count
total = len(traces)
```

### SmartAccess

Base class enabling intelligent dot-notation access with fuzzy matching.

**Features:**
- Dot notation access: `trace.recommendation`
- Bracket access: `trace['recommendation']`
- Case-insensitive matching: `trace.product_type` matches `productType`
- Recursive wrapping: nested objects maintain SmartAccess

**Fuzzy Matching:**
```python
def _normalize_key(key: str) -> str:
    """'product_type' matches 'productType'"""
    return key.lower().replace("_", "")
```

### Trace

Main wrapper providing access to steps and trace-level attributes.

```python
class Trace(SmartAccess):
    def __init__(
        self,
        trace_data: Any,
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    ): ...
```

**Access Patterns:**

| Access | Returns | Example |
|--------|---------|---------|
| `trace.{step_name}` | `TraceStep` | `trace.recommendation` |
| `trace.{trace_attr}` | Any | `trace.id`, `trace.latency` |

### TraceStep

Represents a specific named step (e.g., 'recommendation').

```python
class TraceStep(SmartAccess):
    name: str                              # Step name
    observations: List[ObservationsView]   # Observations in step
    prompt_patterns: PromptPatternsBase    # Pattern registry
```

**Properties:**

| Access | Returns | Description |
|--------|---------|-------------|
| `step.variables` | `Dict[str, str]` | Lazily extracted prompt variables |
| `step.generation` | `ObservationsView` | First GENERATION observation |
| `step.context` | `ObservationsView` | SPAN observation (context) |
| `step.span` | `ObservationsView` | SPAN observation |

---

## Integration with shared/langfuse/trace.py

The Athena implementation builds on the shared Langfuse trace module:

1. **PromptPatternsBase Inheritance** - `WorkflowPromptPatterns` and `ChatPromptPatterns` extend base
2. **TraceCollection Usage** - `fetch_traces()` returns `TraceCollection`
3. **Trace Smart Access** - All Trace objects use SmartAccess for dot notation
4. **Pattern Resolution** - Helper converts class to instance for all wrapped traces

---

## Code Examples

### Basic Usage: Fetch and Join

```python
from implementations.athena.langfuse.join import AthenaNeonLangfuseJoiner, JoinSettings
from implementations.athena.langfuse.prompt_patterns import WorkflowPromptPatterns
from shared.database.neon import NeonConnection
from axion.tracing import LangfuseTraceLoader

# Initialize components
neon_db = NeonConnection()
trace_loader = LangfuseTraceLoader(api_key="YOUR_API_KEY")

# Create joiner
joiner = AthenaNeonLangfuseJoiner(
    neon_db,
    trace_loader,
    prompt_patterns=WorkflowPromptPatterns,
)

# Fetch cases and traces
cases = joiner.fetch_cases(limit=100, where="status = 'completed'")
trace_ids = cases['langfuse_trace_id'].dropna().unique()
traces = joiner.fetch_traces_by_ids(trace_ids, fetch_full_traces=True)

# Join
joined = joiner.join_cases_with_traces(cases, traces)
print(f"Joined {len(joined)} cases with traces")
```

### Analyzing Traces

```python
# Access joined data
for idx, row in joined.iterrows():
    case_id = row['id']
    trace = row['langfuse_trace']

    if trace is None:
        print(f"Case {case_id}: No trace found")
        continue

    # Access trace metadata
    print(f"Case {case_id}:")
    print(f"  Trace ID: {trace.id}")
    print(f"  Latency: {trace.latency}ms")

    # Access recommendation step
    rec_step = trace.recommendation

    # Extract variables from prompt (lazy evaluation)
    variables = rec_step.variables
    print(f"  Case Assessment: {variables.get('caseAssessment', 'N/A')[:100]}...")
    print(f"  Context Data: {variables.get('contextData', 'N/A')[:100]}...")

    # Access generation observation
    gen = rec_step.generation
    print(f"  Generation model: {gen.model}")
```

### Custom Settings

```python
# Configure for staging environment
settings = JoinSettings(
    case_table="athena_cases_staging",
    case_columns=("id", "workflow_id", "langfuse_trace_id", "status"),
    trace_name="athena-staging",
    trace_tags=("staging", "v2"),
)

joiner = AthenaNeonLangfuseJoiner(
    neon_db,
    trace_loader,
    settings=settings,
    prompt_patterns=WorkflowPromptPatterns,
)
```

### Custom Prompt Patterns

```python
from shared.langfuse.trace import PromptPatternsBase, create_extraction_pattern

class CustomPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_my_step() -> Dict[str, str]:
        return {
            "fieldA": create_extraction_pattern(
                "FIELD A HEADER",
                "FIELD B HEADER"
            ),
            "fieldB": create_extraction_pattern(
                "FIELD B HEADER",
                "(?:$)"  # End of text
            ),
        }

# Use custom patterns
joiner = AthenaNeonLangfuseJoiner(
    neon_db,
    trace_loader,
    prompt_patterns=CustomPromptPatterns,
)
```

### Working with TraceCollection

```python
# Fetch traces
traces = joiner.fetch_traces(limit=200)

# Filter traces
recommendation_traces = traces.filter_by(name="recommendation")

# Iterate and analyze
for trace in traces:
    # Access via dot notation (case insensitive)
    trace_id = trace.id

    # Access steps
    if hasattr(trace, 'recommendation'):
        rec = trace.recommendation
        print(f"Trace {trace_id}: has recommendation step")

        # Lazy variable extraction
        vars = rec.variables
        if 'caseAssessment' in vars:
            print(f"  Assessment: {vars['caseAssessment'][:50]}...")
```

---

## Exports

```python
from implementations.athena.langfuse.join import (
    JoinSettings,
    AthenaNeonLangfuseJoiner,
)

from implementations.athena.langfuse.prompt_patterns import (
    WorkflowPromptPatterns,
    ChatPromptPatterns,
)

from shared.langfuse.trace import (
    PromptPatternsBase,
    SmartAccess,
    SmartDict,
    SmartObject,
    Trace,
    TraceStep,
    TraceCollection,
    create_extraction_pattern,
)
```
