# Langfuse Integration (Shared)

The shared Langfuse module provides prompt management, trace handling, and
webhook integration for LLM observability. It enables intelligent access to
prompts and traces with smart dot-notation access patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   LangfusePromptManager                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ • create_or_update_prompt()  [→ PromptClient]           │   │
│  │ • get_prompt()               [→ PromptClient + caching] │   │
│  │ • get_compiled_prompt()      [→ str|List with variables]│   │
│  │ • get_template()             [→ raw template]           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  • mark_prompt_as_stale(name)                                    │
│  • notify_prompt_change(name, payload)                           │
│  • on_prompt_change(listener) [registers callback]               │
└─────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Webhook Handler                              │
│  POST /webhooks/langfuse                                        │
│  • verify_signature()  [HMAC-SHA256]                            │
│  • Handles: prompt.{created,updated,deleted}                    │
│  • Marks stale → Notifies listeners → Posts Slack alerts        │
└─────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────┐
│              Trace Access & Analysis (trace.py)                 │
│                                                                 │
│  Trace(trace_data, prompt_patterns)                             │
│   ├─ trace.recommendation  → TraceStep                          │
│   │   ├─ step.generation   → ObservationsView                   │
│   │   ├─ step.context      → ObservationsView                   │
│   │   └─ step.variables    → Dict[extracted variables]          │
│   ├─ trace.id              → trace attribute                    │
│   └─ trace.latency         → trace attribute                    │
│                                                                 │
│  SmartAccess Framework (dot-notation + fuzzy matching)          │
│   ├─ SmartDict (dict wrapper)                                   │
│   ├─ SmartObject (object wrapper)                               │
│   └─ Recursive wrapping of nested structures                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### LangfuseSettings

```python
class LangfuseSettings(RepoSettingsBase):
    # API Credentials
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    # Prompt Management
    langfuse_default_label: str = "production"
    langfuse_default_cache_ttl_seconds: int = 60

    # Webhook Configuration
    langfuse_webhook_secret: str | None = None
    langfuse_webhook_notify_url: str | None = None

    # Slack Alerts
    langfuse_slack_channel_id: str | None = None
    langfuse_slack_request_timeout_seconds: float = 10
    langfuse_slack_retry_max_attempts: int = 3
    langfuse_slack_retry_backoff_seconds: float = 0.5
    langfuse_slack_retry_max_backoff_seconds: float = 4.0
```

| Setting | Default | Description |
|---------|---------|-------------|
| `langfuse_default_label` | `"production"` | Default prompt label |
| `langfuse_default_cache_ttl_seconds` | `60` | Local cache duration |
| `langfuse_slack_request_timeout_seconds` | `10` | Slack API timeout |
| `langfuse_slack_retry_max_attempts` | `3` | Retry count for Slack |

---

## LangfusePromptManager

Central manager for all Langfuse prompt operations including CRUD, caching,
invalidation, and change notifications.

### Initialization

```python
from eval_workbench.shared.langfuse.prompt import LangfusePromptManager

manager = LangfusePromptManager(
    public_key="pk-...",      # Optional, uses env if not provided
    secret_key="sk-...",      # Optional, uses env if not provided
    host="https://...",       # Optional, uses env if not provided
)
```

### Creating and Updating Prompts

```python
def create_or_update_prompt(
    self,
    name: str,
    prompt_content: Union[str, List[Dict[str, str]]],
    prompt_type: str = "text",           # "text" or "chat"
    labels: Optional[Sequence[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> PromptClient
```

**Example:**
```python
# Text prompt
manager.create_or_update_prompt(
    name="recommendation",
    prompt_content="Analyze this case: {{ case_data }}",
    prompt_type="text",
    labels=["production", "v2"],
    config={"model": "gpt-4", "temperature": 0.7},
)

# Chat prompt
manager.create_or_update_prompt(
    name="chat_assistant",
    prompt_content=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{{ user_message }}"},
    ],
    prompt_type="chat",
    labels=["production"],
)
```

### Fetching Prompts

```python
def get_prompt(
    self,
    name: str,
    version: Optional[int] = None,
    label: Optional[str] = None,
    cache_ttl_seconds: Optional[int] = None,
    retry_count: int = 0,
    retry_backoff_seconds: float = 0.2,
) -> PromptClient
```

**Behavior:**
- Defaults to "production" label if version/label omitted
- Checks `_stale_prompts` and bypasses cache if flagged
- Implements exponential backoff retry logic

### Compiling Prompts with Variables

```python
def get_compiled_prompt(
    self,
    name: str,
    variables: Dict[str, Any],
    *,
    fallback: Optional[Union[str, List[Dict[str, Any]]]] = None,
    strict: bool = True,
    required_variables: Optional[Sequence[str]] = None,
    strict_mode: Literal["template", "required", "none"] = "template",
    fallback_on_compile_error: bool = True,
    render_with: Optional[Callable] = None,
    **kwargs,
) -> Union[str, List[Dict]]
```

**Strict Mode Options:**
- `"template"`: Check for unrendered placeholders after compilation
- `"required"`: Verify required_variables are present before compilation
- `"none"`: Skip all strict checks

**Example:**
```python
compiled = manager.get_compiled_prompt(
    "recommendation",
    variables={"case_id": "123", "assessment": "high-risk"},
    strict_mode="template",
    fallback="Default prompt text",
)
```

### Cache Invalidation

```python
# Mark prompt for immediate refresh on next fetch
manager.mark_prompt_as_stale("recommendation")

# Promote specific version to labels
manager.promote_version("recommendation", version=5, labels=["production"])
```

### Change Notifications

```python
# Register listener
def on_update(name: str, payload: Dict[str, Any] | None):
    print(f"Prompt {name} updated: {payload}")

manager.on_prompt_change(on_update)

# Manually trigger notifications
manager.notify_prompt_change("recommendation", payload={"version": 6})
```

---

## SmartAccess Framework

The SmartAccess system enables intelligent dot-notation access with fuzzy
matching across trace data.

### SmartAccess (Base Class)

```python
class SmartAccess:
    def __getattr__(self, key: str) -> Any:
        """Dot-notation with 2-step fallback:
        1. Exact match via _lookup(key)
        2. Fuzzy match via _lookup_insensitive(key)
        """

    def _wrap(self, val: Any) -> Any:
        """Recursively wrap results:
        - dict → SmartDict
        - list → list of wrapped items
        - object → SmartObject
        """
```

### Fuzzy Matching

Keys are normalized for case/separator-insensitive matching:

```python
def _normalize_key(key: str) -> str:
    return key.lower().replace("_", "")

# Examples:
# "product_type" matches "productType"
# "CaseAssessment" matches "caseassessment"
```

### SmartDict

Dictionary wrapper with dot-notation and fuzzy key matching.

```python
smart_dict = SmartDict({"productType": "insurance", "caseId": "123"})
smart_dict.product_type  # "insurance" (fuzzy match)
smart_dict.case_id       # "123" (fuzzy match)
smart_dict.to_dict()     # Returns raw dictionary
```

### SmartObject

Generic object wrapper ensuring attributes return Smart wrappers.

```python
smart_obj = SmartObject(some_object)
smart_obj.nested_attr.deep_value  # Recursive wrapping
```

---

## Trace Classes

### Trace

Main entry point for accessing trace data with smart dot-notation navigation.

```python
class Trace(SmartAccess):
    def __init__(
        self,
        trace_data: Any,
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    )
```

**Access Patterns:**

| Access | Returns | Description |
|--------|---------|-------------|
| `trace.{step_name}` | `TraceStep` | Named workflow step |
| `trace.id` | `str` | Trace identifier |
| `trace.latency` | `int` | Execution time |
| `trace.name` | `str` | Trace name |

### TraceStep

Represents a named workflow step (e.g., "recommendation") with observations.

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
| `step.generation` | `ObservationsView` | GENERATION observation |
| `step.context` | `ObservationsView` | SPAN observation (alias) |
| `step.span` | `ObservationsView` | SPAN observation |

### TraceCollection

Container for multiple traces with filtering and iteration.

```python
class TraceCollection:
    def __init__(
        self,
        data: List[Any],
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    )

    def __getitem__(self, index: int) -> Trace: ...
    def __iter__(self) -> Iterator[Trace]: ...
    def __len__(self) -> int: ...
    def filter_by(self, **kwargs) -> TraceCollection: ...
```

**Example:**
```python
traces = TraceCollection(trace_list, prompt_patterns=WorkflowPromptPatterns)

# Filter
filtered = traces.filter_by(name="athena", status="completed")

# Iterate
for trace in traces:
    print(trace.id, trace.recommendation.variables)
```

---

## PromptPatternsBase

Registry for regex extraction patterns. Enables custom pattern definitions per
workflow.

```python
class PromptPatternsBase:
    @classmethod
    def get_for(cls, step_name: str) -> Dict[str, str]:
        """
        Looks for method: _patterns_{step_name_lowercase}()
        Returns: Dict[pattern_name: pattern_regex]
        """
```

### Creating Custom Patterns

```python
from eval_workbench.shared.langfuse.trace import PromptPatternsBase, create_extraction_pattern

class CustomPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_recommendation() -> Dict[str, str]:
        return {
            "caseAssessment": create_extraction_pattern(
                "CASE ASSESSMENT",
                "CONTEXT DATA"
            ),
            "contextData": create_extraction_pattern(
                "CONTEXT DATA",
                "(?:$)"  # End of text
            ),
        }
```

### Helper Function

```python
def create_extraction_pattern(start_text: str, end_pattern: str) -> str:
    """
    Creates regex: escaped(Start) → (Content) → End

    Example:
        create_extraction_pattern("CONTEXT:", "FLAGS:")
        → r"CONTEXT:\s*(.*?)\s*(?:FLAGS:)"
    """
```

---

## Webhook Handler

The webhook handler processes Langfuse prompt update events.

### Endpoint

```
POST /webhooks/langfuse
Header: X-Langfuse-Signature: t=<timestamp>,v1=<signature>
```

### Signature Verification

```python
def verify_signature(
    payload: bytes,
    signature_header: str | None,
    secret: str
) -> None:
    """
    Verifies HMAC-SHA256 signature.
    Supports both timestamped and non-timestamped formats.
    """
```

### Event Handling

**Supported Events:**
- `prompt.created` - New prompt created
- `prompt.updated` - Existing prompt modified
- `prompt.deleted` - Prompt removed

**Processing Flow:**
1. Verify HMAC signature
2. Parse JSON payload
3. Extract prompt name
4. Mark prompt as stale in manager
5. Notify registered listeners
6. Notify external services (if configured)
7. Post Slack alert (async, non-blocking)

### Slack Alerts

When configured, posts formatted alerts to Slack on prompt changes:

```python
# Requires environment variables:
# LANGFUSE_SLACK_CHANNEL_ID - Target Slack channel
# SLACK_ATHENA_TOKEN - Bot token for posting
```

---

## Utility Functions

### parse_chat_transcript

Parses raw string chat transcript into structured messages.

```python
from eval_workbench.shared.langfuse.utils import parse_chat_transcript

transcript = """
Athena: Hello, how can I help?
User: I have a question about my policy.
Athena: I'd be happy to help with that.
"""

messages = parse_chat_transcript(transcript, agent_name="Athena")
# [
#     {"role": "assistant", "content": "Hello, how can I help?"},
#     {"role": "user", "content": "I have a question about my policy."},
#     {"role": "assistant", "content": "I'd be happy to help with that."},
# ]
```

---

## Code Examples

### Prompt Management

```python
from eval_workbench.shared.langfuse.prompt import LangfusePromptManager

manager = LangfusePromptManager()

# Fetch and compile with variables
compiled = manager.get_compiled_prompt(
    "recommendation",
    variables={"case_id": "123", "assessment": "high-risk"},
    strict_mode="template",
    fallback="Default prompt text",
)

# Register for updates
def on_update(name, payload):
    print(f"Prompt {name} updated: {payload}")

manager.on_prompt_change(on_update)
```

### Trace Analysis

```python
from eval_workbench.shared.langfuse.trace import Trace, TraceCollection
from eval_workbench.implementations.athena.langfuse.prompt_patterns import WorkflowPromptPatterns

# Single trace
trace = Trace(trace_data, prompt_patterns=WorkflowPromptPatterns)
assessment = trace.recommendation.variables["caseAssessment"]
generation_output = trace.recommendation.generation.output

# Multiple traces
traces = TraceCollection(trace_list, prompt_patterns=WorkflowPromptPatterns)
for trace in traces:
    if trace.id in important_ids:
        print(trace.recommendation.generation.output)
```

### Deep Trace Navigation

```python
trace = Trace(trace_data, prompt_patterns=WorkflowPromptPatterns)

# Access step
rec = trace.recommendation

# Get generation observation
gen = rec.generation
print(gen.input)   # Prompt input
print(gen.output)  # Model output
print(gen.model)   # Model used

# Extract variables (lazy evaluation)
vars = rec.variables
print(vars["caseAssessment"])
print(vars["contextData"])

# Access trace metadata
print(trace.id)
print(trace.latency)
print(trace.name)
```

---

## Exports

```python
from eval_workbench.shared.langfuse.prompt import (
    LangfuseSettings,
    get_langfuse_settings,
    LangfusePromptManager,
)

from eval_workbench.shared.langfuse.trace import (
    # Enums
    ModelUsageUnit,
    ObservationLevel,

    # Dataclasses
    Usage,

    # SmartAccess Framework
    SmartAccess,
    SmartDict,
    SmartObject,

    # Trace Views
    TraceView,
    ObservationsView,

    # Pattern Extraction
    create_extraction_pattern,
    PromptPatternsBase,

    # Main Classes
    TraceStep,
    Trace,
    TraceCollection,
)

from eval_workbench.shared.langfuse.webhook import (
    verify_signature,
    langfuse_webhook,
)

from eval_workbench.shared.langfuse.utils import (
    parse_chat_transcript,
)
```
