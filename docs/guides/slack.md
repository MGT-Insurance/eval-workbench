# Slack Integration

The Slack integration module provides a complete, production-ready interface for
Slack interactions with async support, retry logic, message parsing, formatting
validation, and subscription management.

## Architecture

```
SlackConfig (Token Management)
    └── get_token(agent_id) -> str

SlackHttpClient (Low-level Async)
    ├── request() -> with retry logic
    └── _get_session()

SlackService (High-level API)
    ├── post_message()
    ├── update_message()
    ├── post_threaded_message()
    ├── get_thread_replies()
    ├── get_channel_history()
    ├── open_modal()
    ├── publish_app_home()
    ├── download_file()
    ├── post_to_subscribed_channels()
    └── post_to_all_threads()

SlackScraper (Message Parsing)
    ├── extract_text_from_blocks()
    ├── extract_text_from_attachments()
    ├── simplify_message()
    ├── simplify_message_extended()
    └── extract_thread_metadata()

SlackBlockBuilder (Message Formatting)
    ├── format_prompt_change_alert()
    ├── format_chat_response()
    └── create_error_message()
```

---


## SlackConfig (Token Management)

Centralized management of Slack bot tokens for multiple agents.

```python
class SlackConfig:
    ATHENA_TOKEN = os.getenv("SLACK_ATHENA_TOKEN")
    AIMEE_TOKEN = os.getenv("SLACK_AIMEE_TOKEN")
    CANARY_TOKEN = os.getenv("SLACK_CANARY_TOKEN")
    PROMETHEUS_TOKEN = os.getenv("SLACK_PROMETHEUS_TOKEN")
    QUILL_TOKEN = os.getenv("SLACK_QUILL_TOKEN")

    @classmethod
    def get_token(
        cls,
        agent_id: Optional[str] = None,
        override_token: Optional[str] = None
    ) -> str:
        """
        Get token by priority:
        1. override_token if provided
        2. agent_map[agent_id] if agent_id provided
        3. ATHENA_TOKEN as fallback
        """
```

**Supported Agents:** `"athena"`, `"aimee"`, `"canary"`, `"quill"`, `"prometheus"`

---

## SlackHttpClient (Async Client with Retry)

Low-level async HTTP client with exponential backoff retry strategy.

```python
class SlackHttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 4.0,
        jitter_seconds: float = 0.1,
    ): ...

    async def request(
        self,
        *,
        method: str,
        url: str,
        token: str,
        json_data: Any = None,
        data: Any = None,
        headers: Dict | None = None,
        params: Dict | None = None,
    ) -> Dict[str, Any]: ...
```

**Retry Strategy:**
1. Detects rate limiting (HTTP 429 or "ratelimited" error)
2. Detects server errors (HTTP 5xx)
3. Exponential backoff: `delay = backoff_seconds * (2 ^ (attempt - 1))`
4. Adds random jitter to prevent thundering herd
5. Respects Slack's `Retry-After` header if provided

---

## SlackScraper (Message Parsing)

Extract and simplify raw Slack message data into readable formats.

### extract_text_from_blocks()

```python
@staticmethod
def extract_text_from_blocks(blocks: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts text from Slack Block Kit elements:
    - section blocks: text.text
    - rich_text blocks: rich_text_section elements
    - context blocks: element text field
    """
```

### extract_text_from_attachments()

```python
@staticmethod
def extract_text_from_attachments(attachments: List[Dict[str, Any]]) -> List[str]:
    """Handles legacy Slack attachments."""
```

### simplify_message()

```python
@staticmethod
def simplify_message(raw: Dict[str, Any]) -> SimplifiedMessage:
    """
    Returns SimplifiedMessage with:
    - ts: Message timestamp
    - sender: Human-readable sender name
    - is_bot: Boolean flag
    - content: Concatenated, deduplicated, normalized text
    """
```

### simplify_message_extended()

```python
@staticmethod
def simplify_message_extended(raw: Dict[str, Any]) -> ExtendedSimplifiedMessage:
    """
    Returns ExtendedSimplifiedMessage with additional fields:
    - timestamp_utc: Parsed datetime
    - user_id: Slack user ID
    - reply_count: Number of thread replies
    - message_url: Permalink to message
    """
```

### extract_thread_metadata()

```python
@staticmethod
def extract_thread_metadata(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
    - thread_created_at: Earliest message timestamp
    - thread_last_activity_at: Latest message timestamp
    - human_participants: List of non-bot user IDs
    """
```

---

## SlackService (Read/Write Operations)

High-level API for Slack interactions.

```python
class SlackService:
    def __init__(
        self,
        storage: SubscriptionStorage = None,  # Default: InMemorySubscriptionStorage
        client: SlackHttpClient = None         # Default: shared global client
    ): ...
```

### Write Operations

#### post_message()

```python
async def post_message(
    channel: str,
    options: PostMessageOptions,
    agent_id: Optional[str] = None
) -> SlackResponse:
    """
    Posts a message to a channel.
    - Validates token and message content
    - Applies payload limits (text: 40000 chars, block text: 3000 chars)
    - Posts to chat.postMessage endpoint
    """
```

#### update_message()

```python
async def update_message(
    channel: str,
    ts: str,
    options: PostMessageOptions,
    agent_id: Optional[str] = None
) -> SlackResponse:
    """Updates an existing message by timestamp."""
```

#### post_threaded_message()

```python
async def post_threaded_message(
    channel: str,
    thread_ts: str,
    options: PostMessageOptions,
    agent_id: Optional[str] = None
) -> SlackResponse:
    """
    Posts a reply in a thread.
    - Validates thread exists
    - Sets thread_ts in options
    """
```

### Read/Scrape Operations

#### get_thread_replies()

```python
async def get_thread_replies(
    channel: str,
    thread_ts: str,
    agent_id: Optional[str] = None
) -> SlackResponse:
    """
    Retrieves all replies in a thread with automatic pagination.
    - Endpoint: conversations.replies
    - Batch size: 200
    """
```

#### get_channel_history()

```python
async def get_channel_history(
    channel: str,
    limit: int = 100,
    agent_id: Optional[str] = None
) -> SlackResponse:
    """Fetches channel message history with pagination."""
```

### Interaction Operations

#### open_modal()

```python
async def open_modal(
    trigger_id: str,
    view: Dict,
    agent_id: Optional[str] = None
) -> SlackResponse:
    """Opens a modal view dialog."""
```

#### publish_app_home()

```python
async def publish_app_home(
    agent_id: str,
    user_id: str
) -> None:
    """Publishes the App Home tab view for a user."""
```

#### download_file()

```python
async def download_file(
    file: SlackFile,
    agent_id: Optional[str] = None
) -> FileDownloadResponse:
    """
    Downloads a file from Slack with redirect handling.
    - Handles up to 5 redirects to preserve auth headers
    - Validates content type
    """
```

### Multi-Channel Broadcasting

#### post_to_subscribed_channels()

```python
async def post_to_subscribed_channels(
    agent_id: str,
    event_type: str,
    event_data: Dict[str, str],
    message_options: PostMessageOptions
) -> MultiChannelPostResult:
    """Posts to all subscribed channels matching event type and filters."""
```

#### post_to_all_threads()

```python
async def post_to_all_threads(
    threads: List[SlackThread],
    message_options: PostMessageOptions,
    agent_id: str
) -> None:
    """Broadcasts reply to multiple tracked threads."""
```

---

## SlackBlockBuilder (Block Kit Formatting)

Utility class for constructing Slack Block Kit formatted messages.

### format_prompt_change_alert()

```python
@staticmethod
def format_prompt_change_alert(
    prompt_name: str,
    event_type: str,
    prompt_version: Optional[Union[int, str]] = None,
    prompt_id: Optional[str] = None,
    prompt_url: Optional[str] = None
) -> Dict[str, Any]:
    """Creates notification for Langfuse prompt updates."""
```

### format_chat_response()

```python
@staticmethod
def format_chat_response(
    message: str,
    citations: List[Dict] = None,
    is_learning_worthy: bool = False,
    is_feature_request: bool = False
) -> Dict[str, Any]:
    """
    Creates AI chat response message with:
    - Main message block
    - Optional citations block
    - Optional action buttons (Save as Learning, Create Feature Request)
    """
```

### create_error_message()

```python
@staticmethod
def create_error_message(error: str) -> Dict[str, Any]:
    """Creates error alert message with section block."""
```

---

## TypedDicts and Dataclasses

### SlackResponse

```python
class SlackResponse(TypedDict, total=False):
    success: bool
    ts: Optional[str]                        # Message timestamp
    channel: Optional[str]                   # Channel ID
    error: Optional[str]                     # Error message
    status: Optional[int]                    # HTTP status code
    messages: Optional[List[Dict[str, Any]]] # Array of message objects
    user: Optional[Dict[str, Any]]
```

### SimplifiedMessage

```python
class SimplifiedMessage(TypedDict):
    ts: str           # Message timestamp
    sender: str       # Sender name/bot name
    is_bot: bool      # Is from a bot
    content: str      # Consolidated text content
```

### ExtendedSimplifiedMessage

```python
class ExtendedSimplifiedMessage(TypedDict):
    ts: str
    timestamp_utc: Optional[datetime]   # Parsed UTC datetime
    sender: str
    user_id: Optional[str]              # Slack user ID
    is_bot: bool
    content: str
    reply_count: int                    # Number of thread replies
    message_url: Optional[str]          # Message permalink
```

### PostMessageOptions

```python
class PostMessageOptions(TypedDict, total=False):
    text: Optional[str]
    blocks: Optional[List[Dict[str, Any]]]
    attachments: Optional[List[Dict[str, Any]]]
    thread_ts: Optional[str]
    reply_broadcast: Optional[bool]
    unfurl_links: Optional[bool]
    unfurl_media: Optional[bool]
    slackToken: Optional[str]           # Override token
```

### SlackFile

```python
@dataclass
class SlackFile:
    id: str
    name: str
    mimetype: str
    size: int
    url_private: str
    url_private_download: Optional[str] = None
```

### FileDownloadResponse

```python
@dataclass
class FileDownloadResponse:
    success: bool
    buffer: Optional[bytes] = None
    filename: Optional[str] = None
    mimetype: Optional[str] = None
    error: Optional[str] = None
```

### SlackSubscription

```python
@dataclass
class SlackSubscription:
    id: str
    agent_id: str                 # Agent receiving events
    event_type: str               # Type of event
    channel_id: str               # Where to post
    filters: Dict[str, str]       # Event filter conditions
    created_by_slack_user_id: str
    created_by_username: Optional[str]
    active: bool
    created_at: str
    updated_at: str
```

### SlackThread

```python
@dataclass
class SlackThread:
    channel_id: str
    thread_ts: str       # Thread message timestamp
    posted_at: str       # ISO format timestamp
```

### MultiChannelPostResult

```python
@dataclass
class MultiChannelPostResult:
    threads: List[SlackThread]
    errors: List[Dict[str, str]]
```

### SlackFormattingCompliance Metric

Located at `src/shared/metrics/slack_compliance.py`, this metric validates AI
outputs comply with Slack formatting rules.

```python
class SlackFormattingCompliance(BaseMetric):
    key = "slack_compliance"
    required_fields = ["actual_output"]
    default_threshold = 1.0
```

**Validation Checks:**
1. **Double Asterisks:** Detects `**text**` (should be `*text*` in Slack)
2. **Headers:** Detects `# Header` markdown (forbidden in Slack)
3. **Unwrapped Data:** Detects `$500` or `50%` not in backticks

**Scoring:** Deducts 0.25 per issue type found, minimum 0.0

---

## Formatting Rules and Compliance

### General Slack Formatting Rules

```python
SLACK_FORMATTING_RULES = """
- NO using # for headers (Slack doesn't render them)
- Use *single asterisks* for bold (NOT **double**)
- Use backticks for variables, numbers, and technical terms
"""
```

### Chat-Specific Formatting Rules

```python
SLACK_CHAT_FORMATTING_RULES = """
- Same as above, plus:
- Reserve backticks for citations and technical terms only
"""
```

### Valid Event Types

```python
VALID_EVENT_TYPES = {
    "athena": ["referrals"],
    "quill": ["binds"],
    "canary": [
        "evaluations", "openprs", "weekly-summary",
        "release-train", "releases", "new-users",
        "new-repo", "new-swallow-projects"
    ],
    "aimee": ["inquiries"],
    "prometheus": ["dust-audit"],
}
```

---

## Code Examples

### Basic Message Posting

```python
from eval_workbench.shared.slack.service import SlackService, PostMessageOptions

slack = SlackService()

# Post simple message
result = await slack.post_message(
    channel="C1234567890",
    options=PostMessageOptions(
        text="Hello from Athena!",
    ),
    agent_id="athena",
)

if result["success"]:
    print(f"Message posted: {result['ts']}")
else:
    print(f"Error: {result['error']}")
```

### Posting with Block Kit

```python
from eval_workbench.shared.slack.service import SlackService, SlackBlockBuilder

slack = SlackService()

# Create formatted response
options = SlackBlockBuilder.format_chat_response(
    message="Here's your analysis...",
    citations=[
        {"url": "https://example.com", "source": "Example Doc"},
    ],
    is_learning_worthy=True,
)

result = await slack.post_message(
    channel="C1234567890",
    options=options,
    agent_id="athena",
)
```

### Reading Thread Replies

```python
from eval_workbench.shared.slack.service import SlackService, SlackScraper

slack = SlackService()

# Get thread replies
result = await slack.get_thread_replies(
    channel="C1234567890",
    thread_ts="1700000000.000000",
    agent_id="athena",
)

if result["success"]:
    messages = result["messages"]

    # Simplify messages for processing
    simplified = [
        SlackScraper.simplify_message_extended(msg)
        for msg in messages
    ]

    # Extract thread metadata
    metadata = SlackScraper.extract_thread_metadata(messages)
    print(f"Thread created: {metadata['thread_created_at']}")
    print(f"Human participants: {metadata['human_participants']}")
```

### Multi-Channel Broadcasting

```python
from eval_workbench.shared.slack.service import SlackService, PostMessageOptions

slack = SlackService()

result = await slack.post_to_subscribed_channels(
    agent_id="canary",
    event_type="releases",
    event_data={"repo": "swallow"},
    message_options=PostMessageOptions(
        text="New release deployed!",
    ),
)

print(f"Posted to {len(result.threads)} channels")
print(f"Errors: {len(result.errors)}")
```

### Downloading Files

```python
from eval_workbench.shared.slack.service import SlackService, SlackFile

slack = SlackService()

file = SlackFile(
    id="F1234567890",
    name="document.pdf",
    mimetype="application/pdf",
    size=1024000,
    url_private="https://files.slack.com/...",
)

response = await slack.download_file(file, agent_id="athena")

if response.success:
    with open(response.filename, "wb") as f:
        f.write(response.buffer)
else:
    print(f"Download failed: {response.error}")
```

---

## Exports

```python
from eval_workbench.shared.slack.service import (
    # Configuration
    SlackConfig,
    SLACK_FORMATTING_RULES,
    SLACK_CHAT_FORMATTING_RULES,
    VALID_EVENT_TYPES,
    AGENT_INFO,

    # TypedDicts
    SlackResponse,
    SimplifiedMessage,
    ExtendedSimplifiedMessage,
    PostMessageOptions,

    # Dataclasses
    SlackFile,
    FileDownloadResponse,
    SlackSubscription,
    SlackThread,
    MultiChannelPostResult,

    # HTTP Client
    SlackHttpClient,
    get_shared_slack_client,

    # Storage
    SubscriptionStorage,
    InMemorySubscriptionStorage,

    # Main Classes
    SlackScraper,
    SlackService,
    SlackBlockBuilder,
)

from eval_workbench.shared.metrics.slack_compliance import (
    SlackFormattingCompliance,
)
```
