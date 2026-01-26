import os
import json
import logging
import re
import asyncio
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, TypedDict
from datetime import datetime
from dataclasses import dataclass
import aiohttp

# =============================================================================
# Configuration & Logging
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlackService")


class SlackConfig:
    """Environment configuration for Slack Tokens."""

    ATHENA_TOKEN = os.getenv("SLACK_ATHENA_TOKEN", "")
    AIMEE_TOKEN = os.getenv("SLACK_AIMEE_TOKEN", "")
    CANARY_TOKEN = os.getenv("SLACK_CANARY_TOKEN", "")
    PROMETHEUS_TOKEN = os.getenv("SLACK_PROMETHEUS_TOKEN", "")
    QUILL_TOKEN = os.getenv("SLACK_QUILL_TOKEN", "")

    @classmethod
    def get_token(
        cls, agent_id: Optional[str] = None, override_token: Optional[str] = None
    ) -> str:
        """Resolves the correct token for a given agent ID."""
        if override_token:
            return override_token

        agent_map = {
            "aimee": cls.AIMEE_TOKEN,
            "canary": cls.CANARY_TOKEN,
            "quill": cls.QUILL_TOKEN,
            "prometheus": cls.PROMETHEUS_TOKEN,
        }
        return agent_map.get(agent_id, cls.ATHENA_TOKEN)


# =============================================================================
# Types & Interfaces
# =============================================================================


class SlackResponse(TypedDict, total=False):
    success: bool
    ts: Optional[str]
    channel: Optional[str]
    error: Optional[str]
    status: Optional[int]
    messages: Optional[List[Dict[str, Any]]]
    user: Optional[Dict[str, Any]]


class SimplifiedMessage(TypedDict):
    ts: str
    sender: str
    is_bot: bool
    content: str


class ExtendedSimplifiedMessage(TypedDict):
    """Extended message format with timestamp parsing and user ID."""

    ts: str
    timestamp_utc: Optional[datetime]
    sender: str
    user_id: Optional[str]
    is_bot: bool
    content: str
    reply_count: int
    message_url: Optional[str]


class PostMessageOptions(TypedDict, total=False):
    text: Optional[str]
    blocks: Optional[List[Dict[str, Any]]]
    attachments: Optional[List[Dict[str, Any]]]
    thread_ts: Optional[str]
    reply_broadcast: Optional[bool]
    unfurl_links: Optional[bool]
    unfurl_media: Optional[bool]
    slackToken: Optional[str]


@dataclass
class SlackFile:
    id: str
    name: str
    mimetype: str
    size: int
    url_private: str
    url_private_download: Optional[str] = None


@dataclass
class FileDownloadResponse:
    success: bool
    buffer: Optional[bytes] = None
    filename: Optional[str] = None
    mimetype: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SlackSubscription:
    id: str
    agent_id: str
    event_type: str
    channel_id: str
    filters: Dict[str, str]
    created_by_slack_user_id: str
    created_by_username: Optional[str]
    active: bool
    created_at: str
    updated_at: str


@dataclass
class SlackThread:
    channel_id: str
    thread_ts: str
    posted_at: str


@dataclass
class MultiChannelPostResult:
    threads: List[SlackThread]
    errors: List[Dict[str, str]]


# =============================================================================
# Constants & Formatting Rules
# =============================================================================

SLACK_FORMATTING_RULES = """
CRITICAL FORMATTING RULES (Slack mrkdwn format):
- NEVER use # or ## or ### for headers
- Use *bold* for emphasis (single asterisks only)
- Wrap key variables in backticks: `$50,000`, `15%`, `Restaurant`
- Use simple formatting - headers will be added programmatically
- Keep responses concise and professional
- Character limit: 2800 characters total
""".strip()

SLACK_CHAT_FORMATTING_RULES = """
SLACK FORMATTING RULES (mrkdwn format):
- NEVER use # or ## or ### for headers
- Use *bold* for emphasis (single asterisks only)
- Use _italics_ for secondary emphasis (single underscores only)
- **CRITICAL - Backticks reserved for citations ONLY:** Use backticks `[1]`, `[2]`, etc. for citation numbers.
- Use bullet points with - or • for lists
- Keep responses clear and professional
""".strip()

VALID_EVENT_TYPES = {
    "athena": ["referrals"],
    "quill": ["binds"],
    "canary": [
        "evaluations",
        "openprs",
        "weekly-summary",
        "release-train",
        "releases",
        "new-users",
        "new-repo",
        "new-swallow-projects",
    ],
    "aimee": ["inquiries"],
    "prometheus": ["dust-audit"],
}

AGENT_INFO = {
    "athena": {
        "name": "Athena",
        "emoji": ":athena:",
        "description": "AI underwriting assistant for insurance quote referrals.",
        "hasChat": True,
    },
    "quill": {
        "name": "Quill",
        "emoji": ":quill:",
        "description": "AI assistant for policy binding workflows.",
        "hasChat": True,
    },
    "canary": {
        "name": "Canary",
        "emoji": ":canary:",
        "description": "AI engineering manager for PR evaluation and code quality analysis.",
        "hasChat": True,
    },
    "aimee": {
        "name": "Aimee",
        "emoji": ":aimee:",
        "description": "AI appetite chatbot for insurance policy inquiries.",
        "hasChat": True,
    },
}

# =============================================================================
# Slack HTTP Client (Retries + Rate Limits)
# =============================================================================


class SlackHttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 4.0,
        jitter_seconds: float = 0.1,
    ):
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_attempts = max(1, max_attempts)
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

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
    ) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "eval-workbench-slack-service",
            }
        )

        for attempt in range(1, self._max_attempts + 1):
            try:
                session = await self._get_session()
                async with session.request(
                    method,
                    url,
                    json=json_data,
                    data=data,
                    headers=headers,
                    params=params,
                ) as response:
                    status = response.status
                    text = await response.text()
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = {
                            "ok": False,
                            "error": f"Invalid JSON response: {text[:100]}",
                        }

                    payload["_status"] = status

                    if status == 429 or payload.get("error") == "ratelimited":
                        retry_after = response.headers.get("Retry-After")
                        if retry_after is not None:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = None
                        else:
                            delay = None
                        if delay is None:
                            delay = min(
                                self._max_backoff_seconds,
                                self._backoff_seconds * (2 ** (attempt - 1)),
                            )
                        delay += random.uniform(0, self._jitter_seconds)
                        logger.warning(
                            "[Slack API] Rate limited (attempt %s), sleeping %.2fs",
                            attempt,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if status >= 500 and attempt < self._max_attempts:
                        delay = min(
                            self._max_backoff_seconds,
                            self._backoff_seconds * (2 ** (attempt - 1)),
                        ) + random.uniform(0, self._jitter_seconds)
                        logger.warning(
                            "[Slack API] Server error %s (attempt %s), sleeping %.2fs",
                            status,
                            attempt,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if not payload.get("ok"):
                        logger.error(
                            "[Slack API] Request failed: %s", payload.get("error")
                        )

                    return payload
            except Exception as exc:
                if attempt >= self._max_attempts:
                    logger.error("[Slack API] Request failed: %s", exc)
                    return {"ok": False, "error": str(exc), "_status": 0}
                delay = min(
                    self._max_backoff_seconds,
                    self._backoff_seconds * (2 ** (attempt - 1)),
                ) + random.uniform(0, self._jitter_seconds)
                logger.warning(
                    "[Slack API] Request error (attempt %s): %s", attempt, exc
                )
                await asyncio.sleep(delay)

        return {"ok": False, "error": "Max retry attempts exceeded", "_status": 0}


_SHARED_SLACK_CLIENT: SlackHttpClient | None = None


def get_shared_slack_client() -> SlackHttpClient:
    global _SHARED_SLACK_CLIENT
    if _SHARED_SLACK_CLIENT is None:
        _SHARED_SLACK_CLIENT = SlackHttpClient()
    return _SHARED_SLACK_CLIENT


# =============================================================================
# Subscription Storage Interface
# =============================================================================


class SubscriptionStorage(ABC):
    """
    Abstract interface for subscription persistence.
    """

    @abstractmethod
    async def create_subscription(self, sub: SlackSubscription) -> SlackSubscription:
        pass

    @abstractmethod
    async def get_subscriptions(
        self, agent_id: str, event_type: str = None, active_only: bool = True
    ) -> List[SlackSubscription]:
        pass

    @abstractmethod
    async def deactivate_subscription(
        self, agent_id: str, event_type: str, channel_id: str, filters: Dict[str, str]
    ) -> bool:
        pass

    @abstractmethod
    async def get_by_user(self, agent_id: str, user_id: str) -> List[SlackSubscription]:
        pass


class InMemorySubscriptionStorage(SubscriptionStorage):
    def __init__(self):
        self._subs: List[SlackSubscription] = []

    async def create_subscription(self, sub: SlackSubscription) -> SlackSubscription:
        self._subs.append(sub)
        return sub

    async def get_subscriptions(
        self, agent_id: str, event_type: str = None, active_only: bool = True
    ) -> List[SlackSubscription]:
        return [
            s
            for s in self._subs
            if s.agent_id == agent_id
            and (event_type is None or s.event_type == event_type)
            and (not active_only or s.active)
        ]

    async def deactivate_subscription(
        self, agent_id: str, event_type: str, channel_id: str, filters: Dict[str, str]
    ) -> bool:
        count = 0
        for s in self._subs:
            if (
                s.agent_id == agent_id
                and s.event_type == event_type
                and s.channel_id == channel_id
                and s.filters == filters
            ):
                s.active = False
                count += 1
        return count > 0

    async def get_by_user(self, agent_id: str, user_id: str) -> List[SlackSubscription]:
        return [
            s
            for s in self._subs
            if s.agent_id == agent_id
            and s.created_by_slack_user_id == user_id
            and s.active
        ]


# =============================================================================
# Slack Scraping Helpers
# =============================================================================


class SlackScraper:
    """Utilities for parsing and simplifying raw Slack messages."""

    @staticmethod
    def extract_text_from_blocks(blocks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract readable text from Slack blocks.
        Handles section blocks, rich_text blocks, context blocks, etc.
        """
        if not blocks:
            return []

        texts: List[str] = []

        for block in blocks:
            b_type = block.get("type")

            if b_type == "section":
                text_obj = block.get("text")
                if text_obj and text_obj.get("text"):
                    texts.append(text_obj["text"])

            elif b_type == "rich_text":
                # Extract from rich_text_section elements
                elements = block.get("elements", [])
                for element in elements:
                    if element.get("type") == "rich_text_section":
                        sub_elements = element.get("elements", [])
                        # Filter for text nodes and join them
                        section_text = "".join(
                            [
                                e.get("text", "")
                                for e in sub_elements
                                if e.get("type") == "text"
                            ]
                        )
                        if section_text:
                            texts.append(section_text)

            elif b_type == "context":
                elements = block.get("elements", [])
                for element in elements:
                    if element.get("text"):
                        texts.append(element["text"])

        return texts

    @staticmethod
    def extract_text_from_attachments(attachments: List[Dict[str, Any]]) -> List[str]:
        """Extract readable text from Slack attachments."""
        if not attachments:
            return []

        texts: List[str] = []

        for attachment in attachments:
            # Get text from attachment blocks
            if "blocks" in attachment:
                texts.extend(
                    SlackScraper.extract_text_from_blocks(attachment["blocks"])
                )

            # Fallback to attachment text/fallback
            if attachment.get("text"):
                texts.append(attachment["text"])

        return texts

    @staticmethod
    def simplify_message(raw: Dict[str, Any]) -> SimplifiedMessage:
        """Simplify a raw Slack message to just the essential content."""
        is_bot = "bot_id" in raw

        # Determine sender name
        bot_profile = raw.get("bot_profile", {})
        sender = bot_profile.get("name") if bot_profile else None

        if not sender:
            sender = "Bot" if is_bot else raw.get("user", "Unknown")

        content_parts: List[str] = []

        # Start with main text field
        text = raw.get("text", "")
        if text and text.strip():
            content_parts.append(text)

        # Extract from blocks
        block_texts = SlackScraper.extract_text_from_blocks(raw.get("blocks", []))
        for t in block_texts:
            if t and t not in content_parts:
                content_parts.append(t)

        # Extract from attachments
        attachment_texts = SlackScraper.extract_text_from_attachments(
            raw.get("attachments", [])
        )
        for t in attachment_texts:
            if t and t not in content_parts:
                content_parts.append(t)

        # Join content and cleanup
        content = "\n\n".join([c for c in content_parts if c])
        # Regex to replace 3 or more newlines with 2
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        return {
            "ts": raw.get("ts", ""),
            "sender": sender,
            "is_bot": is_bot,
            "content": content,
        }

    @staticmethod
    def simplify_message_extended(raw: Dict[str, Any]) -> ExtendedSimplifiedMessage:
        """
        Extended message simplification with timestamp parsing and user ID extraction.

        Includes all fields from simplify_message plus:
        - timestamp_utc: Parsed datetime from ts
        - user_id: Slack user ID
        - reply_count: Number of replies
        - message_url: Link to the message (if available)
        """
        base = SlackScraper.simplify_message(raw)

        # Parse timestamp
        ts_str = raw.get("ts", "")
        timestamp_utc = None
        if ts_str:
            try:
                ts_float = float(ts_str)
                timestamp_utc = datetime.utcfromtimestamp(ts_float)
            except (ValueError, TypeError):
                pass

        return {
            **base,
            "timestamp_utc": timestamp_utc,
            "user_id": raw.get("user", ""),
            "reply_count": raw.get("reply_count", 0),
            "message_url": raw.get("permalink"),
        }

    @staticmethod
    def extract_thread_metadata(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract thread-level metadata from a list of messages.

        Returns:
            Dict with thread_created_at, thread_last_activity_at, and human_participants
        """
        timestamps = []
        human_participants = set()

        for msg in messages:
            # Parse timestamp
            ts_str = msg.get("ts", "")
            if ts_str:
                try:
                    ts_float = float(ts_str)
                    timestamps.append(datetime.utcfromtimestamp(ts_float))
                except (ValueError, TypeError):
                    pass

            # Track human participants
            is_bot = "bot_id" in msg
            user_id = msg.get("user")
            if not is_bot and user_id:
                human_participants.add(user_id)

        return {
            "thread_created_at": min(timestamps) if timestamps else None,
            "thread_last_activity_at": max(timestamps) if timestamps else None,
            "human_participants": list(human_participants),
        }


# =============================================================================
# Core Slack Service
# =============================================================================


class SlackService:
    def __init__(
        self, storage: SubscriptionStorage = None, client: "SlackHttpClient" = None
    ):
        self.storage = storage or InMemorySubscriptionStorage()
        self._client = client or get_shared_slack_client()

    def _validate_message_options(self, options: PostMessageOptions) -> Optional[str]:
        if options.get("text"):
            return None
        if options.get("blocks"):
            return None
        if options.get("attachments"):
            return None
        return "Message text, blocks, or attachments are required"

    def _apply_payload_limits(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        max_text = 40000
        max_block_text = 3000
        truncated_suffix = "…"

        def _truncate(value: str, limit: int) -> str:
            if len(value) <= limit:
                return value
            return value[: max(0, limit - len(truncated_suffix))] + truncated_suffix

        trimmed = dict(payload)
        if isinstance(trimmed.get("text"), str):
            trimmed["text"] = _truncate(trimmed["text"], max_text)

        blocks = trimmed.get("blocks")
        if isinstance(blocks, list):
            new_blocks = []
            for block in blocks:
                if isinstance(block, dict):
                    block_copy = dict(block)
                    text_obj = block_copy.get("text")
                    if isinstance(text_obj, dict) and isinstance(
                        text_obj.get("text"), str
                    ):
                        text_obj = dict(text_obj)
                        text_obj["text"] = _truncate(text_obj["text"], max_block_text)
                        block_copy["text"] = text_obj
                    new_blocks.append(block_copy)
                else:
                    new_blocks.append(block)
            trimmed["blocks"] = new_blocks

        return trimmed

    async def _request(
        self,
        method: str,
        url: str,
        token: str,
        json_data: Any = None,
        data: Any = None,
        headers: Dict = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        """Base internal request helper."""
        response = await self._client.request(
            method=method,
            url=url,
            token=token,
            json_data=json_data,
            data=data,
            headers=headers,
            params=params,
        )
        return response

    # --- Write Operations ---

    async def post_message(
        self, channel: str, options: PostMessageOptions, agent_id: Optional[str] = None
    ) -> SlackResponse:
        """Posts a message to a Slack channel."""
        token = SlackConfig.get_token(agent_id, options.get("slackToken"))

        if not token:
            return {"success": False, "error": "Slack token not configured"}
        if not channel:
            return {"success": False, "error": "Channel ID is required"}
        payload_error = self._validate_message_options(options)
        if payload_error:
            return {"success": False, "error": payload_error}

        # Filter out local options
        payload = {k: v for k, v in options.items() if k != "slackToken"}
        payload["channel"] = channel
        payload = self._apply_payload_limits(payload)

        data = await self._request(
            "POST", "https://slack.com/api/chat.postMessage", token, json_data=payload
        )

        if not data.get("ok"):
            logger.error(f"[Slack API] Failed to post message: {data.get('error')}")
            return {
                "success": False,
                "error": data.get("error"),
                "status": data.get("_status"),
            }

        logger.info("[Slack API] Message posted to %s", data.get("channel"))
        return {"success": True, "ts": data.get("ts"), "channel": data.get("channel")}

    async def update_message(
        self,
        channel: str,
        ts: str,
        options: PostMessageOptions,
        agent_id: Optional[str] = None,
    ) -> SlackResponse:
        """Updates an existing message."""
        token = SlackConfig.get_token(agent_id, options.get("slackToken"))
        if not token:
            return {"success": False, "error": "Slack token not configured"}
        if not channel or not ts:
            return {"success": False, "error": "Channel ID and ts are required"}
        payload_error = self._validate_message_options(options)
        if payload_error:
            return {"success": False, "error": payload_error}

        payload = {k: v for k, v in options.items() if k != "slackToken"}
        payload.update({"channel": channel, "ts": ts})
        payload = self._apply_payload_limits(payload)

        data = await self._request(
            "POST", "https://slack.com/api/chat.update", token, json_data=payload
        )

        if not data.get("ok"):
            return {
                "success": False,
                "error": data.get("error"),
                "status": data.get("_status"),
            }
        return {"success": True, "ts": data.get("ts"), "channel": data.get("channel")}

    async def post_threaded_message(
        self,
        channel: str,
        thread_ts: str,
        options: PostMessageOptions,
        agent_id: Optional[str] = None,
    ) -> SlackResponse:
        """Posts a threaded reply, validating existence first."""
        if not channel or not thread_ts:
            return {"success": False, "error": "Channel ID and thread_ts are required"}
        # Simple existence check using fetch (could be optimized)
        exists_check = await self.get_thread_replies(channel, thread_ts, agent_id)
        if not exists_check.get("success") or not exists_check.get("messages"):
            logger.error(f"[Slack API] Thread {thread_ts} does not exist in {channel}")
            return {"success": False, "error": "thread_not_found"}

        options["thread_ts"] = thread_ts
        return await self.post_message(channel, options, agent_id)

    # --- Read/Scrape Operations ---

    async def get_thread_replies(
        self, channel: str, thread_ts: str, agent_id: Optional[str] = None
    ) -> SlackResponse:
        """Retrieves all replies in a thread with automatic pagination."""
        token = SlackConfig.get_token(agent_id)
        if not token:
            return {"success": False, "error": "Slack token not configured"}

        url = "https://slack.com/api/conversations.replies"
        all_messages = []
        next_cursor = None

        try:
            while True:
                params = {"channel": channel, "ts": thread_ts, "limit": "200"}
                if next_cursor:
                    params["cursor"] = next_cursor

                data = await self._request("GET", url, token, params=params)

                if not data.get("ok"):
                    return {"success": False, "error": data.get("error")}

                messages = data.get("messages", [])
                all_messages.extend(messages)

                next_cursor = data.get("response_metadata", {}).get("next_cursor")
                if not next_cursor:
                    break

            return {"success": True, "messages": all_messages}

        except Exception as e:
            logger.error(f"[Slack API] Error retrieving thread: {e}")
            return {"success": False, "error": str(e)}

    async def get_channel_history(
        self, channel: str, limit: int = 100, agent_id: Optional[str] = None
    ) -> SlackResponse:
        """Fetches channel history (messages) with optional pagination."""
        token = SlackConfig.get_token(agent_id)
        if not token:
            return {"success": False, "error": "Slack token not configured"}

        url = "https://slack.com/api/conversations.history"
        all_messages = []
        next_cursor = None
        fetched_count = 0

        try:
            while True:
                batch_size = min(200, limit - fetched_count)
                params = {"channel": channel, "limit": str(batch_size)}
                if next_cursor:
                    params["cursor"] = next_cursor

                data = await self._request("GET", url, token, params=params)

                if not data.get("ok"):
                    return {"success": False, "error": data.get("error")}

                messages = data.get("messages", [])
                all_messages.extend(messages)
                fetched_count += len(messages)

                next_cursor = data.get("response_metadata", {}).get("next_cursor")

                if not next_cursor or fetched_count >= limit:
                    break

            return {"success": True, "messages": all_messages}
        except Exception as e:
            logger.error(f"[Slack API] Error getting channel history: {e}")
            return {"success": False, "error": str(e)}

    # --- Interaction Operations ---

    async def validate_thread_exists(
        self, channel: str, thread_ts: str, agent_id: Optional[str] = None
    ) -> bool:
        """Checks if a thread exists and is valid."""
        check = await self.get_thread_replies(channel, thread_ts, agent_id)
        return check.get("success", False) and bool(check.get("messages"))

    async def open_modal(
        self, trigger_id: str, view: Dict, agent_id: Optional[str] = None
    ) -> SlackResponse:
        """Opens a modal view."""
        token = SlackConfig.get_token(agent_id)
        payload = {"trigger_id": trigger_id, "view": view}
        data = await self._request(
            "POST", "https://slack.com/api/views.open", token, json_data=payload
        )

        if not data.get("ok"):
            return {"success": False, "error": data.get("error")}
        return {"success": True}

    async def publish_app_home(self, agent_id: str, user_id: str) -> None:
        """Publishes the App Home tab for a user."""
        blocks = await self._build_app_home_blocks(agent_id, user_id)
        token = SlackConfig.get_token(agent_id)

        if not token:
            logger.error(f"[App Home] No token for agent {agent_id}")
            return

        payload = {"user_id": user_id, "view": {"type": "home", "blocks": blocks}}
        data = await self._request(
            "POST", "https://slack.com/api/views.publish", token, json_data=payload
        )

        if not data.get("ok"):
            logger.error(f"[App Home] Error publishing view: {data.get('error')}")
        else:
            logger.info(f"[App Home] Published for {agent_id} to user {user_id}")

    async def download_file(
        self, file: SlackFile, agent_id: Optional[str] = None
    ) -> FileDownloadResponse:
        """
        Downloads a file from Slack. Handles manual redirects to preserve Auth headers.
        """
        token = SlackConfig.get_token(agent_id)
        download_url = file.url_private_download or file.url_private

        if not token:
            return FileDownloadResponse(success=False, error="Token missing")
        if not download_url:
            return FileDownloadResponse(success=False, error="No download URL")

        logger.info(f"[Slack API] Downloading file: {file.name} ({file.size} bytes)")

        async with aiohttp.ClientSession() as session:
            current_url = download_url
            redirect_count = 0
            max_redirects = 5

            try:
                while redirect_count < max_redirects:
                    # Manual request to handle redirect logic
                    async with session.get(
                        current_url,
                        headers={"Authorization": f"Bearer {token}"},
                        allow_redirects=False,
                    ) as response:
                        if 300 <= response.status < 400:
                            location = response.headers.get("Location")
                            if not location:
                                return FileDownloadResponse(
                                    success=False, error="Redirect without Location"
                                )
                            current_url = location
                            redirect_count += 1
                            continue

                        if not response.ok:
                            return FileDownloadResponse(
                                success=False, error=f"HTTP {response.status}"
                            )

                        # Check content type for HTML error pages
                        content_type = response.headers.get("Content-Type", "")
                        if "text/html" in content_type:
                            return FileDownloadResponse(
                                success=False,
                                error="Received HTML instead of file (likely auth error)",
                            )

                        content = await response.read()

                        # Size validation
                        if len(content) < file.size * 0.5:
                            # Basic check if it's an HTML error page inside the buffer
                            preview = content[:500].decode("utf-8", errors="ignore")
                            if "<!DOCTYPE" in preview or "<html" in preview:
                                return FileDownloadResponse(
                                    success=False,
                                    error="Buffer appears to be HTML error",
                                )

                        return FileDownloadResponse(
                            success=True,
                            buffer=content,
                            filename=file.name,
                            mimetype=file.mimetype,
                        )

                return FileDownloadResponse(success=False, error="Too many redirects")

            except Exception as e:
                logger.error(f"[Slack API] Download error: {e}")
                return FileDownloadResponse(success=False, error=str(e))

    # =========================================================================
    # Multi-Channel Logic
    # =========================================================================

    async def post_to_subscribed_channels(
        self,
        agent_id: str,
        event_type: str,
        event_data: Dict[str, str],
        message_options: PostMessageOptions,
    ) -> MultiChannelPostResult:
        """Posts a message to all channels subscribed to the event."""

        subs = await self._find_matching_subscriptions(agent_id, event_type, event_data)

        if not subs:
            logger.info(
                f"[Multi-Channel] No subscriptions match for {agent_id}/{event_type}"
            )
            return MultiChannelPostResult(threads=[], errors=[])

        logger.info(f"[Multi-Channel] Posting to {len(subs)} channels")

        tasks = []
        for sub in subs:
            tasks.append(self.post_message(sub.channel_id, message_options, agent_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        threads: List[SlackThread] = []
        errors: List[Dict[str, str]] = []

        for i, res in enumerate(results):
            sub = subs[i]
            if isinstance(res, dict) and res.get("success"):
                threads.append(
                    SlackThread(
                        channel_id=sub.channel_id,
                        thread_ts=res.get("ts"),
                        posted_at=datetime.utcnow().isoformat(),
                    )
                )
            else:
                err_msg = res.get("error") if isinstance(res, dict) else str(res)
                errors.append({"channel_id": sub.channel_id, "error": err_msg})

        return MultiChannelPostResult(threads=threads, errors=errors)

    async def post_to_all_threads(
        self,
        threads: List[SlackThread],
        message_options: PostMessageOptions,
        agent_id: str,
    ) -> None:
        """Broadcasts a reply to all tracked threads."""
        if not threads:
            return

        tasks = [
            self.post_threaded_message(
                t.channel_id, t.thread_ts, message_options, agent_id
            )
            for t in threads
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # =========================================================================
    # Subscription Helpers
    # =========================================================================

    async def _find_matching_subscriptions(
        self, agent_id: str, event_type: str, event_data: Dict[str, str]
    ) -> List[SlackSubscription]:
        all_subs = await self.storage.get_subscriptions(
            agent_id, event_type, active_only=True
        )
        return [s for s in all_subs if self._matches_filters(s.filters, event_data)]

    def _matches_filters(
        self, sub_filters: Dict[str, str], event_data: Dict[str, str]
    ) -> bool:
        if not sub_filters:
            return True

        for key, val in sub_filters.items():
            event_val = event_data.get(key)
            if event_val is None:
                return False

            # Special list logic
            if key in ["repos", "swallow-projects"]:
                allowed = [x.strip().lower() for x in val.split(",")]
                if event_val.lower() not in allowed:
                    return False
                continue

            if event_val.lower() != val.lower():
                return False

        return True

    # =========================================================================
    # App Home Builders
    # =========================================================================

    async def _build_app_home_blocks(self, agent_id: str, user_id: str) -> List[Dict]:
        info = AGENT_INFO.get(agent_id, AGENT_INFO["athena"])
        blocks = []

        # Header
        blocks.append(
            {
                "type": "header",
                "text": {"type": "plain_text", "text": info["name"], "emoji": True},
            }
        )
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": info["description"]}}
        )
        blocks.append({"type": "divider"})

        # Subscriptions Section
        subscriptions = await self.storage.get_by_user(agent_id, user_id)
        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Your Subscriptions",
                    "emoji": True,
                },
            }
        )

        if not subscriptions:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "You haven't created any subscriptions yet.",
                    },
                }
            )
        else:
            sub_list = "\n".join(
                [
                    f"• {s.event_type} ({self._format_filters(s.filters)})"
                    for s in subscriptions
                ]
            )
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Subscriptions you've created:*\n{sub_list}",
                    },
                }
            )

        return blocks

    def _format_filters(self, filters: Dict[str, str]) -> str:
        if not filters:
            return "all"
        return ", ".join([f"{k}:{v}" for k, v in filters.items()])


# =============================================================================
# Block Builders (Static Helpers)
# =============================================================================


class SlackBlockBuilder:
    @staticmethod
    def format_prompt_change_alert(
        prompt_name: str,
        event_type: str,
        prompt_version: Optional[Union[int, str]] = None,
        prompt_id: Optional[str] = None,
        prompt_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        details = [f"*Event:* {event_type}", f"*Prompt:* `{prompt_name}`"]
        if prompt_version is not None:
            details.append(f"*Version:* `{prompt_version}`")
        if prompt_id:
            details.append(f"*Prompt ID:* `{prompt_id}`")
        if prompt_url:
            details.append(f"*Link:* <{prompt_url}|Open prompt>")

        text = "*Langfuse prompt updated*\n" + "\n".join(details)
        return {
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            ],
            "text": f"Langfuse prompt updated: {prompt_name}",
            "unfurl_links": False,
            "unfurl_media": False,
        }

    @staticmethod
    def format_chat_response(
        message: str,
        citations: List[Dict] = None,
        is_learning_worthy: bool = False,
        is_feature_request: bool = False,
    ) -> Dict[str, Any]:
        """Creates a standard Block Kit message with text, citations, and action buttons."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": message}}]

        if citations:
            citation_lines = []
            for c in citations:
                if c.get("url"):
                    citation_lines.append(f"[{c['number']}] <{c['url']}|{c['source']}>")
                else:
                    citation_lines.append(f"[{c['number']}] {c['source']}")

            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "\n".join(citation_lines)}],
                }
            )

        actions = []
        if is_learning_worthy:
            actions.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "💾 Save as Learning"},
                    "action_id": "save_learning",
                    "value": json.dumps(
                        {"timestamp": datetime.now().isoformat()}
                    ),  # Placeholder value
                }
            )

        if is_feature_request:
            actions.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📋 Create Feature Request"},
                    "action_id": "create_feature_request",
                }
            )

        if actions:
            blocks.append({"type": "actions", "elements": actions})

        return {
            "blocks": blocks,
            "text": message,
            "unfurl_links": False,
            "unfurl_media": False,
        }

    @staticmethod
    def create_error_message(error: str) -> Dict[str, Any]:
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"❌ *Error*\n{error}"},
                }
            ],
            "text": f"Error: {error}",
        }
