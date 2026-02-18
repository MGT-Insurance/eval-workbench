import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Mapping, Optional

from axion._core.asyncio import SemaphoreExecutor
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation

from eval_workbench.shared.slack.service import SlackScraper, SlackService

logger = logging.getLogger(__name__)


class SlackExporter:
    """
    Export Slack channel messages (and optionally threads) into DatasetItems.

    Each root message becomes a DatasetItem with a MultiTurnConversation.
    Thread replies are appended in order when `scrape_threads` is enabled.

    Features:
      - Multiple channel IDs
      - Optional thread scraping
      - Optional root-sender filtering
      - Optional sender exclusion
      - Optional message filtering by regex
      - Optional citation block stripping
      - Optional dropping conversations where the first turn is from a user
      - Optional dropping conversations that contain only AI messages
      - Permalink + metadata enrichment (including channel_id on every item)

    Args:
        channel_ids: Slack channel IDs to scrape.
        limit: Max number of root messages per channel.
        reverse: If True, oldest-first order.
        scrape_threads: Include replies for threaded messages.
        filter_sender: Only keep roots from this sender.
        bot_name: Sender name used to mark AI vs human.
        bot_names: Optional list of sender names to mark AI vs human.
        workspace_domain: Slack workspace subdomain for permalinks.
        agent_id: Agent identifier used by Slack service calls.
        slack_token: Optional Slack API token override for read/scrape calls.
        drop_if_first_is_user: Drop if first turn is HumanMessage.
        drop_if_all_ai: Drop if all turns are AIMessage.
        max_concurrent: Max concurrent channel scrapes.
        exclude_senders: Optional list of sender names to omit from conversations.
        drop_message_regexes: Optional list of regex patterns to drop messages.
        strip_citation_block: Remove trailing citation blocks like "[1] ..." lines.
        member_id_to_display_name: Mapping of Slack member IDs to display names.
        human_mention_token: Replacement token for non-mapped Slack member mentions.
        oldest_ts: Optional inclusive lower timestamp bound for channel history pulls.
        latest_ts: Optional inclusive upper timestamp bound for channel history pulls.
        window_days: Relative lookback window in days from "now" (if oldest_ts is unset).
        window_hours: Relative lookback window in hours from "now" (if oldest_ts is unset).
        window_minutes: Relative lookback window in minutes from "now" (if oldest_ts is unset).

    Example:
        exporter = SlackExporter(
            channel_ids=["C09MAP9HR9D", "C09JE5SSP43"],
            limit=10,
            reverse=True,
            scrape_threads=True,
            filter_sender=None,
            bot_name="Athena",
            workspace_domain="mgtinsurance",
            drop_if_first_is_user=True,
            drop_if_all_ai=True,
            max_concurrent=2,
        )
        items = await exporter.execute()
    """

    def __init__(
        self,
        *,
        channel_ids: List[str],
        limit: int = 10,
        reverse: bool = True,
        scrape_threads: bool = True,
        filter_sender: Optional[str] = None,
        bot_name: str = 'Athena',
        bot_names: Optional[List[str]] = None,
        workspace_domain: str = 'mgtinsurance',
        agent_id: str = 'athena',
        slack_token: Optional[str] = None,
        drop_if_first_is_user: bool = False,
        drop_if_all_ai: bool = False,
        max_concurrent: int = 2,
        exclude_senders: Optional[List[str]] = None,
        drop_message_regexes: Optional[List[str]] = None,
        strip_citation_block: bool = False,
        member_id_to_display_name: Optional[Dict[str, str]] = None,
        human_mention_token: str = '@human',
        oldest_ts: Optional[float] = None,
        latest_ts: Optional[float] = None,
        window_days: Optional[float] = None,
        window_hours: Optional[float] = None,
        window_minutes: Optional[float] = None,
    ) -> None:
        """Initialize exporter settings and Slack service client."""
        if not channel_ids:
            raise ValueError(
                'channel_ids must be a non-empty list of Slack channel IDs'
            )

        self.channel_ids = channel_ids
        self.limit = limit
        self.reverse = reverse
        self.scrape_threads = scrape_threads
        self.filter_sender = filter_sender
        self.bot_name = bot_name
        self.bot_names = bot_names or [bot_name]
        self.workspace_domain = workspace_domain
        self.agent_id = agent_id
        self.slack_token = slack_token
        self.drop_if_first_is_user = drop_if_first_is_user
        self.drop_if_all_ai = drop_if_all_ai
        self.max_concurrent = max_concurrent
        self.exclude_senders = set(exclude_senders or [])
        self.strip_citation_block = strip_citation_block
        self.member_id_to_display_name = {
            str(member_id): str(display_name)
            for member_id, display_name in (member_id_to_display_name or {}).items()
        }
        self.human_mention_token = human_mention_token
        self._drop_message_regexes = [
            re.compile(pattern) for pattern in (drop_message_regexes or [])
        ]
        self._citation_line_re = re.compile(r'^\s*\[\d+\]\s*')
        self._slack_member_mention_re = re.compile(r'<@([A-Z0-9]+)(?:\|[^>]+)?>')
        self.default_oldest_ts, self.default_latest_ts = self._resolve_time_window(
            oldest_ts=oldest_ts,
            latest_ts=latest_ts,
            window_days=window_days,
            window_hours=window_hours,
            window_minutes=window_minutes,
        )

        self._slack = SlackService()
        self._semaphore_runner = SemaphoreExecutor(max_concurrent=max_concurrent)

    def _resolve_time_window(
        self,
        *,
        oldest_ts: Optional[float],
        latest_ts: Optional[float],
        window_days: Optional[float],
        window_hours: Optional[float],
        window_minutes: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Resolve explicit timestamps and relative window settings."""
        if oldest_ts is not None and latest_ts is not None and oldest_ts > latest_ts:
            raise ValueError('oldest_ts must be <= latest_ts')

        window_fields = {
            'window_days': window_days,
            'window_hours': window_hours,
            'window_minutes': window_minutes,
        }
        provided_windows = {
            name: value for name, value in window_fields.items() if value is not None
        }

        for name, value in provided_windows.items():
            if value is not None and value <= 0:
                raise ValueError(f'{name} must be > 0 when provided')

        if len(provided_windows) > 1:
            raise ValueError(
                'Provide only one of window_days, window_hours, or window_minutes'
            )

        resolved_oldest_ts = oldest_ts
        resolved_latest_ts = latest_ts

        if provided_windows and oldest_ts is None:
            now_ts = time.time()
            window_name, window_value = next(iter(provided_windows.items()))
            multiplier_seconds = {
                'window_days': 24 * 60 * 60,
                'window_hours': 60 * 60,
                'window_minutes': 60,
            }[window_name]
            resolved_oldest_ts = now_ts - (window_value * multiplier_seconds)
            if resolved_latest_ts is None:
                resolved_latest_ts = now_ts

        if (
            resolved_oldest_ts is not None
            and resolved_latest_ts is not None
            and resolved_oldest_ts > resolved_latest_ts
        ):
            raise ValueError('Resolved oldest_ts must be <= latest_ts')

        return resolved_oldest_ts, resolved_latest_ts

    def _permalink(self, channel_id: str, ts: Optional[str]) -> Optional[str]:
        """Build a Slack permalink for a channel message timestamp."""
        if not (self.workspace_domain and channel_id and ts):
            return None
        return f'https://{self.workspace_domain}.slack.com/archives/{channel_id}/p{ts.replace(".", "")}'

    def _to_axion_message(
        self, msg: Mapping[str, Any], text_override: Optional[str] = None
    ):
        """Convert a simplified Slack message into an Axion message."""
        sender = msg.get('sender') or 'Unknown'
        text = text_override
        if text is None:
            text = msg.get('content') or msg.get('text') or ''
            text = self._normalize_message_text(text)
        is_assistant = sender in self.bot_names
        return AIMessage(content=text) if is_assistant else HumanMessage(content=text)

    def _is_excluded_sender(self, sender: Optional[str]) -> bool:
        """Return True when sender is configured to be excluded."""
        if not sender:
            return False
        return sender in self.exclude_senders

    def _should_drop_message(self, text: str) -> bool:
        """Return True when message content matches any drop regex."""
        if not text or not self._drop_message_regexes:
            return False
        return any(pattern.search(text) for pattern in self._drop_message_regexes)

    def _strip_citation_block(self, text: str) -> str:
        """Strip trailing citation blocks like '[1] ...' from message text."""
        if not text:
            return text

        lines = text.splitlines()
        last_non_empty = len(lines) - 1
        while last_non_empty >= 0 and not lines[last_non_empty].strip():
            last_non_empty -= 1

        if last_non_empty < 0:
            return text

        footer_line = lines[last_non_empty]
        footer_start = last_non_empty
        if footer_line.lstrip().startswith('_Generated by'):
            footer_start -= 1
            while footer_start >= 0 and not lines[footer_start].strip():
                footer_start -= 1

        if footer_start < 0 or not self._citation_line_re.match(lines[footer_start]):
            return text

        block_start = footer_start
        while block_start >= 0 and (
            not lines[block_start].strip()
            or self._citation_line_re.match(lines[block_start])
        ):
            block_start -= 1

        kept = lines[: block_start + 1]
        if footer_line.lstrip().startswith('_Generated by'):
            kept.extend(lines[footer_start + 1 : last_non_empty + 1])

        cleaned = '\n'.join(kept).rstrip()
        return cleaned

    def _normalize_message_text(self, text: str) -> str:
        """Apply optional message cleanup transforms."""
        cleaned = text or ''
        if self.strip_citation_block:
            cleaned = self._strip_citation_block(cleaned)
        cleaned = self._normalize_member_mentions(cleaned)
        return cleaned.strip()

    def _normalize_member_mentions(self, text: str) -> str:
        """Normalize Slack member-id mentions for clearer human-vs-bot context."""
        if not text:
            return text

        def _replace(match: re.Match[str]) -> str:
            member_id = match.group(1)
            display_name = self.member_id_to_display_name.get(member_id)
            if display_name:
                return f'@{display_name}'
            return self.human_mention_token

        return self._slack_member_mention_re.sub(_replace, text)

    def _first_turn_is_user(self, messages: List[Any]) -> bool:
        """Return True when the first turn is from a human."""
        return bool(messages) and isinstance(messages[0], HumanMessage)

    def _all_turns_are_ai(self, messages: List[Any]) -> bool:
        """Return True when every turn is from the assistant."""
        return bool(messages) and all(isinstance(m, AIMessage) for m in messages)

    async def _fetch_channel_root_messages(
        self,
        channel_id: str,
        oldest_ts: Optional[float] = None,
        latest_ts: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch root (non-thread) messages for a channel."""
        response = await self._slack.get_channel_history(
            channel_id,
            limit=limit if limit is not None else self.limit,
            agent_id=self.agent_id,
            override_token=self.slack_token,
            oldest_ts=oldest_ts,
            latest_ts=latest_ts,
        )
        if not response.get('success'):
            raise RuntimeError(
                response.get('error')
                or f'Failed to fetch messages for channel {channel_id}'
            )
        return response.get('messages', []) or []

    async def build_thread_index(
        self,
        channel_ids: Optional[List[str]] = None,
        *,
        oldest_ts: Optional[float] = None,
        latest_ts: Optional[float] = None,
        per_channel_limit: Optional[int] = None,
    ) -> Dict[tuple[str, str], Dict[str, Any]]:
        """
        Build a map of (channel_id, thread_ts) -> simplified/root messages.

        This bulk helper reuses the same Slack scraping pipeline without requiring
        DatasetItem construction.
        """
        channel_ids_to_use = channel_ids or self.channel_ids
        effective_oldest_ts = self.default_oldest_ts if oldest_ts is None else oldest_ts
        effective_latest_ts = self.default_latest_ts if latest_ts is None else latest_ts

        async def _fetch_channel(
            channel_id: str,
        ) -> tuple[str, List[Dict[str, Any]], Optional[str]]:
            try:
                messages = await self._fetch_channel_root_messages(
                    channel_id,
                    oldest_ts=effective_oldest_ts,
                    latest_ts=effective_latest_ts,
                    limit=per_channel_limit,
                )
                return channel_id, messages, None
            except Exception as e:
                logger.warning(
                    'Failed bulk Slack channel fetch for %s: %s', channel_id, e
                )
                return channel_id, [], str(e)

        tasks = [
            self._semaphore_runner.run(_fetch_channel, channel_id)
            for channel_id in channel_ids_to_use
        ]
        channel_results = await asyncio.gather(*tasks)

        index: Dict[tuple[str, str], Dict[str, Any]] = {}
        for channel_id, raw_messages, error in channel_results:
            if error:
                continue
            for raw in raw_messages:
                thread_ts = raw.get('thread_ts') or raw.get('ts')
                if not thread_ts:
                    continue
                key = (channel_id, str(thread_ts))
                simplified = SlackScraper.simplify_message(raw)
                simplified['messageUrl'] = self._permalink(channel_id, raw.get('ts'))
                data = index.setdefault(
                    key,
                    {
                        'messages_raw': [],
                        'messages_simplified': [],
                        'error': None,
                    },
                )
                data['messages_raw'].append(raw)
                data['messages_simplified'].append(simplified)
        return index

    async def _fetch_thread_replies(
        self, channel_id: str, thread_ts: str
    ) -> List[Dict[str, Any]]:
        """Fetch thread replies for a root message."""
        response = await self._slack.get_thread_replies(
            channel_id,
            thread_ts,
            agent_id=self.agent_id,
            override_token=self.slack_token,
        )
        if not response.get('success'):
            return []
        return response.get('messages', []) or []

    async def _scrape_channel(
        self,
        channel_id: str,
        *,
        oldest_ts: Optional[float] = None,
        latest_ts: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[DatasetItem]:
        """Scrape a single channel and return DatasetItems."""
        raw_roots = await self._fetch_channel_root_messages(
            channel_id,
            oldest_ts=oldest_ts,
            latest_ts=latest_ts,
            limit=limit,
        )
        roots = [SlackScraper.simplify_message(m) for m in raw_roots]

        # Enrich roots with id + permalink
        for raw, root in zip(raw_roots, roots):
            ts = raw.get('ts')
            if ts:
                root['id'] = ts.replace('.', '')
            root['messageUrl'] = self._permalink(channel_id, ts)

        # Thread scraping
        if self.scrape_threads:
            for raw, root in zip(raw_roots, roots):
                thread_ts = raw.get('thread_ts')
                reply_count = raw.get('reply_count') or 0

                if thread_ts and reply_count:
                    thread_raw = await self._fetch_thread_replies(channel_id, thread_ts)
                    thread_simple = [
                        SlackScraper.simplify_message(m) for m in thread_raw
                    ]

                    for reply_raw, reply_simple in zip(thread_raw, thread_simple):
                        reply_simple['messageUrl'] = self._permalink(
                            channel_id, reply_raw.get('ts')
                        )

                    root['threadReplies'] = thread_simple
                else:
                    root['threadReplies'] = []
        else:
            for root in roots:
                root['threadReplies'] = []

        # Optional filter by ROOT sender
        if self.filter_sender:
            roots = [r for r in roots if r.get('sender') == self.filter_sender]

        if self.reverse:
            roots.reverse()

        items: List[DatasetItem] = []

        for root in roots:
            if self._is_excluded_sender(root.get('sender')):
                continue
            root_text = self._normalize_message_text(
                root.get('content') or root.get('text') or ''
            )
            if not root_text:
                continue
            if self._should_drop_message(root_text):
                continue
            convo_messages = [self._to_axion_message(root, text_override=root_text)]

            root_ts = root.get('ts')
            for reply in root.get('threadReplies', []) or []:
                if self._is_excluded_sender(reply.get('sender')):
                    continue
                reply_text = self._normalize_message_text(
                    reply.get('content') or reply.get('text') or ''
                )
                if not reply_text:
                    continue
                if self._should_drop_message(reply_text):
                    continue
                # Avoid duplicate parent message if Slack returns it in replies
                if reply.get('ts') and reply.get('ts') == root_ts:
                    continue
                convo_messages.append(
                    self._to_axion_message(reply, text_override=reply_text)
                )

            # Optional: Drop conversations based on first turn or all AI turns
            if self.drop_if_first_is_user and self._first_turn_is_user(convo_messages):
                continue
            if self.drop_if_all_ai and self._all_turns_are_ai(convo_messages):
                continue

            conversation = MultiTurnConversation(messages=convo_messages)

            items.append(
                DatasetItem(
                    id=root.get('id'),
                    conversation=conversation,
                    metadata=json.dumps(
                        {
                            'message_url': root.get('messageUrl'),
                            'channel_id': channel_id,
                            'thread_ts': root.get('thread_ts') or root_ts,
                        }
                    ),
                    additional_input={
                        'message_url': root.get('messageUrl'),
                        'channel_id': channel_id,
                        'thread_ts': root.get('thread_ts') or root_ts,
                        'reply_count': len(root.get('threadReplies', []) or []),
                        'sender': root.get('sender'),
                    },
                )
            )

        return items

    async def build_dataset_item_index(
        self,
        channel_ids: Optional[List[str]] = None,
        *,
        oldest_ts: Optional[float] = None,
        latest_ts: Optional[float] = None,
        per_channel_limit: Optional[int] = None,
    ) -> Dict[tuple[str, str], DatasetItem]:
        """
        Build a lookup map from (channel_id, thread_ts) to DatasetItem.

        This allows other components to reuse SlackExporter's full conversation
        extraction semantics and then join by thread key.
        """
        channel_ids_to_use = channel_ids or self.channel_ids
        effective_oldest_ts = self.default_oldest_ts if oldest_ts is None else oldest_ts
        effective_latest_ts = self.default_latest_ts if latest_ts is None else latest_ts
        tasks = [
            self._semaphore_runner.run(
                self._scrape_channel,
                channel_id,
                oldest_ts=effective_oldest_ts,
                latest_ts=effective_latest_ts,
                limit=per_channel_limit,
            )
            for channel_id in channel_ids_to_use
        ]
        results = await asyncio.gather(*tasks)

        index: Dict[tuple[str, str], DatasetItem] = {}
        for channel_items in results:
            for item in channel_items:
                additional_input = item.additional_input or {}
                channel_id = additional_input.get('channel_id')
                thread_ts = additional_input.get('thread_ts')
                if not channel_id or not thread_ts:
                    continue
                key = (str(channel_id), str(thread_ts))
                # Keep first item deterministically if duplicates appear.
                index.setdefault(key, item)
        return index

    async def execute(
        self, channel_ids: Optional[List[str]] = None
    ) -> List[DatasetItem]:
        """Run export across provided or configured channels."""
        channel_ids_to_use = channel_ids or self.channel_ids
        tasks = [
            self._semaphore_runner.run(
                self._scrape_channel,
                channel_id,
                oldest_ts=self.default_oldest_ts,
                latest_ts=self.default_latest_ts,
            )
            for channel_id in channel_ids_to_use
        ]
        results = await asyncio.gather(*tasks)
        all_items: List[DatasetItem] = []
        for channel_items in results:
            all_items.extend(channel_items)
        return all_items
