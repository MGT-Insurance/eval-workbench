import asyncio
import json
import re
from typing import Any, Dict, List, Mapping, Optional

from axion._core.asyncio import SemaphoreExecutor
from axion._core.schema import AIMessage, HumanMessage
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation

from eval_workbench.shared.slack.service import SlackScraper, SlackService


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
        drop_if_first_is_user: Drop if first turn is HumanMessage.
        drop_if_all_ai: Drop if all turns are AIMessage.
        max_concurrent: Max concurrent channel scrapes.
        exclude_senders: Optional list of sender names to omit from conversations.
        drop_message_regexes: Optional list of regex patterns to drop messages.
        strip_citation_block: Remove trailing citation blocks like "[1] ..." lines.

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
        drop_if_first_is_user: bool = False,
        drop_if_all_ai: bool = False,
        max_concurrent: int = 2,
        exclude_senders: Optional[List[str]] = None,
        drop_message_regexes: Optional[List[str]] = None,
        strip_citation_block: bool = False,
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
        self.drop_if_first_is_user = drop_if_first_is_user
        self.drop_if_all_ai = drop_if_all_ai
        self.max_concurrent = max_concurrent
        self.exclude_senders = set(exclude_senders or [])
        self.strip_citation_block = strip_citation_block
        self._drop_message_regexes = [
            re.compile(pattern) for pattern in (drop_message_regexes or [])
        ]
        self._citation_line_re = re.compile(r'^\s*\[\d+\]\s*')

        self._slack = SlackService()
        self._semaphore_runner = SemaphoreExecutor(max_concurrent=max_concurrent)

    def _permalink(self, channel_id: str, ts: Optional[str]) -> Optional[str]:
        """Build a Slack permalink for a channel message timestamp."""
        if not (self.workspace_domain and channel_id and ts):
            return None
        return f'https://{self.workspace_domain}.slack.com/archives/{channel_id}/p{ts.replace(".", "")}'

    def _to_axion_message(self, msg: Mapping[str, Any]):
        """Convert a simplified Slack message into an Axion message."""
        sender = msg.get('sender') or 'Unknown'
        text = msg.get('content') or msg.get('text') or ''
        if self.strip_citation_block:
            text = self._strip_citation_block(text)
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

    def _first_turn_is_user(self, messages: List[Any]) -> bool:
        """Return True when the first turn is from a human."""
        return bool(messages) and isinstance(messages[0], HumanMessage)

    def _all_turns_are_ai(self, messages: List[Any]) -> bool:
        """Return True when every turn is from the assistant."""
        return bool(messages) and all(isinstance(m, AIMessage) for m in messages)

    async def _fetch_channel_root_messages(
        self, channel_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch root (non-thread) messages for a channel."""
        response = await self._slack.get_channel_history(
            channel_id,
            limit=self.limit,
            agent_id=self.agent_id,
        )
        if not response.get('success'):
            raise RuntimeError(
                response.get('error')
                or f'Failed to fetch messages for channel {channel_id}'
            )
        return response.get('messages', []) or []

    async def _fetch_thread_replies(
        self, channel_id: str, thread_ts: str
    ) -> List[Dict[str, Any]]:
        """Fetch thread replies for a root message."""
        response = await self._slack.get_thread_replies(
            channel_id,
            thread_ts,
            agent_id=self.agent_id,
        )
        if not response.get('success'):
            return []
        return response.get('messages', []) or []

    async def _scrape_channel(self, channel_id: str) -> List[DatasetItem]:
        """Scrape a single channel and return DatasetItems."""
        raw_roots = await self._fetch_channel_root_messages(channel_id)
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
            root_text = root.get('content') or root.get('text') or ''
            if self._should_drop_message(root_text):
                continue
            convo_messages = [self._to_axion_message(root)]

            root_ts = root.get('ts')
            for reply in root.get('threadReplies', []) or []:
                if self._is_excluded_sender(reply.get('sender')):
                    continue
                reply_text = reply.get('content') or reply.get('text') or ''
                if self._should_drop_message(reply_text):
                    continue
                # Avoid duplicate parent message if Slack returns it in replies
                if reply.get('ts') and reply.get('ts') == root_ts:
                    continue
                convo_messages.append(self._to_axion_message(reply))

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

    async def execute(
        self, channel_ids: Optional[List[str]] = None
    ) -> List[DatasetItem]:
        """Run export across provided or configured channels."""
        channel_ids_to_use = channel_ids or self.channel_ids
        tasks = [
            self._semaphore_runner.run(self._scrape_channel, channel_id)
            for channel_id in channel_ids_to_use
        ]
        results = await asyncio.gather(*tasks)
        all_items: List[DatasetItem] = []
        for channel_items in results:
            all_items.extend(channel_items)
        return all_items
