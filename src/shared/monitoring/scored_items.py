"""
File-based storage for tracking scored items (deduplication).

Tracks which items have been processed by source and monitor combination
to avoid re-evaluating the same items.

Example:
    from shared.monitoring.scored_items import ScoredItemsStore

    store = ScoredItemsStore("data/scored_items.csv")

    # Check what's already scored
    scored = store.get_scored_item_ids("langfuse:athena", "athena_monitor")

    # After processing, record new items
    store.record_scored_items("langfuse:athena", "athena_monitor", ["id1", "id2"])
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ScoredItemsStore:
    """Simple file-based storage for scored item IDs.

    Uses a CSV file to track which items have been evaluated by which
    monitor. Each row contains:
    - item_id: The unique item identifier
    - source_key: Data source identifier (e.g., "langfuse:athena")
    - monitor_key: Monitor name (e.g., "athena_recommendation_monitor")
    - scored_at: ISO timestamp of when the item was scored

    Args:
        file_path: Path to CSV file (default: "data/scored_items.csv")
    """

    HEADERS = ['item_id', 'source_key', 'monitor_key', 'scored_at']

    def __init__(self, file_path: str | Path = 'data/scored_items.csv'):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the CSV file with headers if it doesn't exist."""
        if not self.file_path.exists():
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
            logger.info(f'Created scored items file: {self.file_path}')

    def get_scored_item_ids(self, source_key: str, monitor_key: str) -> set[str]:
        """Get item IDs already scored for this source + monitor combo.

        Args:
            source_key: Data source identifier (e.g., "langfuse:athena")
            monitor_key: Monitor name

        Returns:
            Set of item IDs that have been scored
        """
        scored = set()
        try:
            with open(self.file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (
                        row['source_key'] == source_key
                        and row['monitor_key'] == monitor_key
                    ):
                        scored.add(row['item_id'])
        except FileNotFoundError:
            logger.warning(f'Scored items file not found: {self.file_path}')
        except Exception as e:
            logger.error(f'Error reading scored items: {e}')

        logger.debug(f'Found {len(scored)} scored items for {source_key}/{monitor_key}')
        return scored

    def record_scored_items(
        self,
        source_key: str,
        monitor_key: str,
        item_ids: list[str],
    ) -> None:
        """Record items as scored.

        Args:
            source_key: Data source identifier
            monitor_key: Monitor name
            item_ids: List of item IDs to record
        """
        if not item_ids:
            return

        now = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for item_id in item_ids:
                    writer.writerow([item_id, source_key, monitor_key, now])
            logger.info(
                f'Recorded {len(item_ids)} scored items for {source_key}/{monitor_key}'
            )
        except Exception as e:
            logger.error(f'Error recording scored items: {e}')

    def clear(
        self, source_key: str | None = None, monitor_key: str | None = None
    ) -> int:
        """Clear scored items, optionally filtered by source/monitor.

        Args:
            source_key: Filter by source key (optional)
            monitor_key: Filter by monitor key (optional)

        Returns:
            Number of items removed
        """
        if not self.file_path.exists():
            return 0

        rows_to_keep = []
        rows_removed = 0

        with open(self.file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                should_remove = True

                if source_key is not None and row['source_key'] != source_key:
                    should_remove = False
                if monitor_key is not None and row['monitor_key'] != monitor_key:
                    should_remove = False

                if should_remove:
                    rows_removed += 1
                else:
                    rows_to_keep.append(row)

        # Rewrite file with remaining rows
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writeheader()
            writer.writerows(rows_to_keep)

        logger.info(f'Cleared {rows_removed} scored items')
        return rows_removed

    def get_stats(self) -> dict[str, int]:
        """Get statistics about scored items by source/monitor.

        Returns:
            Dict mapping "source_key:monitor_key" to count of items
        """
        stats: dict[str, int] = {}
        try:
            with open(self.file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f'{row["source_key"]}:{row["monitor_key"]}'
                    stats[key] = stats.get(key, 0) + 1
        except FileNotFoundError:
            pass

        return stats
