import csv
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from shared.database.neon import NeonConnection

logger = logging.getLogger(__name__)


class ScoredItemsStore(ABC):
    """Abstract base class for scored items storage.

    Tracks which items have been evaluated by source to avoid re-processing.
    Source keys are in the format "source_type:source_name" (e.g., "langfuse:athena").
    """

    @staticmethod
    def _parse_source_key(source_key: str) -> tuple[str, str]:
        """Parse source_key into (source_type, source_name)."""
        parts = source_key.split(':', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid source_key format: {source_key}. Expected 'type:name'")
        return parts[0], parts[1]

    @abstractmethod
    def get_scored_item_ids(self, source_key: str) -> set[str]:
        """Get item IDs already scored for this source."""
        ...

    @abstractmethod
    def record_scored_items(self, source_key: str, item_ids: list[str]) -> None:
        """Record items as scored."""
        ...

    @abstractmethod
    def clear(self, source_key: str | None = None) -> int:
        """Clear scored items, optionally filtered by source."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        """Get statistics about scored items."""
        ...


class CSVScoredItemsStore(ScoredItemsStore):
    """CSV-based storage for scored item IDs.

    Uses a hybrid approach for performance:
    - Appends (record_scored_items): Raw CSV writer with 'a' mode - O(n) for n new items
    - Reads/filters (get_scored_item_ids, get_stats): Pandas - convenient filtering
    - Rewrites (clear): Pandas - only when deleting rows

    CSV columns:
    - dataset_id: The unique item identifier
    - source_type: Data source type (e.g., "langfuse", "slack")
    - source_name: Data source name (e.g., "athena")
    - created_at: ISO timestamp of when the item was scored

    Args:
        file_path: Path to CSV file (default: "data/scored_items.csv")
    """

    COLUMNS = ['dataset_id', 'source_type', 'source_name', 'created_at']

    def __init__(self, file_path: str | Path = 'data/scored_items.csv'):
        self.file_path = Path(file_path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the CSV file with headers if it doesn't exist."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.COLUMNS)
            logger.info(f'Created scored items file: {self.file_path}')

    def _read_df(self) -> pd.DataFrame:
        """Read CSV file into pandas DataFrame."""
        return pd.read_csv(self.file_path, dtype=str)

    def get_scored_item_ids(self, source_key: str) -> set[str]:
        """
        Get item IDs already scored for this source.

        Args:
            source_key: Source identifier (e.g., "langfuse:athena")

        Returns:
            Set of item IDs that have been scored
        """
        try:
            source_type, source_name = self._parse_source_key(source_key)
            df = self._read_df()
            mask = (df['source_type'] == source_type) & (df['source_name'] == source_name)
            scored = set(df.loc[mask, 'dataset_id'].tolist())
            logger.debug(f'Found {len(scored)} scored items for {source_key}')
            return scored
        except FileNotFoundError:
            logger.warning(f'Scored items file not found: {self.file_path}')
            return set()
        except Exception as e:
            logger.error(f'Error reading scored items: {e}')
            return set()

    def record_scored_items(self, source_key: str, item_ids: list[str]) -> None:
        """
        Record items as scored using raw CSV append.

        Args:
            source_key: Source identifier (e.g., "langfuse:athena")
            item_ids: List of item IDs to record
        """
        if not item_ids:
            return

        try:
            source_type, source_name = self._parse_source_key(source_key)
            now = datetime.now(timezone.utc).isoformat()
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for item_id in item_ids:
                    writer.writerow([item_id, source_type, source_name, now])
            logger.info(f'Recorded {len(item_ids)} scored items for {source_key}')
        except Exception as e:
            logger.error(f'Error recording scored items: {e}')

    def clear(self, source_key: str | None = None) -> int:
        """
        Clear scored items, optionally filtered by source.

        Uses pandas for rewrite when filtering is needed.

        Args:
            source_key: Optional source key to filter by. If None, clears all items.

        Returns:
            Number of items removed
        """
        if not self.file_path.exists():
            return 0

        try:
            df = self._read_df()

            if source_key is None:
                count = len(df)
                with open(self.file_path, 'w', newline='') as f:
                    csv.writer(f).writerow(self.COLUMNS)
                logger.info(f'Cleared all {count} scored items')
                return count

            source_type, source_name = self._parse_source_key(source_key)
            mask = (df['source_type'] == source_type) & (df['source_name'] == source_name)
            count = mask.sum()
            df[~mask].to_csv(self.file_path, index=False)
            logger.info(f'Cleared {count} scored items for {source_key}')
            return count
        except Exception as e:
            logger.error(f'Error clearing scored items: {e}')
            return 0

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about scored items by source.

        Returns:
            Dict mapping "source_type:source_name" to count of items
        """
        try:
            df = self._read_df()
            if df.empty:
                return {}
            df['source_key'] = df['source_type'] + ':' + df['source_name']
            return df.groupby('source_key').size().to_dict()
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f'Error getting stats: {e}')
            return {}


class DBScoredItemsStore(ScoredItemsStore):
    """
    Database-backed storage using evaluation_dataset table.

    Reads scored items from the evaluation_dataset table to avoid re-processing.
    Recording is a no-op since items are recorded via OnlineMonitor._push_to_db().

    Args:
        db: Optional NeonConnection instance to reuse
        connection_string: Optional connection string (creates new connection if db not provided)
    """

    def __init__(
        self,
        db: NeonConnection | None = None,
        connection_string: str | None = None,
    ):
        self._db = db
        self._connection_string = connection_string

    def _get_db(self) -> NeonConnection:
        """Get or create database connection."""
        if self._db is None:
            self._db = NeonConnection(connection_string=self._connection_string)
        return self._db

    def get_scored_item_ids(self, source_key: str) -> set[str]:
        """
        Get item IDs already scored for this source from database.

        Args:
            source_key: Source identifier (e.g., "langfuse:athena")

        Returns:
            Set of item IDs that have been scored
        """
        try:
            source_type, source_name = self._parse_source_key(source_key)
            query = """
                SELECT dataset_id FROM evaluation_dataset
                WHERE source_type = %s AND source_name = %s
            """
            rows = self._get_db().fetch_all(query, (source_type, source_name))
            scored = {row['dataset_id'] for row in rows}
            logger.debug(f'Found {len(scored)} scored items for {source_key} in database')
            return scored
        except Exception as e:
            logger.error(f'Error reading scored items from database: {e}')
            return set()

    def record_scored_items(self, source_key: str, item_ids: list[str]) -> None:
        """
        No-op: Items are recorded via OnlineMonitor._push_to_db().
        """
        # Items are recorded to database via EvaluationUploader in OnlineMonitor._push_to_db()
        pass

    def clear(self, source_key: str | None = None) -> int:
        """Clear scored items from database."""
        raise NotImplementedError(
            "Use database admin tools to clear data from evaluation_dataset table"
        )

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about scored items by source from database.

        Returns:
            Dict mapping "source_type:source_name" to count of items
        """
        try:
            query = """
                SELECT source_type || ':' || source_name as source_key, COUNT(*) as count
                FROM evaluation_dataset
                GROUP BY source_type, source_name
            """
            rows = self._get_db().fetch_all(query, ())
            return {row['source_key']: row['count'] for row in rows}
        except Exception as e:
            logger.error(f'Error getting stats from database: {e}')
            return {}
