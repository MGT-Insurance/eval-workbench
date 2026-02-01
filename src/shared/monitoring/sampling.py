import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from axion.dataset import DatasetItem


class SamplingStrategy(ABC):
    """Abstract base for item sampling strategies."""

    @abstractmethod
    def sample(self, items: list[DatasetItem]) -> list[DatasetItem]:
        """Sample items from the list.

        Args:
            items: List of items to sample from

        Returns:
            Sampled subset of items
        """
        pass


class AllSampling(SamplingStrategy):
    """No sampling - return all items (default behavior)."""

    def sample(self, items: list[DatasetItem]) -> list[DatasetItem]:
        return items


class RandomSampling(SamplingStrategy):
    """Randomly sample N items.

    Args:
        n: Number of items to sample
        seed: Optional seed for reproducible sampling
    """

    def __init__(self, n: int, seed: int | None = None):
        self.n = n
        self.seed = seed

    def sample(self, items: list[DatasetItem]) -> list[DatasetItem]:
        if len(items) <= self.n:
            return items
        rng = random.Random(self.seed)
        return rng.sample(items, self.n)


class MostRecentSampling(SamplingStrategy):
    """Select N most recent items.

    Assumes items are ordered by time (most recent first).

    Args:
        n: Number of items to select
    """

    def __init__(self, n: int):
        self.n = n

    def sample(self, items: list[DatasetItem]) -> list[DatasetItem]:
        return items[: self.n]


class OldestSampling(SamplingStrategy):
    """Select N oldest items.

    Assumes items are ordered by time (most recent first).

    Args:
        n: Number of items to select
    """

    def __init__(self, n: int):
        self.n = n

    def sample(self, items: list[DatasetItem]) -> list[DatasetItem]:
        return items[-self.n :] if len(items) > self.n else items


class SamplingStrategyType(Enum):
    """Enum of available sampling strategies with factory method."""

    ALL = 'all'
    RANDOM = 'random'
    MOST_RECENT = 'most_recent'
    OLDEST = 'oldest'

    def create(self, **kwargs: Any) -> SamplingStrategy:
        """Create a SamplingStrategy instance with the given parameters.

        Args:
            **kwargs: Strategy-specific parameters
                - n: Number of items to sample (for RANDOM, MOST_RECENT, OLDEST)
                - seed: Random seed (for RANDOM only)

        Returns:
            Configured SamplingStrategy instance
        """
        n = kwargs.get('n', 10)
        seed = kwargs.get('seed')

        match self:
            case SamplingStrategyType.ALL:
                return AllSampling()
            case SamplingStrategyType.RANDOM:
                return RandomSampling(n=n, seed=seed)
            case SamplingStrategyType.MOST_RECENT:
                return MostRecentSampling(n=n)
            case SamplingStrategyType.OLDEST:
                return OldestSampling(n=n)
