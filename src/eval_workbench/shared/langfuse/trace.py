import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class ModelUsageUnit(Enum):
    TOKENS = 'TOKENS'
    CHARACTERS = 'CHARACTERS'
    MILLISECONDS = 'MILLISECONDS'
    SECONDS = 'SECONDS'
    IMAGES = 'IMAGES'
    REQUESTS = 'REQUESTS'


class ObservationLevel(Enum):
    DEFAULT = 'DEFAULT'
    DEBUG = 'DEBUG'
    WARNING = 'WARNING'
    ERROR = 'ERROR'


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    total: int = 0
    unit: Any = ModelUsageUnit.TOKENS


def _normalize_key(key: str) -> str:
    """Helper to normalize keys for fuzzy matching (snake_case -> camelCase support)."""
    return key.lower().replace('_', '')


class SmartAccess:
    """
    A base class that allows dictionary values to be accessed via dot notation.
    It recursively wraps returned dictionaries, lists, and objects.
    """

    def __getattr__(self, key: str) -> Any:
        # 1. Try to find the key in the object's internal dictionary (exact match)
        try:
            val = self._lookup(key)
            return self._wrap(val)
        except (KeyError, AttributeError):
            pass

        # 2. Case/Separator-insensitive fallback (e.g., .product_type -> productType)
        val = self._lookup_insensitive(key)
        if val is not None:
            return self._wrap(val)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __getitem__(self, key: Any) -> Any:
        """Keep support for trace['key'] syntax just in case."""
        return self._wrap(self._lookup(key))

    def _lookup(self, key: str) -> Any:
        """Subclasses must implement how to fetch raw data."""
        raise NotImplementedError

    def _lookup_insensitive(self, key: str) -> Any:
        """Optional hook for fuzzy matching."""
        return None

    def _wrap(self, val: Any) -> Any:
        """Recursively wrap results to ensure dot-notation connectivity."""
        if isinstance(val, dict):
            return SmartDict(val)
        if isinstance(val, list):
            return [self._wrap(x) for x in val]

        # If it's a generic object (has attributes) but not already a SmartAccess wrapper,
        # wrap it in SmartObject so we can traverse its attributes smartly.
        if hasattr(val, '__dict__') and not isinstance(val, SmartAccess):
            return SmartObject(val)

        return val


class SmartDict(SmartAccess):
    """Wraps a standard dictionary to allow dot access with fuzzy matching."""

    def __init__(self, data: Dict):
        self._data = data

    def _lookup(self, key: str) -> Any:
        return self._data[key]

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k, v in self._data.items():
            if _normalize_key(k) == target:
                return v
        return None

    def __repr__(self):
        return f'<SmartDict keys={list(self._data.keys())}>'

    def to_dict(self) -> Dict:
        """Return the underlying raw dictionary."""
        return self._data


class SmartObject(SmartAccess):
    """Wraps a generic Python object to ensure its attributes return Smart wrappers."""

    def __init__(self, obj: Any):
        self._obj = obj

    def _lookup(self, key: str) -> Any:
        return getattr(self._obj, key)

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        # Check standard attributes
        for k in dir(self._obj):
            if _normalize_key(k) == target:
                return getattr(self._obj, k)
        return None

    def __repr__(self):
        return repr(self._obj)


class TraceView(SmartAccess):
    """
    Wraps the root trace object.
    Holds attributes like id, latency, environment, and the list of observations.
    """

    def __init__(self, **kwargs):
        self._data = kwargs
        # Ensure observations is at least an empty list
        if 'observations' not in self._data:
            self._data['observations'] = []

    def _lookup(self, key: str) -> Any:
        return self._data[key]

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k, v in self._data.items():
            if _normalize_key(k) == target:
                return v
        return None

    def __repr__(self):
        return f"TraceView(id='{self._data.get('id', 'N/A')}', name='{self._data.get('name', 'N/A')}')"

    # Allow property access for the wrapper to pick up
    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            # Fallback to SmartAccess logic logic
            return super().__getattr__(item)


class ObservationsView(SmartAccess):
    """
    Wraps a single observation (Span/Generation).
    Uses internal storage to ensure attribute access triggers __getattr__.
    """

    def __init__(self, **kwargs):
        self._data = kwargs

    def _lookup(self, key: str) -> Any:
        return self._data[key]

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)
        for k, v in self._data.items():
            if _normalize_key(k) == target:
                return v
        return None

    def __repr__(self):
        name = self._data.get('name', 'unnamed')
        type_ = self._data.get('type', 'unknown')
        return f"ObservationsView(name='{name}', type='{type_}')"


def create_extraction_pattern(start_text: str, end_pattern: str) -> str:
    r"""Helper for Regex: escaped(Start) -> (Content) -> End"""
    return rf'{re.escape(start_text)}:\s*(.*?)\s*(?:{end_pattern})'


class PromptPatternsBase:
    """Base registry for regex extraction patterns (empty by default)."""

    @classmethod
    def get_for(cls, step_name: str) -> Dict[str, str]:
        # Step names may include separators (e.g. "location-extraction") that are
        # not valid in Python method names. We try a few normalized variants.
        raw = step_name.lower()
        candidates = [
            raw,
            raw.replace('-', '_'),
            re.sub(r'[^a-z0-9_]+', '_', raw).strip('_'),
            re.sub(r'[^a-z0-9]+', '', raw),
        ]
        for candidate in candidates:
            method_name = f'_patterns_{candidate}'
            if hasattr(cls, method_name):
                return getattr(cls, method_name)()
        return {}


def _resolve_prompt_patterns(
    patterns: PromptPatternsBase | type[PromptPatternsBase] | None,
) -> PromptPatternsBase:
    if patterns is None:
        return PromptPatternsBase()
    return patterns() if isinstance(patterns, type) else patterns


class TraceStep(SmartAccess):
    """
    Represents a specific named step (e.g., 'recommendation').
    SmartAccess enables: step.generation, step.variables.caseAssessment
    """

    def __init__(
        self,
        name: str,
        observations: List[ObservationsView],
        prompt_patterns: PromptPatternsBase,
    ):
        self.name = name
        self.observations = observations
        self.prompt_patterns = prompt_patterns

    def _lookup(self, key: str) -> Any:
        # Handle special property 'variables' for prompt extraction
        if key == 'variables':
            return self._extract_variables()

        # Handle aliases like 'generation' or 'context'
        target_type = self._resolve_type_alias(key)

        # Find the observation
        for obs in self.observations:
            if getattr(obs, 'type', '').upper() == target_type:
                return obs

        raise KeyError(
            f"No observation of type '{target_type}' found in step '{self.name}'."
        )

    def _resolve_type_alias(self, key: str) -> str:
        key_upper = key.upper()
        mapping = {
            'CONTEXT': 'SPAN',
            'SPAN': 'SPAN',
            'GENERATION': 'GENERATION',
            'EVENT': 'EVENT',
        }
        return mapping.get(key_upper, key_upper)

    def _extract_variables(self) -> Dict[str, str]:
        """Lazy extraction of prompt variables."""
        try:
            # We reuse our own lookup to find the generation object
            gen = self._lookup('GENERATION')
            raw_text = getattr(gen, 'input', '')
            prompt_text = self._normalize_prompt_text(raw_text)
            if not prompt_text:
                return {}

            patterns = self.prompt_patterns.get_for(self.name)
            extracted = {}
            for k, pattern in patterns.items():
                match = re.search(pattern, prompt_text, re.DOTALL)
                if match:
                    extracted[k] = match.group(1).strip()
            return extracted
        except (KeyError, AttributeError):
            return {}

    @staticmethod
    def _normalize_prompt_text(raw_text: Any) -> str:
        if isinstance(raw_text, str):
            return raw_text
        if isinstance(raw_text, dict):
            for key in ('content', 'prompt', 'text'):
                value = raw_text.get(key)
                if isinstance(value, str):
                    return value
            return ''
        if isinstance(raw_text, list):
            parts: list[str] = []
            for item in raw_text:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get('content')
                    if isinstance(value, str):
                        parts.append(value)
            return '\n'.join(parts)
        return ''

    def __repr__(self):
        types = [getattr(o, 'type', '') for o in self.observations]
        return f"<TraceStep name='{self.name}' types={types}>"


class Trace(SmartAccess):
    """
    Main wrapper.
    SmartAccess enables: trace.recommendation
    Also provides access to trace-level attributes (id, latency, etc.)
    """

    def __init__(
        self,
        trace_data: Any,
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    ):
        """
        Args:
            trace_data: Can be a list of observations OR the root trace object containing .observations
        """
        # 1. Detect input type
        if hasattr(trace_data, 'observations') and isinstance(
            trace_data.observations, list
        ):
            self._trace_obj = trace_data
            self.observations = trace_data.observations
        elif isinstance(trace_data, dict) and isinstance(
            trace_data.get('observations'), list
        ):
            self._trace_obj = TraceView(**trace_data)
            self.observations = trace_data.get('observations', [])
        elif isinstance(trace_data, list):
            self._trace_obj = None
            self.observations = trace_data
        else:
            # Fallback
            self._trace_obj = trace_data
            self.observations = []

        # Normalize observation dicts into ObservationsView for attribute access.
        self.observations = [
            ObservationsView(**obs) if isinstance(obs, dict) else obs
            for obs in self.observations
        ]

        self._grouped: dict[str, list[ObservationsView]] = {}
        self._prompt_patterns = _resolve_prompt_patterns(prompt_patterns)
        self._group_observations()

    def _group_observations(self):
        for obs in self.observations:
            name = getattr(obs, 'name', 'unnamed')
            if name not in self._grouped:
                self._grouped[name] = []
            self._grouped[name].append(obs)

    def _lookup(self, key: str) -> Any:
        # 1. Check if key matches a grouped step name (e.g., 'recommendation')
        if key in self._grouped:
            return TraceStep(key, self._grouped[key], self._prompt_patterns)

        # 2. If not a step, check if it's an attribute of the root trace object (e.g., 'id', 'latency')
        if self._trace_obj:
            if hasattr(self._trace_obj, key):
                return getattr(self._trace_obj, key)

            # Special handling if trace_obj is a dict-like view/wrapper
            if isinstance(self._trace_obj, (TraceView, dict, ObservationsView)):
                try:
                    # If it's a wrapper, allow it to use its own lookup
                    if hasattr(self._trace_obj, '_lookup'):
                        return self._trace_obj._lookup(key)
                    return self._trace_obj[key]
                except (KeyError, AttributeError):
                    pass

        raise KeyError(f"Attribute '{key}' not found in Trace steps or properties.")

    def _lookup_insensitive(self, key: str) -> Any:
        target = _normalize_key(key)

        # 1. Check step names (e.g., allow trace.Recommendation)
        for k, v in self._grouped.items():
            if _normalize_key(k) == target:
                return TraceStep(k, v, self._prompt_patterns)

        # 2. Check root object attributes (e.g. trace.created_at -> createdAt)
        if self._trace_obj:
            # If it's a dict or wrapper, check keys
            if hasattr(self._trace_obj, 'keys'):  # Dict-like
                try:
                    # If it's a wrapper class
                    if hasattr(self._trace_obj, '_lookup_insensitive'):
                        return self._trace_obj._lookup_insensitive(key)
                    # Standard dict loop
                    for k in self._trace_obj.keys():
                        if _normalize_key(k) == target:
                            return self._trace_obj[k]
                except Exception:
                    pass

            # If it's a general object, check dir()
            for k in dir(self._trace_obj):
                if _normalize_key(k) == target:
                    return getattr(self._trace_obj, k)

        return None

    def __repr__(self):
        base = f'<Trace steps={list(self._grouped.keys())}'
        if self._trace_obj:
            tid = getattr(self._trace_obj, 'id', 'unknown')
            base += f" id='{tid}'"
        base += '>'
        return base


class TraceCollection:
    """
    Wraps a list of trace data items (from JSON or API response).
    Each item in the list is converted to a Trace object.
    """

    def __init__(
        self,
        data: List[Any],
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    ):
        self._traces = [Trace(item, prompt_patterns=prompt_patterns) for item in data]

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: TraceCollection._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [TraceCollection._to_jsonable(v) for v in value]
        if is_dataclass(value) and not isinstance(value, type):
            return TraceCollection._to_jsonable(asdict(value))
        if hasattr(value, 'model_dump'):
            return TraceCollection._to_jsonable(value.model_dump())
        if hasattr(value, 'dict'):
            try:
                return TraceCollection._to_jsonable(value.dict())
            except TypeError:
                pass
        if hasattr(value, 'to_dict'):
            return TraceCollection._to_jsonable(value.to_dict())
        if hasattr(value, '__dict__'):
            return TraceCollection._to_jsonable(vars(value))
        return str(value)

    def to_list(self) -> List[Any]:
        return [t._trace_obj for t in self._traces]

    def save_json(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._to_jsonable(t._trace_obj) for t in self._traces]
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        prompt_patterns: PromptPatternsBase | type[PromptPatternsBase] | None = None,
    ) -> 'TraceCollection':
        source = Path(path).expanduser()
        data = json.loads(source.read_text())
        if not isinstance(data, list):
            raise ValueError('TraceCollection.load_json expects a JSON list.')
        return cls(data, prompt_patterns=prompt_patterns)

    def __getitem__(self, index: int) -> Trace:
        return self._traces[index]

    def __iter__(self):
        return iter(self._traces)

    def __len__(self):
        return len(self._traces)

    def filter_by(self, **kwargs) -> 'TraceCollection':
        """
        Simple filter helper.
        """
        filtered = []
        for t in self._traces:
            match = True
            for k, v in kwargs.items():
                if not hasattr(t, k) or getattr(t, k) != v:
                    match = False
                    break
            if match:
                filtered.append(t)

        return TraceCollection(
            [t._trace_obj for t in filtered],
            prompt_patterns=self._traces[0]._prompt_patterns if self._traces else None,
        )

    def __repr__(self):
        return f'<TraceCollection count={len(self._traces)}>'
