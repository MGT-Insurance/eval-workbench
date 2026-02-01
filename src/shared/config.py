from __future__ import annotations

import operator
import os
import re
import typing as t
from copy import deepcopy
from functools import reduce
from pathlib import Path

import yaml


class ConfigurationError(Exception):
    """Raised when there's an error loading configuration."""

    pass


ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

# Global config dictionary
config: dict = {}


def _interpolate_env_vars(value: t.Any) -> t.Any:
    """Interpolate ${VAR} and ${VAR:-default} in config values."""
    if isinstance(value, str):

        def replacer(match: re.Match) -> str:
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.environ.get(var_name.strip(), default)
            else:
                var_name = var_expr.strip()
                return os.environ.get(
                    var_name, match.group(0)
                )  # Keep original if not found

        return ENV_VAR_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def get(
    key_str: str, error: bool = False, default: t.Any = None, cfg: dict | None = None
) -> t.Any:
    """
    Get item from config. Use '.' for nested access.

    Parameters
    ----------
    key_str : str
        Config key to pull. If multi-level key for nested config, use '.'
    error : bool
        Raise KeyError if not found.
    default : Any
        What to return if not found.
    cfg : dict | None
        Config dict to use. If None, uses global config.

    Examples
    --------
    >>> from shared import config
    >>> config.get('name')  # Outer level
    >>> config.get('trace_loader.limit')  # Nested
    >>> config.get('does.not.exist', error=True)  # raises error
    >>> config.get('does.not.exist', default={})  # returns type dict
    """
    target_config = cfg if cfg is not None else config
    multikey = key_str.split('.')
    tmp_config_copy = deepcopy(target_config)
    try:
        return reduce(operator.getitem, multikey, tmp_config_copy)
    except KeyError:
        if error:
            raise KeyError(f'Multi-key {multikey} not found in config.')
        return default


class set:
    """
    Temporarily set configuration values within a context manager.

    Parameters
    ----------
    arg : mapping or None, optional
        A mapping of configuration key-value pairs to set.
    cfg : dict, optional
        Config dict to use. If None, uses global config.

    Examples
    --------
    >>> from shared import config
    >>> with config.set({'trace_loader.limit': 10}):
    ...     pass
    """

    config: dict
    _record: list[tuple[str, tuple[str, ...], t.Any]]

    def __init__(self, arg: t.Mapping, cfg: dict | None = None):
        self.config = cfg if cfg is not None else config
        self._record = []

        for key, value in arg.items():
            self._assign(key.split('.'), value, self.config)

    def __enter__(self):
        return self.config

    def __exit__(self, type, value, traceback):
        for op, path, value in reversed(self._record):
            d = self.config
            if op == 'replace':
                for key in path[:-1]:
                    d = d.setdefault(key, {})
                d[path[-1]] = value
            else:
                for key in path[:-1]:
                    try:
                        d = d[key]
                    except KeyError:
                        break
                else:
                    d.pop(path[-1], None)

    def _assign(
        self,
        keys: t.Sequence[str],
        value: t.Any,
        d: dict,
        path: tuple[str, ...] = (),
        record: bool = True,
    ) -> None:
        """
        Assign value into a nested configuration dictionary.

        Parameters
        ----------
        keys : Sequence[str]
            The nested path of keys to assign the value.
        value : object
        d : dict
            The part of the nested dictionary into which we want to assign the
            value
        path : tuple[str], optional
            The path history up to this point.
        record : bool, optional
            Whether this operation needs to be recorded to allow for rollback.
        """
        key = keys[0]

        path = path + (key,)

        if len(keys) == 1:
            if record:
                if key in d:
                    self._record.append(('replace', path, d[key]))
                else:
                    self._record.append(('insert', path, None))
            d[key] = value
        else:
            if key not in d:
                if record:
                    self._record.append(('insert', path, None))
                d[key] = {}
                # No need to record subsequent operations after an insert
                record = False
            self._assign(keys[1:], value, d[key], path, record=record)


class load(set):
    """
    Load YAML file or dictionary into config.

    Parameters
    ----------
    config_file : os.PathLike, optional
        Path to YAML file. If it exists, its contents are loaded.
    config_dict : dict, optional
        Dictionary to use as config if no file is provided.
    cfg : dict, optional
        Config dict to load into. If None, uses global config.
    interpolate : bool, optional
        Whether to interpolate env vars. Default True.

    Examples
    --------
    >>> from shared import config
    >>> config.load('config/monitoring.yaml')

    >>> with config.load('config/monitoring.yaml'):
    ...     pass
    """

    def __init__(
        self,
        config_file: os.PathLike | None = None,
        config_dict: dict | None = None,
        cfg: dict | None = None,
        interpolate: bool = True,
    ):
        if config_file:
            path = Path(config_file)
            if not path.exists():
                raise ConfigurationError(f'Config file not found: {path}')
            with open(path) as f:
                raw_config = yaml.safe_load(f) or {}
        elif config_dict:
            raw_config = config_dict
        else:
            raw_config = {}

        if interpolate:
            raw_config = _interpolate_env_vars(raw_config)

        super().__init__(raw_config, cfg=cfg)


def load_config(
    path: str | Path, overrides: dict[str, t.Any] | None = None
) -> dict[str, t.Any]:
    """Load YAML config with env var interpolation and optional overrides.

    This function returns a new config dict rather than modifying the global config.
    For global config management, use the `load` class instead.

    Args:
        path: Path to YAML file
        overrides: Dict with dot-notation keys to override config values

    Returns:
        Configuration dictionary

    Examples:
        >>> from shared import config
        >>> # Basic load
        >>> cfg = config.load_config("config/monitoring.yaml")

        >>> # With overrides using dot notation
        >>> cfg = config.load_config(
        ...     "config/monitoring.yaml",
        ...     overrides={"trace_loader.limit": 5, "publishing.push_to_db": True}
        ... )
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f'Config file not found: {path}')

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _interpolate_env_vars(cfg)

    if overrides:
        # Apply overrides using dot notation
        for key, value in overrides.items():
            _set_nested(cfg, key, value)

    return cfg


def _set_nested(d: dict, key_str: str, value: t.Any) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = key_str.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
