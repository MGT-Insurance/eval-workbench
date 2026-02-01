from typing import Any, List, Set, Tuple


def flatten_json_lines(data: Any, parent_key: str = '', sep: str = '.') -> List[str]:
    """Recursively flatten nested data into 'path: value' lines."""
    out: List[str] = []

    def _walk(obj: Any, prefix: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f'{prefix}{sep}{k}' if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f'{prefix}[{i}]')
        else:
            out.append(f'{prefix}: {str(obj)}')

    _walk(data, parent_key)
    return out


def flatten_paths(data: Any) -> Tuple[Set[str], Set[str]]:
    """Collect lowercase leaf paths and keys for membership checks."""
    paths: Set[str] = set()
    keys: Set[str] = set()

    def _walk(obj: Any, prefix: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(k.lower())
                new_prefix = f'{prefix}.{k}' if prefix else k
                _walk(v, new_prefix)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_prefix = f'{prefix}[{i}]' if prefix else f'[{i}]'
                _walk(v, new_prefix)
        else:
            if prefix:
                paths.add(prefix.lower())

    _walk(data, '')
    return paths, keys
