from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_path(value: str | Path | None) -> Path:
    if value is None:
        return Path.cwd()
    return Path(value)


@lru_cache(maxsize=None)
def find_repo_root(start_path: str | Path | None = None) -> Path:
    start = _normalize_path(start_path).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def infer_implementation_name(from_path: str | Path | None = None) -> str | None:
    path = _normalize_path(from_path).resolve()
    parts = path.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "implementations" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def resolve_env_files(
    *,
    from_path: str | Path | None = None,
    implementation_name: str | None = None,
) -> list[str]:
    repo_root = find_repo_root(from_path)
    impl_name = implementation_name or infer_implementation_name(from_path)

    env_files: list[Path] = [repo_root / ".env"]
    if impl_name:
        env_files.append(repo_root / "implementations" / impl_name / ".env")

    return [str(path) for path in env_files if path.exists()]


def build_settings_config(
    *,
    from_path: str | Path | None = None,
    implementation_name: str | None = None,
    env_prefix: str | None = None,
) -> SettingsConfigDict:
    env_files = resolve_env_files(
        from_path=from_path,
        implementation_name=implementation_name,
    )

    config = SettingsConfigDict(
        env_file=env_files,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    if env_prefix:
        config["env_prefix"] = env_prefix
    return config


class RepoSettingsBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langfuse_host: str | None = Field(
        default=None,
        description="Langfuse host URL shared across implementations.",
    )
