from __future__ import annotations

from pathlib import Path

from pydantic import Field

from eval_workbench.shared.settings import RepoSettingsBase, build_settings_config


class AthenaSettings(RepoSettingsBase):
    model_config = build_settings_config(from_path=Path(__file__))

    langfuse_athena_public_key: str | None = Field(
        default=None,
        description='Langfuse public key for Athena traces.',
    )
    langfuse_athena_secret_key: str | None = Field(
        default=None,
        description='Langfuse secret key for Athena traces.',
    )
