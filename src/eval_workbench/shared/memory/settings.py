from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ZepSettings(BaseSettings):
    """Configuration for Zep Cloud graph memory backend."""

    model_config = SettingsConfigDict(
        env_prefix='ZEP_',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    api_key: str | None = Field(
        default=None,
        description='Zep Cloud API key.',
    )
    base_url: str | None = Field(
        default=None,
        description='Zep Cloud base URL (only needed for self-hosted).',
    )
    admin_email: str = Field(
        default='system@eval-workbench.local',
        description='Admin email for Zep user creation.',
    )
    user_id_template: str = Field(
        default='{agent_name}_global_rules',
        description='Template for generating per-agent Zep user IDs.',
    )

    def resolve_user_id(self, agent_name: str) -> str:
        """Fill the user_id_template with the given agent name."""
        return self.user_id_template.format(agent_name=agent_name)


@lru_cache(maxsize=1)
def get_zep_settings() -> ZepSettings:
    """Return a cached ZepSettings instance."""
    return ZepSettings()
