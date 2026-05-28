"""Centralized configuration via pydantic-settings.

All previously-hardcoded values live here. Override at deploy time with
environment variables prefixed PIXELSCRIPT_:

  export PIXELSCRIPT_GCP_PROJECT=other-project
  export PIXELSCRIPT_CORS_ALLOWED_ORIGINS="http://localhost:3000,https://prod.example.com"
  export PIXELSCRIPT_RATE_LIMIT_PER_MINUTE=20
  export PIXELSCRIPT_ESTIMATED_COST_PER_IMAGE_USD=0.12

Defaults are chosen for local dev. Production must override at minimum:
  - cors_allowed_origins (NEVER ship `*` to prod)
  - rate_limit_* (tune to expected concurrency)
  - estimated_cost_per_image_usd (set from Vertex billing once measured)
"""
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIXELSCRIPT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # The repo already has a stale .env with API_HOST/UPLOAD_DIR/etc. left
        # from initial scaffolding. Those aren't ours; ignore unknown fields
        # so they don't crash startup.
        extra="ignore",
    )

    # GCP / Vertex AI
    gcp_project: str = "pixelscript-prod"
    model_id: str = "gemini-3-pro-image-preview"
    vertex_location: str = "global"
    generation_timeout_s: int = 300  # 5 min — covers 9.5-min outlier with kill margin

    # Output storage
    outputs_dir: str = "outputs"
    output_cleanup_age_hours: int = 24

    # CORS — comma-separated string; access via cors_origins_list
    cors_allowed_origins: str = "http://localhost:3000"

    # Rate limiting (slowapi). Layered minute + hour caps.
    rate_limit_per_minute: int = 5
    rate_limit_per_hour: int = 30

    # Cost tracking — rough estimate for Gemini 3 Pro Image at 2K.
    # Tune from actual Vertex billing once a calibration window is available.
    estimated_cost_per_image_usd: float = 0.10

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def rate_limit_str(self) -> str:
        """slowapi layered-limit format."""
        return f"{self.rate_limit_per_minute}/minute;{self.rate_limit_per_hour}/hour"


settings = Settings()
