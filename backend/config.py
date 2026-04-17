from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Falls back to sensible defaults for local development.

    In production (Render), override any value via the Environment Variables
    section in the Render dashboard — no code changes needed.

    Pydantic-settings automatically reads from:
    1. Environment variables (highest priority)
    2. .env file (if present)
    3. Default values defined here (lowest priority)

    Usage anywhere in the app:
        from config import get_settings
        settings = get_settings()
    """

    # ── Model artifact paths ──────────────────────────────────────────────
    # Override in production if using external storage (S3, GCS)
    # instead of committed pickle files.
    model_path:   str = os.path.join(
        os.path.dirname(__file__), "model", "model_xgb.pkl"
    )
    medians_path: str = os.path.join(
        os.path.dirname(__file__), "model", "train_medians.json"
    )
    meta_path:    str = os.path.join(
        os.path.dirname(__file__), "model", "model_metadata.json"
    )

    # ── Rate limiting ─────────────────────────────────────────────────────
    # Format: "N/period" where period is second, minute, hour, day.
    # Default is generous for a demo. Tighten in production.
    # Example: RATE_LIMIT=10/minute
    rate_limit: str = "30/minute"

    # ── Logging ───────────────────────────────────────────────────────────
    # Set to DEBUG locally for verbose output, INFO in production.
    # Example: LOG_LEVEL=DEBUG
    log_level: str = "INFO"

    # ── CORS ──────────────────────────────────────────────────────────────
    # Currently open ("*") for demo purposes.
    # In production restrict to trusted origins.
    # Example: CORS_ORIGINS=["https://myapp.com"]
    cors_origins: list[str] = ["*"]

    class Config:
        # Reads from .env file if present — silently ignored if not found.
        # Never commit .env to git — use .env.example as a template instead.
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    lru_cache ensures the .env file and environment variables are only
    read once per process — not on every request that calls get_settings().
    This makes config lookups essentially free after the first call.
    """
    return Settings()