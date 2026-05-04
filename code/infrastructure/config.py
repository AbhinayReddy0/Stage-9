"""
config.py — single source of truth for all Stage 9 environment configuration.

Loads .env from the project root at import time. No external packages needed —
falls back to a built-in parser if python-dotenv is not installed.
Every module must read configuration from the constants exported here;
never call os.environ directly.
"""
import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _load_env_file(path: Path) -> None:
    """Parse a .env file and populate os.environ for keys not already set."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)
except ImportError:
    _load_env_file(_ENV_FILE)

# ── Database ──────────────────────────────────────────────────────────────────
DB_HOST            = os.environ.get("DB_HOST", "localhost")
DB_PORT            = int(os.environ.get("DB_PORT", "5432"))
DB_NAME            = os.environ.get("DB_NAME", "dev")
DB_USER            = os.environ.get("DB_USER", "postgres")
DB_PASSWORD        = os.environ.get("DB_PASSWORD", "")
DB_SSLMODE         = os.environ.get("DB_SSLMODE", "disable")
DB_CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "10"))

DB_DSN = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode={DB_SSLMODE}&connect_timeout={DB_CONNECT_TIMEOUT}"
)

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL       = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_POOL_SIZE = int(os.environ.get("REDIS_POOL_SIZE", "20"))

# ── Application ───────────────────────────────────────────────────────────────
PLANNING_THREADS      = int(os.environ.get("STAGE9_PLANNING_THREADS", "16"))
ALLOW_FORCE_RELEASE   = os.environ.get("STAGE9_ALLOW_FORCE_RELEASE", "").lower() == "true"
PROJECT_ROOT          = os.environ.get("STAGE9_PROJECT_ROOT", "/mnt/project")
RUN_INTEGRATION_TESTS = os.environ.get("RUN_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes")
