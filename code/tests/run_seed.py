"""
Run tenant onboarding seed from the command line.

Usage:
    python run_seed.py <tenant_id> <tenant_maturity> [--override key=value ...]

Examples:
    python run_seed.py 550e8400-e29b-41d4-a716-446655440000 new
    python run_seed.py 550e8400-e29b-41d4-a716-446655440000 established --override service_level_target=0.95
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from decimal import Decimal
from pathlib import Path

# Add M:\ (grandparent of this script) so that `import stage_9` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# "db files" has a space so it can't be imported with normal syntax — load by path.
_mig_path = Path(__file__).parent / "db files" / "run_migrations_local.py"
_spec = importlib.util.spec_from_file_location("run_migrations_local", _mig_path)
_mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mig)
get_conn = _mig.get_conn

from infrastructure.seed import seed_tenant_params

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5.5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_overrides(pairs: list[str]) -> dict[str, Decimal]:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got: {pair!r}")
        key, _, val = pair.partition("=")
        overrides[key.strip()] = Decimal(val.strip())
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed tenant_learning_params for a new tenant.")
    parser.add_argument("tenant_id", help="UUID of the tenant")
    parser.add_argument("tenant_maturity", choices=["new", "developing", "established"])
    parser.add_argument("--override", metavar="key=value", nargs="*", default=[])
    args = parser.parse_args()

    overrides = parse_overrides(args.override)

    conn = get_conn()
    try:
        inserted = seed_tenant_params(
            tenant_id=args.tenant_id,
            tenant_maturity=args.tenant_maturity,
            overrides_dict=overrides or None,
            conn=conn,
        )
        conn.commit()
        logger.info("Done — inserted %d rows.", inserted)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
