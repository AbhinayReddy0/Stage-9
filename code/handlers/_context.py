"""
handlers/_context.py — Shared run context registry.

Each Stage 9 run creates one RunContext stored here by run_id.
Handlers retrieve it by run_id to share state without altering
the state_machine.run() call signatures.

Lifecycle: created in preloading_handler, removed in reporting_handler.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from backtesting.backtesting import BacktestContext
    from forecasting.forecasting import ForecastContext
    from infrastructure.batch_writer import BatchWriter
    from infrastructure.tenant_params import TenantParams
    from learning.self_assessment import SKUResult
    from pipeline.dual_pool import SkuPipelineInput
    from pipeline.preloader import PreloadedData
    from signals.consumer import SignalConsumer

__all__ = ["RunContext", "store", "fetch", "remove"]


@dataclass
class RunContext:
    tenant_id: str
    run_id: str
    run_start_time: float = field(default_factory=time.time)
    execution_mode: str = "full"

    # Set by preloading_handler
    preloaded: PreloadedData | None = None
    batch_writer: BatchWriter | None = None
    signal_consumer: SignalConsumer | None = None
    sku_ids: list[str] = field(default_factory=list)

    # Set by perceiving_handler
    params: TenantParams | None = None
    pattern_signals: list[dict] = field(default_factory=list)  # PATTERN_CONFIDENCE signals from Stage 8

    # Set by planning_handler
    calibrated_cache: dict[tuple[str, str], int] = field(default_factory=dict)
    calibration_gaps: dict[tuple[str, str], float] = field(default_factory=dict)
    pipeline_inputs: list[SkuPipelineInput] = field(default_factory=list)
    cache_sku_ids: list[str] = field(default_factory=list)
    # Demand history loaded once by planning_handler and reused by acting_handler
    # so we don't pay the 730-day × 5M-SKU bulk SELECT twice per run.
    demand_data: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Set by acting_handler
    sku_results: list[SKUResult] = field(default_factory=list)
    backtest_contexts: dict[str, BacktestContext] = field(default_factory=dict)
    forecast_contexts: dict[str, ForecastContext] = field(default_factory=dict)
    pattern_feedback_failures_count: int = 0


_REGISTRY: dict[str, RunContext] = {}


def store(ctx: RunContext) -> None:
    _REGISTRY[ctx.run_id] = ctx


def fetch(run_id: str) -> RunContext:
    ctx = _REGISTRY.get(run_id)
    if ctx is None:
        raise KeyError(
            f"No RunContext for run_id={run_id!r} — "
            "preloading_handler may have failed or not yet run."
        )
    return ctx


def remove(run_id: str) -> None:
    _REGISTRY.pop(run_id, None)
