"""
Stage 9 cross-agent signal bus package.

Public surface — import from here or from signal_bus (backward-compat shim):

    from signals import SignalEmitter, SignalConsumer, Signal
"""
from signals._base import (  # noqa: F401
    Signal,
    SignalEmitFailed,
    SIGNAL_TYPE_FORECAST_ACCURACY,
    SIGNAL_TYPE_FORECAST_RISK,
    SIGNAL_TYPE_CROSS_SKU_LEARNING,
    SIGNAL_TYPE_MODEL_HEALTH,
    TTL_FORECAST_ACCURACY_DAYS,
    TTL_FORECAST_RISK_DAYS,
    TTL_CROSS_SKU_LEARNING_DAYS,
    TTL_MODEL_HEALTH_DAYS,
    FROM_AGENT_STAGE_9,
    TO_AGENT_STAGE_8,
    TO_AGENT_STAGE_9,
    TO_AGENT_STAGE_10,
    TO_AGENT_BROADCAST,
    DEFAULT_PEEK_LIMIT,
    DEFAULT_CONSUME_LIMIT,
)
from signals.emitter import SignalEmitter  # noqa: F401
from signals.consumer import SignalConsumer  # noqa: F401

__all__ = [
    "SignalEmitter",
    "SignalConsumer",
    "Signal",
    "SignalEmitFailed",
    "SIGNAL_TYPE_FORECAST_ACCURACY",
    "SIGNAL_TYPE_FORECAST_RISK",
    "SIGNAL_TYPE_CROSS_SKU_LEARNING",
    "SIGNAL_TYPE_MODEL_HEALTH",
    "TTL_FORECAST_ACCURACY_DAYS",
    "TTL_FORECAST_RISK_DAYS",
    "TTL_CROSS_SKU_LEARNING_DAYS",
    "TTL_MODEL_HEALTH_DAYS",
    "FROM_AGENT_STAGE_9",
    "TO_AGENT_STAGE_8",
    "TO_AGENT_STAGE_9",
    "TO_AGENT_STAGE_10",
    "TO_AGENT_BROADCAST",
    "DEFAULT_PEEK_LIMIT",
    "DEFAULT_CONSUME_LIMIT",
]
