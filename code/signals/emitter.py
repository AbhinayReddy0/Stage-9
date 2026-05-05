"""
SignalEmitter — direct-write half of the Stage 9 cross-agent signal bus.

NOT THREAD-SAFE. Each thread / worker must construct its own SignalEmitter
with its own psycopg2 connection. In dual_pool's ThreadPoolExecutor,
instantiate per worker — never share a single emitter across the 16 thread slots.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

from infrastructure.db_utils import DBConnection
from infrastructure.tenant_params import TenantParams
from signals._base import (
    SignalEmitFailed,
    SIGNAL_TYPE_FORECAST_ACCURACY, SIGNAL_TYPE_FORECAST_RISK,
    SIGNAL_TYPE_CROSS_SKU_LEARNING, SIGNAL_TYPE_MODEL_HEALTH,
    TTL_FORECAST_ACCURACY_DAYS, TTL_FORECAST_RISK_DAYS,
    TTL_CROSS_SKU_LEARNING_DAYS, TTL_MODEL_HEALTH_DAYS,
    FROM_AGENT_STAGE_9, TO_AGENT_STAGE_8, TO_AGENT_STAGE_9,
    TO_AGENT_STAGE_10, TO_AGENT_BROADCAST,
    SIGNAL_MAX_RETRIES, SIGNAL_RETRY_DELAY_S,
    _INSERT_SIGNAL_SQL, _DELETE_PRIOR_MODEL_HEALTH_SQL,
    _clamp_confidence, _wrap_jsonb, _maybe_warn_shared_conn,
)

logger = logging.getLogger(__name__)


class SignalEmitter:
    """
    Owns one DEDICATED connection (same isolation pattern as
    pattern_feedback's `pf_conn`) and emits signals through it.

    By default each emit immediately INSERTs and commits, so Stage 8 /
    Stage 10 can read it the moment we return. For high-volume signal
    types (forecast_accuracy / forecast_risk during 5M-SKU runs), pass
    `flush_every=N` to accumulate INSERTs in one open transaction and
    commit only every N emits — and call flush() at end-of-batch.

    Use as a context manager to guarantee a final flush:

        with SignalEmitter(conn, ..., flush_every=100) as e:
            for sku in skus:
                e.emit_forecast_accuracy(sku.id, ...)
        # flush() ran on __exit__

    `log_failure_fn(tenant_id, run_id, sku_id, reason)` records each
    retry-exhausted emit to stage9_sku_execution_log. Optional but
    recommended for production (Principle 3 audit trail).

    `raise_on_failure=True` makes retry exhaustion raise SignalEmitFailed
    instead of returning False.
    """

    def __init__(
        self,
        conn: DBConnection,
        *,
        tenant_id: str,
        run_id: str,
        max_retries: int = SIGNAL_MAX_RETRIES,
        retry_delay_seconds: float = SIGNAL_RETRY_DELAY_S,
        flush_every: int = 1,
        log_failure_fn: Optional[Callable[[str, str, str, str], None]] = None,
        raise_on_failure: bool = False,
    ) -> None:
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1, got {flush_every}")
        _maybe_warn_shared_conn(conn, label="SignalEmitter")
        self._conn = conn
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._flush_every = flush_every
        self._log_failure_fn = log_failure_fn
        self._raise_on_failure = raise_on_failure
        self._pending_count = 0

    # ---- context manager ------------------------------------------------

    def __enter__(self) -> "SignalEmitter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.flush()
        except Exception:
            logger.exception("flush failed during SignalEmitter __exit__")

    @property
    def pending_count(self) -> int:
        """Number of INSERTs in the open txn awaiting commit."""
        return self._pending_count

    def flush(self) -> bool:
        """Commit all pending signals. No-op if nothing pending."""
        if self._pending_count == 0:
            return True
        return self._commit_and_reset(signal_type="<flush>", sku_id=None)

    # ---- public emit methods --------------------------------------------

    def emit_forecast_accuracy(
        self,
        sku_id: str,
        *,
        pattern_label: str,
        model_used: str,
        mape: float,
        bias: Optional[float],
        quality: str,
        hint_matched: bool,
    ) -> bool:
        """
        Stage 9 → Stage 8. Fired immediately after the 9.4 backtest
        produces metrics. `quality` is the same 'good'/'acceptable'/
        'poor'/'proxy' string written to pattern_feedback.
        """
        payload = {
            "pattern_label": pattern_label,
            "model_used": model_used,
            "mape": float(mape) if mape is not None else None,
            "bias": float(bias) if bias is not None else None,
            "quality": quality,
            "hint_matched": bool(hint_matched),
        }
        signal_confidence = (
            (1.0 - float(mape)) if mape is not None else None
        )
        return self._emit(
            signal_type=SIGNAL_TYPE_FORECAST_ACCURACY,
            to_agent=TO_AGENT_STAGE_8,
            sku_id=sku_id,
            payload=payload,
            confidence=signal_confidence,
            ttl_days=TTL_FORECAST_ACCURACY_DAYS,
        )

    def emit_forecast_risk(
        self,
        sku_id: str,
        *,
        confidence: float,
        confidence_tier: str,
        risk_level: str,
        exception_flags: list[str],
        mape_30d: Optional[float],
        forecast_30d_selected: Optional[float],
        selected_quantile: float,
    ) -> bool:
        """
        Stage 9 → Stage 10. Fired after 9.5 computes confidence. Does NOT
        filter by risk_level — the caller decides whether to skip 'low'.
        """
        payload = {
            "confidence": float(confidence),
            "confidence_tier": confidence_tier,
            "risk_level": risk_level,
            "exception_flags": list(exception_flags),
            "mape_30d": float(mape_30d) if mape_30d is not None else None,
            "forecast_30d_selected": (
                float(forecast_30d_selected)
                if forecast_30d_selected is not None else None
            ),
            "selected_quantile": float(selected_quantile),
        }
        return self._emit(
            signal_type=SIGNAL_TYPE_FORECAST_RISK,
            to_agent=TO_AGENT_STAGE_10,
            sku_id=sku_id,
            payload=payload,
            confidence=float(confidence),
            ttl_days=TTL_FORECAST_RISK_DAYS,
        )

    def emit_cross_sku_learning(
        self,
        sku_id: str,
        *,
        pattern_label: str,
        vendor: Optional[str],
        product_type: Optional[str],
        best_hp: dict,
        best_features: list[str],
        mape: float,
        params: TenantParams,
    ) -> bool:
        """
        Stage 9 → Stage 9. Fired in the LEARNING state for converged SKUs.
        ONLY emitted when `mape < warm_start_max_mape` — never share noisy
        SKUs as warm-start candidates.
        """
        threshold = params.get("warm_start_max_mape")
        if mape is None or float(mape) >= threshold:
            logger.debug(
                "cross_sku_learning skipped sku_id=%s mape=%s threshold=%s",
                sku_id, mape, threshold,
            )
            return False

        payload = {
            "pattern_label": pattern_label,
            "vendor": vendor,
            "product_type": product_type,
            "best_hp": dict(best_hp) if best_hp else {},
            "best_features": list(best_features) if best_features else [],
            "mape": float(mape),
        }
        return self._emit(
            signal_type=SIGNAL_TYPE_CROSS_SKU_LEARNING,
            to_agent=TO_AGENT_STAGE_9,
            sku_id=sku_id,
            payload=payload,
            confidence=1.0,  # only emit converged SKUs
            ttl_days=TTL_CROSS_SKU_LEARNING_DAYS,
        )

    def emit_model_health(
        self,
        *,
        model_health: dict,
        recommendations: list[Any],
    ) -> bool:
        """
        Stage 9 → All. Fired in the REPORTING state, ONCE per run.
        Tenant-level (sku_id is NULL).

        Idempotent: deletes any prior un-consumed model_health row for the
        same (tenant_id, run_id) before inserting the new one.

        ATOMICITY: DELETE + INSERT happen in ONE retry-protected transaction.
        If either fails, BOTH are rolled back together.
        """
        payload = {
            "model_health": dict(model_health),
            "recommendations": list(recommendations),
        }
        delete_args = (
            self._tenant_id, self._run_id, SIGNAL_TYPE_MODEL_HEALTH,
        )
        insert_args = (
            str(uuid.uuid4()),
            self._tenant_id,
            FROM_AGENT_STAGE_9,
            TO_AGENT_BROADCAST,
            SIGNAL_TYPE_MODEL_HEALTH,
            None,                # sku_id — tenant-level
            self._run_id,
            _wrap_jsonb(payload),
            None,                # confidence — N/A for tenant-level
            int(TTL_MODEL_HEALTH_DAYS),
        )

        last_err: Optional[BaseException] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._conn.cursor() as cur:
                    cur.execute(_DELETE_PRIOR_MODEL_HEALTH_SQL, delete_args)
                    cur.execute(_INSERT_SIGNAL_SQL, insert_args)
                self._pending_count += 1
                if self._pending_count >= self._flush_every:
                    return self._commit_and_reset(
                        signal_type=SIGNAL_TYPE_MODEL_HEALTH, sku_id=None,
                    )
                return True
            except Exception as e:
                last_err = e
                logger.warning(
                    "model_health emit attempt %d/%d failed err=%s",
                    attempt, self._max_retries, e,
                )
                self._safe_rollback("model_health DELETE+INSERT retry")
                self._pending_count = 0
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        logger.error(
            "model_health emit FAILED after %d attempts err=%s",
            self._max_retries, last_err,
        )
        self._notify_failure(
            SIGNAL_TYPE_MODEL_HEALTH, None, f"emit_failed:{last_err}",
        )
        if self._raise_on_failure:
            raise SignalEmitFailed(
                attempts=self._max_retries, last_error=last_err,
                signal_type=SIGNAL_TYPE_MODEL_HEALTH, sku_id=None,
            )
        return False

    # ---- internal write loop --------------------------------------------

    def _emit(
        self,
        *,
        signal_type: str,
        to_agent: Optional[str],
        sku_id: Optional[str],
        payload: dict,
        confidence: Optional[float],
        ttl_days: int,
    ) -> bool:
        args = (
            str(uuid.uuid4()),
            self._tenant_id,
            FROM_AGENT_STAGE_9,
            to_agent,
            signal_type,
            sku_id,
            self._run_id,
            _wrap_jsonb(payload),
            _clamp_confidence(confidence),
            int(ttl_days),
        )

        if not self._try_insert_with_retry(args, signal_type, sku_id):
            return False

        self._pending_count += 1
        if self._pending_count >= self._flush_every:
            return self._commit_and_reset(
                signal_type=signal_type, sku_id=sku_id,
            )
        return True

    def _try_insert_with_retry(
        self, args: tuple, signal_type: str, sku_id: Optional[str],
    ) -> bool:
        last_err: Optional[BaseException] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._conn.cursor() as cur:
                    cur.execute(_INSERT_SIGNAL_SQL, args)
                return True
            except Exception as e:
                last_err = e
                logger.warning(
                    "signal emit attempt %d/%d failed type=%s sku=%s err=%s",
                    attempt, self._max_retries, signal_type, sku_id, e,
                )
                self._safe_rollback("signal emit retry")
                self._pending_count = 0
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        logger.error(
            "signal emit FAILED after %d attempts type=%s sku=%s err=%s",
            self._max_retries, signal_type, sku_id, last_err,
        )
        self._notify_failure(signal_type, sku_id, f"insert_failed:{last_err}")
        if self._raise_on_failure:
            raise SignalEmitFailed(
                attempts=self._max_retries, last_error=last_err,
                signal_type=signal_type, sku_id=sku_id,
            )
        return False

    def _commit_and_reset(self, *, signal_type: str, sku_id: Optional[str]) -> bool:
        try:
            self._conn.commit()
            self._pending_count = 0
            return True
        except Exception as e:
            lost = self._pending_count
            logger.error(
                "signal commit failed lost=%d type=%s sku=%s err=%s",
                lost, signal_type, sku_id, e,
            )
            self._safe_rollback("commit failure")
            self._pending_count = 0
            self._notify_failure(
                signal_type, sku_id, f"commit_failed_lost_{lost}",
            )
            if self._raise_on_failure:
                raise SignalEmitFailed(
                    attempts=1, last_error=e,
                    signal_type=signal_type, sku_id=sku_id,
                )
            return False

    def _safe_rollback(self, ctx: str) -> None:
        try:
            self._conn.rollback()
        except Exception:
            logger.debug("rollback failed during %s", ctx, exc_info=True)

    def _notify_failure(
        self, signal_type: str, sku_id: Optional[str], reason: str,
    ) -> None:
        if self._log_failure_fn is None:
            return
        try:
            self._log_failure_fn(
                self._tenant_id, self._run_id,
                sku_id or "<tenant>",
                f"signal:{signal_type}:{reason}",
            )
        except Exception:
            logger.exception(
                "log_failure_fn raised — failure not recorded type=%s sku=%s",
                signal_type, sku_id,
            )
