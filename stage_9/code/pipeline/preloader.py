"""
preloader.py — Atheera Stage 9 Forecasting Agent
=================================================
PreloadedData: typed container for all bulk-loaded data passed into sub-stages.
Preloader: runs all 7 bulk reads plus TenantParams and signal_context loads.

The Preloader is instantiated once per run by preloading_handler. After
calling load(), the handler stores preloaded data and loader.params on the
RunContext so every subsequent handler and sub-stage can access them without
any further DB reads inside the SKU-processing loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["PreloadedData", "Preloader"]


# ---------------------------------------------------------------------------
# PreloadedData — typed container
# ---------------------------------------------------------------------------

@dataclass
class PreloadedData:
    """
    Bulk-loaded data for one tenant run.

    pattern_ctx:
        Per-SKU dict from the 3-way JOIN (pattern_history + feature_decisions
        + canonical_sku + signal_context). Keys per entry: pattern_label,
        confidence_calibrated, model_hint, obs_days, lifecycle_stage,
        composite_confidence, drift_detected, weekend_zero_ratio,
        velocity_signature, on_watchlist, feature_reliability_map,
        vendor, product_type, parent_style_id, shelf_life_days,
        planned_end_date, criticality_tier, service_level_target,
        product_lifecycle_type, seed_daily_demand.

    sku_metadata:
        Per-SKU canonical_sku columns sliced out of pattern_ctx for
        sub_stage_91 backward compatibility. Keys: criticality_tier,
        service_level_target, planned_end_date, shelf_life_days,
        product_lifecycle_type, seed_daily_demand.

    oos_ctx:
        Per-SKU OOS impact estimates. Keys: oos_pct, detection_confidence.
        Missing key = no record (factor defaults to 1.0).

    channel_splits:
        Per-SKU list of {sale_date, qty, channel} rows. Populated only
        when signal_context.pipeline_mode = 'multi_channel'.

    promo_decisions:
        Keyed by (sku_id, date_iso) tuple — O(1) lookup per day.
        Value: float multiplier. Empty dict when no promo decisions.

    portfolio_alerts:
        List of alert dicts from portfolio_intelligence_reports (last 90 days,
        filtered to 4 structural alert types).

    tenant_thresholds:
        Tenant-level confidence_floor and confidence_ceiling from Stage 8.

    signal_context:
        Tenant-level context (pipeline_mode, tenant_maturity).

    thompson_ctx:
        Per-SKU aggregate Thompson state (alpha, beta, historical_runs)
        for the exploit decision in sub_stage_91.

    thompson_state:
        Keyed by (sku_id, model_name) tuple. Value: Dict[config_hash,
        {"alpha", "beta", "config"}]. Mutated in-memory by sub_stage_93
        and bulk-flushed to DB by learning_handler.

    feature_reliability:
        Per-SKU feature reliability map {sku_id: {feature_name: float}}.
        Populated from Stage 8's feature_decisions table via pattern_ctx.

    promo_decisions:
        Per-(sku_id, date_iso) promo multipliers for sub_stage_92.

    feature_history:
        Per-SKU prior feature selections from stage9.feature_decisions_s9
        for warm-start in sub_stage_92. Keyed by sku_id; values list[str].
    """

    tenant_id: str = ""

    # Sub-Stage 9.1 inputs
    pattern_ctx: dict[str, Any] = field(default_factory=dict)
    sku_metadata: dict[str, Any] = field(default_factory=dict)
    oos_ctx: dict[str, Any] = field(default_factory=dict)
    thompson_ctx: dict[str, Any] = field(default_factory=dict)
    signal_context: dict[str, Any] = field(default_factory=dict)

    # New bulk reads
    channel_splits: dict[str, Any] = field(default_factory=dict)
    portfolio_alerts: list = field(default_factory=list)
    tenant_thresholds: dict[str, Any] = field(default_factory=dict)

    # Sub-Stage 9.2 inputs (passed as plain dict slice)
    feature_reliability: dict = field(default_factory=dict)
    promo_decisions: dict = field(default_factory=dict)
    feature_history: dict = field(default_factory=dict)

    # Sub-Stage 9.3 inputs (passed as plain dict slice)
    thompson_state: dict = field(default_factory=dict)

    # Sub-Stage 9.0 — fingerprinting and tier classification
    fingerprint_cache: dict[str, Any] = field(default_factory=dict)  # loaded from data_fingerprint_cache
    sku_tiers: dict[str, str] = field(default_factory=dict)  # sku_id → "cache"|"partial"|"full"
    new_fingerprints: dict[str, Any] = field(default_factory=dict)  # sku_id → {fingerprint} for post-run write


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

# Signal context — tenant-level pipeline_mode and tenant_maturity.
# Read before the 7 bulk reads so Read 3 can be gated on pipeline_mode.
_SQL_SIGNAL_CONTEXT = """
    SELECT pipeline_mode, tenant_maturity
    FROM stage8.signal_context
    WHERE tenant_id = %s
    LIMIT 1
"""

# Read 1 — 3-way JOIN: pattern_history + feature_decisions + canonical_sku
# + signal_context (for per-SKU on_watchlist).
# Returns all SKU context in one round-trip.
_SQL_MAIN_SKU_CTX = """
    SELECT
        ph.sku_id::text,
        ph.pattern_label,
        ph.confidence_calibrated,
        ph.model_hint,
        ph.observation_days        AS obs_days,
        ph.lifecycle_stage,
        ph.composite_confidence,
        ph.drift_detected,
        ph.weekend_zero_ratio,
        ph.velocity_signature,
        COALESCE(sc.on_watchlist, FALSE)             AS on_watchlist,
        COALESCE(fd.feature_reliability_map, '{}'::jsonb) AS feature_reliability_map,
        cs.vendor,
        cs.product_type,
        cs.parent_style_id::text,
        cs.shelf_life_days,
        cs.planned_end_date,
        cs.criticality_tier,
        cs.service_level_target,
        cs.product_lifecycle_type,
        cs.seed_daily_demand
    FROM stage8.pattern_history ph
    LEFT JOIN stage8.feature_decisions fd
           ON fd.sku_id = ph.sku_id
          AND fd.tenant_id = ph.tenant_id
    LEFT JOIN stage8.canonical_sku cs
           ON cs.sku_id = ph.sku_id
    LEFT JOIN stage8.signal_context sc
           ON sc.sku_id = ph.sku_id
          AND sc.tenant_id = ph.tenant_id
    WHERE ph.tenant_id = %s
"""

# Read 2 — OOS impact estimates, one row per SKU (newest first).
_SQL_OOS = """
    SELECT DISTINCT ON (sku_id)
        sku_id::text,
        oos_pct,
        detection_confidence
    FROM stage8.oos_impact_estimates
    WHERE tenant_id = %s
      AND (expires_at IS NULL OR expires_at > NOW())
    ORDER BY sku_id, created_at DESC
"""

# Read 3 — Channel demand splits. Only run when pipeline_mode = 'multi_channel'.
_SQL_CHANNEL_SPLITS = """
    SELECT sku_id::text, sale_date, qty::float, channel
    FROM stage8.channel_demand_splits
    WHERE tenant_id = %s
    ORDER BY sku_id, sale_date
"""

# Read 4 — Promo decisions. Key: (sku_id, promo_date.isoformat()).
_SQL_PROMO = """
    SELECT sku_id::text, promo_date, multiplier::float
    FROM stage8.promo_decisions
    WHERE tenant_id = %s
"""

# Read 5 — Portfolio intelligence reports, last 90 days, 4 alert types.
_SQL_PORTFOLIO_ALERTS = """
    SELECT alert_type, payload, created_at
    FROM stage8.portfolio_intelligence_reports
    WHERE tenant_id = %s
      AND created_at >= NOW() - INTERVAL '90 days'
      AND alert_type IN (
            'market_shift', 'channel_count_changed',
            'data_density_drop', 'structural_break'
          )
    ORDER BY created_at DESC
"""

# Read 6 — Stage 8 tenant-level confidence thresholds.
_SQL_TENANT_THRESHOLDS = """
    SELECT confidence_floor, confidence_ceiling
    FROM stage8.tenant_thresholds
    WHERE tenant_id = %s
    LIMIT 1
"""

# Read 7 — Thompson sampling state, all configs for all SKU-model pairs.
_SQL_THOMPSON_STATE = """
    SELECT
        sku_id::text,
        assigned_model,
        config_hash,
        config_json,
        alpha_param,
        beta_param,
        total_trials
    FROM stage9.thompson_sampling_state
    WHERE tenant_id = %s
"""

# Feature history from Stage 9's own prior runs (warm-start for sub_stage_92).
_SQL_FEATURE_HISTORY = """
    SELECT DISTINCT ON (sku_id)
        sku_id::text,
        features_used
    FROM stage9.feature_decisions_s9
    WHERE tenant_id = %s
    ORDER BY sku_id, created_at DESC
"""

# Fingerprint cache — cached fingerprint from the prior run.
# Missing rows → classify_tier returns "full" automatically.
_SQL_FINGERPRINT_CACHE = """
    SELECT sku_id::text, fingerprint, pattern_label, demand_total
    FROM stage9.data_fingerprint_cache
    WHERE tenant_id = %s
"""

# Demand summary — last 30 days of daily qty per SKU for fingerprint computation.
# Lighter than the full 400-day load in acting_handler.
_SQL_DEMAND_SUMMARY = """
    SELECT sku_id::text, qty::float
    FROM stage8.demand_history
    WHERE tenant_id = %s
      AND sale_date >= CURRENT_DATE - INTERVAL '30 days'
    ORDER BY sku_id, sale_date
"""


# ---------------------------------------------------------------------------
# Preloader
# ---------------------------------------------------------------------------

class Preloader:
    """
    Runs all 7 bulk queries plus TenantParams and signal_context loads.

    Instantiate once per run. After calling load():
      - preloaded  — fully populated PreloadedData container
      - self.params — TenantParams snapshot (validated in PERCEIVING state)
      - self.signal_ctx — tenant-level pipeline_mode / tenant_maturity
    """

    def __init__(self) -> None:
        self.params: Optional[Any] = None  # TenantParams set by load()
        self.signal_ctx: dict = {}

    def load(self, tenant_id: str, db: Any) -> PreloadedData:
        from infrastructure.tenant_params import TenantParams

        self.params = TenantParams.load(tenant_id, db)

        preloaded = PreloadedData(tenant_id=tenant_id)

        with db.cursor() as cur:
            # tenant-level context needed before Read 3 gate check
            self._read_signal_context(cur, tenant_id, preloaded)

            # Read 1 — 3-way JOIN
            self._read_main_sku_context(cur, tenant_id, preloaded)

            # Read 2 — OOS
            self._read_oos(cur, tenant_id, preloaded)

            # Read 3 — channel splits (conditional)
            if preloaded.signal_context.get("pipeline_mode") == "multi_channel":
                self._read_channel_splits(cur, tenant_id, preloaded)

            # Read 4 — promo decisions
            self._read_promo_decisions(cur, tenant_id, preloaded)

            # Read 5 — portfolio alerts
            self._read_portfolio_alerts(cur, tenant_id, preloaded)

            # Read 6 — tenant thresholds
            self._read_tenant_thresholds(cur, tenant_id, preloaded)

            # Read 7 — Thompson sampling state
            self._read_thompson_state(cur, tenant_id, preloaded)

            # Stage 9 own prior feature selections (warm-start)
            self._read_feature_history(cur, tenant_id, preloaded)

            # Fingerprint cache (prior-run baseline for tier classification)
            self._read_fingerprint_cache(cur, tenant_id, preloaded)

            # Demand summary — last 30 days per SKU (for fingerprint computation)
            demand_summary = self._read_demand_summary(cur, tenant_id)

        # Tier classification — pure Python, no DB I/O
        self._compute_sku_tiers(preloaded, demand_summary)

        return preloaded

    # ---- private read methods ----------------------------------------

    def _read_signal_context(
            self, cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_SIGNAL_CONTEXT, (tenant_id,))
        row = cur.fetchone()
        if row:
            pipeline_mode, tenant_maturity = row
            preloaded.signal_context["pipeline_mode"] = pipeline_mode
            preloaded.signal_context["tenant_maturity"] = tenant_maturity
            self.signal_ctx = dict(preloaded.signal_context)

    @staticmethod
    def _read_main_sku_context(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_MAIN_SKU_CTX, (tenant_id,))
        for row in cur.fetchall():
            (
                sku_id, pattern_label, confidence_calibrated, model_hint,
                obs_days, lifecycle_stage, composite_confidence,
                drift_detected, weekend_zero_ratio, velocity_signature,
                on_watchlist, feature_reliability_map,
                vendor, product_type, parent_style_id,
                shelf_life_days, planned_end_date, criticality_tier,
                service_level_target, product_lifecycle_type, seed_daily_demand,
            ) = row

            rel_map = feature_reliability_map if isinstance(feature_reliability_map, dict) else {}

            preloaded.pattern_ctx[sku_id] = {
                "pattern_label": pattern_label,
                "confidence_calibrated": float(confidence_calibrated) if confidence_calibrated is not None else None,
                "model_hint": model_hint,
                "obs_days": int(obs_days or 0),
                "lifecycle_stage": lifecycle_stage,
                "composite_confidence": float(composite_confidence) if composite_confidence is not None else None,
                "drift_detected": bool(drift_detected),
                "weekend_zero_ratio": float(weekend_zero_ratio or 0.0),
                "velocity_signature": velocity_signature,
                "on_watchlist": bool(on_watchlist),
                "feature_reliability_map": rel_map,
                "vendor": vendor,
                "product_type": product_type,
                "parent_style_id": parent_style_id,
                "shelf_life_days": int(shelf_life_days) if shelf_life_days is not None else None,
                "planned_end_date": planned_end_date,
                "criticality_tier": criticality_tier,
                "service_level_target": float(service_level_target) if service_level_target is not None else None,
                "product_lifecycle_type": product_lifecycle_type,
                "seed_daily_demand": float(seed_daily_demand) if seed_daily_demand is not None else None,
            }

            # sku_metadata — backward-compat slice for sub_stage_91
            preloaded.sku_metadata[sku_id] = {
                "criticality_tier": criticality_tier,
                "service_level_target": float(service_level_target) if service_level_target is not None else None,
                "planned_end_date": planned_end_date,
                "shelf_life_days": int(shelf_life_days) if shelf_life_days is not None else None,
                "product_lifecycle_type": product_lifecycle_type,
                "seed_daily_demand": float(seed_daily_demand) if seed_daily_demand is not None else None,
            }

            # feature_reliability — per-feature reliability scores for sub_stage_92
            if rel_map:
                preloaded.feature_reliability[sku_id] = rel_map

    @staticmethod
    def _read_oos(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_OOS, (tenant_id,))
        for sku_id, oos_pct, det_conf in cur.fetchall():
            preloaded.oos_ctx[sku_id] = {
                "oos_pct": float(oos_pct or 0.0),
                "detection_confidence": float(det_conf or 0.0),
            }

    @staticmethod
    def _read_channel_splits(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_CHANNEL_SPLITS, (tenant_id,))
        for sku_id, sale_date, qty, channel in cur.fetchall():
            preloaded.channel_splits.setdefault(sku_id, []).append(
                {"sale_date": sale_date, "qty": qty, "channel": channel}
            )

    @staticmethod
    def _read_promo_decisions(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_PROMO, (tenant_id,))
        for sku_id, promo_date, multiplier in cur.fetchall():
            preloaded.promo_decisions[(sku_id, promo_date.isoformat())] = float(multiplier)

    @staticmethod
    def _read_portfolio_alerts(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_PORTFOLIO_ALERTS, (tenant_id,))
        for alert_type, payload, created_at in cur.fetchall():
            entry = {"alert_type": alert_type, "created_at": created_at}
            if isinstance(payload, dict):
                entry.update(payload)
            preloaded.portfolio_alerts.append(entry)

    @staticmethod
    def _read_tenant_thresholds(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_TENANT_THRESHOLDS, (tenant_id,))
        row = cur.fetchone()
        if row:
            confidence_floor, confidence_ceiling = row
            preloaded.tenant_thresholds = {
                "confidence_floor": float(confidence_floor) if confidence_floor is not None else None,
                "confidence_ceiling": float(confidence_ceiling) if confidence_ceiling is not None else None,
            }

    @staticmethod
    def _read_thompson_state(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_THOMPSON_STATE, (tenant_id,))
        for sku_id, model, cfg_hash, cfg_json, alpha, beta, trials in cur.fetchall():
            if sku_id not in preloaded.thompson_ctx:
                preloaded.thompson_ctx[sku_id] = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "historical_runs": int(trials),
                }
            state_key = (sku_id, model)
            if state_key not in preloaded.thompson_state:
                preloaded.thompson_state[state_key] = {}
            preloaded.thompson_state[state_key][cfg_hash] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "config": cfg_json,
            }

    @staticmethod
    def _read_feature_history(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_FEATURE_HISTORY, (tenant_id,))
        for sku_id, features_json in cur.fetchall():
            preloaded.feature_history[sku_id] = (
                features_json if isinstance(features_json, list) else []
            )

    @staticmethod
    def _read_fingerprint_cache(
            cur: Any, tenant_id: str, preloaded: PreloadedData,
    ) -> None:
        cur.execute(_SQL_FINGERPRINT_CACHE, (tenant_id,))
        for sku_id, fingerprint, pattern_label, demand_total in cur.fetchall():
            preloaded.fingerprint_cache[sku_id] = {
                "fingerprint": fingerprint,
                "pattern_label": pattern_label,
                "demand_total": float(demand_total) if demand_total is not None else None,
            }

    @staticmethod
    def _read_demand_summary(
            cur: Any, tenant_id: str,
    ) -> dict[str, list[float]]:
        cur.execute(_SQL_DEMAND_SUMMARY, (tenant_id,))
        summary: dict[str, list[float]] = {}
        for sku_id, qty in cur.fetchall():
            summary.setdefault(sku_id, []).append(float(qty or 0.0))
        return summary

    @staticmethod
    def _compute_sku_tiers(
            preloaded: PreloadedData,
            demand_summary: dict[str, list[float]],
    ) -> None:
        from forecasting.fingerprint import compute_fingerprint, classify_tier

        for sku_id, pctx in preloaded.pattern_ctx.items():
            sales = demand_summary.get(sku_id, [])
            oos_pct = preloaded.oos_ctx.get(sku_id, {}).get("oos_pct", 0.0)
            pattern_label = pctx["pattern_label"]
            demand_total = float(sum(sales))

            fp = compute_fingerprint(
                sku_id=sku_id,
                sales_last_30d=sales,
                pattern_label=pattern_label,
                oos_pct=oos_pct,
                lifecycle_stage=pctx.get("lifecycle_stage"),
            )

            tier = classify_tier(
                sku_id=sku_id,
                current_fingerprint=fp,
                fingerprint_cache=preloaded.fingerprint_cache,
                current_pattern_label=pattern_label,
                current_demand_total=demand_total,
            )

            # Override 1: drift detected → force full.
            # Structural break invalidates the cached model; the fingerprint
            # may still match if sales figures are identical this run.
            if tier != "full" and pctx.get("drift_detected"):
                tier = "full"

            # Override 2: on watchlist → floor at partial.
            # Cache bypasses 9.4 and 9.5 entirely; watchlist SKUs must reach
            # 9.5 so that their status is written as 'watchlist_review'.
            elif tier == "cache" and pctx.get("on_watchlist"):
                tier = "partial"

            preloaded.sku_tiers[sku_id] = tier
            preloaded.new_fingerprints[sku_id] = {
                "fingerprint": fp,
                "pattern_label": pattern_label,
                "demand_total": demand_total,
            }
