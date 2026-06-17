"""
credit_model.py — RECEIV MVP, Layer 2: the reusable credit-scoring interface.

This is the single entry point the marketplace (Layer 3) calls to turn a paying club's
features into an explainable credit decision. It loads the trained XGBoost model produced
by train_model.py and applies the business-plan rules on top of it:

    raw features --> XGBoost PD --> A-E rating --> recommended discount-rate floor
                          |              ^
                          |              |
                          +--> hard filter (runs BEFORE the rating is finalised; can
                               cap a structurally distressed club out of investment grade)

Public API:
    load_model()                  -> loads the model + metadata once (cached)
    score_club(features: dict)    -> full, explainable credit decision (see RETURN below)

All thresholds, spreads and rules live here as documented constants so the logic is
auditable on camera. train_model.py imports these same constants and writes them into the
model metadata JSON, so there is a single source of truth.

NOTE: the model is trained on SYNTHETIC data (see generate_data.py). Nothing here is a
real credit rating.
"""

import json
import os

import numpy as np
import xgboost as xgb

# --------------------------------------------------------------------------------------
# The EXACTLY nine model features (order matters — it must match training).
# Leakage guard: true_pd, defaulted, club_name, league and annual_revenue_eur_m are
# DELIBERATELY absent. annual_revenue is context only and never an input.
# --------------------------------------------------------------------------------------
FEATURES = [
    "wage_to_revenue",
    "operating_cash_flow_eur_m",
    "debt_to_assets",
    "regulatory_headroom_eur_m",
    "uefa_coefficient_trend",
    "league_position_trend",
    "broadcast_revenue_share",
    "payment_timeliness_score",
    "management_tenure_years",
]

# Human-readable labels for explanations in the UI.
FEATURE_LABELS = {
    "wage_to_revenue": "Wage-to-revenue ratio",
    "operating_cash_flow_eur_m": "Operating cash flow (€m)",
    "debt_to_assets": "Debt-to-assets ratio",
    "regulatory_headroom_eur_m": "Regulatory headroom (€m)",
    "uefa_coefficient_trend": "UEFA coefficient trend",
    "league_position_trend": "League-position trend",
    "broadcast_revenue_share": "Broadcast-revenue share",
    "payment_timeliness_score": "Payment-timeliness score",
    "management_tenure_years": "Management tenure (years)",
}

# --------------------------------------------------------------------------------------
# Rating bands: probability of default -> letter grade.
# Stored as (rating, inclusive upper PD bound); the last band catches everything else.
# Investment grade = A and B only.
# --------------------------------------------------------------------------------------
RATING_BANDS = [
    ("A", 0.03),
    ("B", 0.07),
    ("C", 0.15),
    ("D", 0.30),
    ("E", 1.0),  # PD is bounded by 1.0; using 1.0 (not inf) keeps the metadata valid JSON
]
INVESTMENT_GRADE = ("A", "B")

# --------------------------------------------------------------------------------------
# Discount-rate floor = base rate + credit spread for the band.
# BASE_RATE_PCT stands in for the euro short-term rate (€STR). In production this is pulled
# live from the ECB; for this offline MVP we hard-code a recent value (~2.4%).
# --------------------------------------------------------------------------------------
BASE_RATE_PCT = 2.4  # €STR proxy (production: live ECB feed)

# Credit spread per rating band, in percentage points.
SPREADS_PP = {
    "A": 1.5,
    "B": 3.0,
    "C": 5.0,
    "D": 8.0,
    "E": 13.0,
}

# --------------------------------------------------------------------------------------
# HARD FILTER (business plan): structural red flags checked BEFORE the rating is finalised.
# If ANY rule trips, the club may NOT be investment grade — its rating is capped at C (or
# whatever the model already assigned, if that is worse). Each rule returns a plain-English
# reason so the UI can show exactly why a club was blocked (transparent / auditable).
#
# Each entry: feature -> (predicate, reason-template). The predicate takes the feature value
# and returns True when the red flag is present.
# --------------------------------------------------------------------------------------
HARD_FILTER_RULES = [
    ("regulatory_headroom_eur_m", lambda v: v < 0,
     "Negative regulatory headroom (€{v:.1f}m)"),
    ("operating_cash_flow_eur_m", lambda v: v < 0,
     "Negative operating cash flow (€{v:.1f}m)"),
    ("wage_to_revenue", lambda v: v > 0.85,
     "Wage-to-revenue {v:.2f} exceeds 0.85 cap"),
    ("debt_to_assets", lambda v: v > 0.80,
     "Debt-to-assets {v:.2f} exceeds 0.80"),
    ("payment_timeliness_score", lambda v: v < 0.30,
     "Poor payment timeliness ({v:.2f} < 0.30)"),
]

# The worst rating an investment-grade club is demoted to when the hard filter trips.
HARD_FILTER_RATING_CAP = "C"

# Default artifact locations (written by train_model.py).
MODEL_PATH = os.path.join("models", "credit_model.json")
META_PATH = os.path.join("models", "credit_model_meta.json")


# --------------------------------------------------------------------------------------
# Model loading (cached) — the app scores many clubs without retraining.
# --------------------------------------------------------------------------------------
_CACHE = {}


def load_model(model_path=MODEL_PATH, meta_path=META_PATH):
    """
    Load the trained booster and its metadata once and cache it.

    Returns a dict: {"booster": xgb.Booster, "meta": {...}, "features": [...]}.
    Raises a clear error if the artifacts are missing (run train_model.py first).
    """
    cache_key = (model_path, meta_path)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run `python train_model.py` first."
        )

    booster = xgb.Booster()
    booster.load_model(model_path)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    # Trust the trained feature order if present; otherwise fall back to the module list.
    features = meta.get("features", FEATURES)

    loaded = {"booster": booster, "meta": meta, "features": features}
    _CACHE[cache_key] = loaded
    return loaded


# --------------------------------------------------------------------------------------
# Rating / floor / hard-filter helpers (pure functions — easy to read and test)
# --------------------------------------------------------------------------------------
def pd_to_rating(pd_value):
    """Map a probability of default to its A-E band."""
    for rating, upper in RATING_BANDS:
        if pd_value <= upper:
            return rating
    return RATING_BANDS[-1][0]  # unreachable (last bound is +inf), kept for safety


def rating_to_floor_pct(rating):
    """Recommended minimum discount rate = €STR base rate + the band's credit spread."""
    return round(BASE_RATE_PCT + SPREADS_PP[rating], 2)


def run_hard_filter(features):
    """
    Evaluate the structural red-flag rules.

    Returns (triggered: bool, reasons: list[str]).
    """
    reasons = []
    for feature, predicate, template in HARD_FILTER_RULES:
        value = features[feature]
        if predicate(value):
            reasons.append(template.format(v=value))
    return (len(reasons) > 0), reasons


def _apply_filter_cap(model_rating, hard_filter_triggered):
    """
    If the hard filter tripped, an investment-grade rating (A/B) is demoted to the cap (C).
    A rating already at or below the cap is left untouched (the filter never improves it).
    """
    if not hard_filter_triggered:
        return model_rating
    if model_rating in INVESTMENT_GRADE:
        return HARD_FILTER_RATING_CAP
    return model_rating


# --------------------------------------------------------------------------------------
# Explainability: top risk drivers via XGBoost's native per-prediction SHAP contributions
# --------------------------------------------------------------------------------------
def _top_drivers(booster, dmatrix, features, n=4):
    """
    Use XGBoost's built-in SHAP values (pred_contribs=True) to find the features that most
    moved THIS club's risk. Contributions are in log-odds space; the final column is the
    model bias and is dropped. Positive contribution => pushed default risk UP.
    """
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # shape: n_features + 1
    feature_contribs = contribs[: len(FEATURES)]

    # Rank by absolute impact, keep the top n.
    order = np.argsort(np.abs(feature_contribs))[::-1][:n]

    drivers = []
    for idx in order:
        name = FEATURES[idx]
        contribution = float(feature_contribs[idx])
        drivers.append({
            "feature": name,
            "label": FEATURE_LABELS[name],
            "value": float(features[name]),
            "contribution": round(contribution, 4),
            # Direction from the club's perspective: does this driver raise or lower risk?
            "direction": "increases risk" if contribution > 0 else "reduces risk",
        })
    return drivers


# --------------------------------------------------------------------------------------
# The public scoring entry point
# --------------------------------------------------------------------------------------
def score_club(features, model=None):
    """
    Score one paying club and return a full, explainable credit decision.

    Args:
        features: dict containing at least the nine FEATURES (extra keys are ignored).
        model:    optional pre-loaded model dict from load_model(); loaded+cached if None.

    Returns a dict:
        {
          "probability_of_default": float,   # model PD in [0, 1]
          "rating": str,                     # final A-E rating (after the hard filter)
          "model_rating": str,               # rating from PD alone, before the filter cap
          "floor_rate_pct": float,           # recommended minimum discount rate
          "investment_grade": bool,          # final rating in {A, B}
          "hard_filter_triggered": bool,
          "hard_filter_reasons": list[str],
          "top_drivers": list[dict],         # 3-4 features that most moved this club's risk
        }
    """
    if model is None:
        model = load_model()
    booster = model["booster"]

    # Validate inputs and build a single-row DMatrix in the trained feature order.
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise KeyError(f"score_club is missing required features: {missing}")

    row = np.array([[float(features[f]) for f in FEATURES]], dtype=float)
    dmatrix = xgb.DMatrix(row, feature_names=FEATURES)

    # 1. Model probability of default.
    pd_value = float(booster.predict(dmatrix)[0])

    # 2. Rating from PD alone.
    model_rating = pd_to_rating(pd_value)

    # 3. Hard filter (runs BEFORE finalising the rating).
    triggered, reasons = run_hard_filter(features)

    # 4. Apply the investment-grade cap if the filter tripped.
    final_rating = _apply_filter_cap(model_rating, triggered)

    # 5. Discount-rate floor for the final rating.
    floor_rate_pct = rating_to_floor_pct(final_rating)

    # 6. Explainability.
    top_drivers = _top_drivers(booster, dmatrix, features)

    return {
        "probability_of_default": round(pd_value, 4),
        "rating": final_rating,
        "model_rating": model_rating,
        "floor_rate_pct": floor_rate_pct,
        "investment_grade": final_rating in INVESTMENT_GRADE,
        "hard_filter_triggered": triggered,
        "hard_filter_reasons": reasons,
        "top_drivers": top_drivers,
    }
