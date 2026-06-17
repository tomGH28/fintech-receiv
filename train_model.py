"""
train_model.py — RECEIV MVP, Layer 2: train, evaluate and save the credit risk model.

Trains an XGBoost classifier to predict the synthetic `defaulted` label from the nine
business-plan club features, evaluates it on a held-out stratified test split, and saves:

    models/credit_model.json        - the booster in XGBoost native format
    models/credit_model_meta.json   - feature list, rating bands, base rate and spreads,
                                      so the app (via credit_model.py) scores without
                                      retraining.

Leakage guard (critical): the model NEVER sees true_pd, the defaulted label as an input,
club_name, league, or annual_revenue_eur_m. annual_revenue is context only.

All data is SYNTHETIC (see generate_data.py).
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Reuse the single source of truth for features, bands, base rate and spreads.
from credit_model import (
    BASE_RATE_PCT,
    FEATURES,
    MODEL_PATH,
    META_PATH,
    RATING_BANDS,
    SPREADS_PP,
    pd_to_rating,
)

SEED = 42
TRAINING_CSV = os.path.join("data", "synthetic_training.csv")
MODELS_DIR = "models"

# Columns that must NEVER be used as model inputs (leakage / non-features).
FORBIDDEN_INPUTS = ["true_pd", "defaulted", "club_name", "league", "annual_revenue_eur_m"]


def load_training_data():
    """Load the labelled synthetic training set and split into X (9 features) and y."""
    df = pd.read_csv(TRAINING_CSV)

    # Defensive leakage check: confirm we are only feeding the nine intended features.
    assert set(FEATURES).isdisjoint(FORBIDDEN_INPUTS), "FEATURES overlaps forbidden inputs!"
    for col in FORBIDDEN_INPUTS:
        assert col not in FEATURES, f"Leakage: '{col}' must not be a model feature."

    X = df[FEATURES].copy()
    y = df["defaulted"].astype(int)
    return X, y


def train_and_evaluate(X, y):
    """Stratified 25% test split, train XGBoost, return the booster and a metrics dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # Modest, readable hyperparameters — enough for a clean ~0.9 AUC on synthetic data
    # without overfitting. We deliberately do NOT use scale_pos_weight: reweighting the
    # minority class inflates predicted PDs, and our rating + discount rate are derived
    # directly from the PD, so the probabilities must be CALIBRATED, not just rank-ordered.
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predicted probabilities of default on the held-out test set.
    pd_test = clf.predict_proba(X_test)[:, 1]

    y_test = y_test.reset_index(drop=True)

    metrics = {
        "test_auc": roc_auc_score(y_test, pd_test),
        "brier_score": brier_score_loss(y_test, pd_test),
        "confusion_matrix_0p5": confusion_matrix(y_test, (pd_test >= 0.5).astype(int)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_default_rate": float(y_test.mean()),
        # Calibration check: mean predicted PD should sit close to the actual default rate.
        "mean_predicted_pd": float(np.mean(pd_test)),
    }

    # Band assignment of the TEST clubs.
    test_ratings = pd.Series([pd_to_rating(p) for p in pd_test])

    # Distribution across A-E (confirms every band populates).
    metrics["test_band_distribution"] = test_ratings.value_counts().reindex(
        [r for r, _ in RATING_BANDS], fill_value=0
    )

    # Reliability table: for each band, the ACTUAL default rate of the test clubs in it.
    # A well-calibrated model puts each band's actual rate inside (or near) its PD range.
    reliability = pd.DataFrame({"rating": test_ratings, "defaulted": y_test})
    reliability = reliability.groupby("rating")["defaulted"].agg(["size", "mean"])
    reliability = reliability.reindex([r for r, _ in RATING_BANDS])
    metrics["reliability"] = reliability

    return clf, metrics, pd_test


def save_artifacts(clf, overall_default_rate):
    """Save the booster (native format) and a metadata JSON for the app."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Native XGBoost format so credit_model.py can load it as a plain Booster.
    clf.get_booster().save_model(MODEL_PATH)

    meta = {
        "features": FEATURES,
        "rating_bands": [{"rating": r, "pd_upper": b} for r, b in RATING_BANDS],
        "base_rate_pct": BASE_RATE_PCT,        # €STR proxy used for the discount floor
        "spreads_pp": SPREADS_PP,
        "overall_default_rate": round(float(overall_default_rate), 4),
        "data_note": "Model trained on SYNTHETIC data (generate_data.py).",
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    return MODEL_PATH, META_PATH


def main():
    X, y = load_training_data()
    clf, metrics, _ = train_and_evaluate(X, y)
    model_path, meta_path = save_artifacts(clf, overall_default_rate=y.mean())

    # ---- Report --------------------------------------------------------------------------
    print("=" * 78)
    print("RECEIV credit model (Layer 2) — trained on SYNTHETIC data")
    print("=" * 78)
    print(f"Features used ({len(FEATURES)}): {FEATURES}")
    print(f"Excluded to prevent leakage: {FORBIDDEN_INPUTS}")
    print(f"\nTrain rows: {metrics['n_train']}   Test rows: {metrics['n_test']}   "
          f"Test default rate: {metrics['test_default_rate']:.1%}")

    print("\n--- Test metrics ---")
    print(f"AUC               : {metrics['test_auc']:.3f}   (target ~0.91)")
    print(f"Brier score       : {metrics['brier_score']:.4f}  (lower is better)")
    print(f"Mean predicted PD : {metrics['mean_predicted_pd']:.3f}   "
          f"vs actual default rate {metrics['test_default_rate']:.3f}  (calibration)")

    cm = metrics["confusion_matrix_0p5"]
    print("\nConfusion matrix @ 0.5 threshold:")
    print("                 pred_no_default  pred_default")
    print(f"  actual_no_default   {cm[0, 0]:>10d}    {cm[0, 1]:>10d}")
    print(f"  actual_default      {cm[1, 0]:>10d}    {cm[1, 1]:>10d}")

    print("\n--- Reliability by band (actual default rate should fall in the PD range) ---")
    rel = metrics["reliability"]
    lower = 0.0
    for rating, upper in RATING_BANDS:
        n = rel.loc[rating, "size"]
        if pd.isna(n) or n == 0:
            print(f"  {rating} (PD {lower:.2f}-{upper:.2f}):   n=0   (no test clubs)")
        else:
            actual = rel.loc[rating, "mean"]
            print(f"  {rating} (PD {lower:.2f}-{upper:.2f}):   n={int(n):>3d}   "
                  f"actual default rate = {actual:.1%}")
        lower = upper

    print("\n--- Test-set A-E band distribution (confirms bands populate) ---")
    dist = metrics["test_band_distribution"]
    for rating, count in dist.items():
        pct = count / dist.sum()
        print(f"  {rating}: {count:>4d}  ({pct:.1%})")

    print(f"\nSaved model    -> {model_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
