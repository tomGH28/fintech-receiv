"""
generate_data.py — RECEIV MVP, Build step 1: synthetic data generator.

Creates SYNTHETIC, clearly-labelled data for the RECEIV credit model (Layer 2).
NONE of this is real: club names are fictional, financials are drawn from random
distributions, and the default outcomes are simulated. The data exists only so the
credit model has something realistic-looking to learn from in an offline, no-network
academic MVP.

What it produces (into ./data/):
  1. synthetic_training.csv  — ~1500 paying-club observations WITH a `defaulted` label,
     used to train the XGBoost credit model.
  2. synthetic_listings.csv  — ~20 CURRENT receivables WITHOUT a label. These stand in
     for the (stubbed) Layer-1 contract-parser output and become the marketplace listings.

Modelling logic (see build_label / sample_clubs below):
  - Each club gets the business-plan feature set, with mild realistic correlations
    induced among the financial features (e.g. bigger clubs have stronger cash flow).
  - A club's true probability of default (PD) is built from a weighted linear
    combination of STANDARDISED features in the directions the business plan implies,
    plus a threshold penalty for breaching the ~0.70 wage-to-revenue FSR cap, plus a
    couple of mild feature interactions, plus gaussian noise, squashed through a sigmoid.
  - The binary `defaulted` label is then drawn from that PD (Bernoulli).
  - `true_pd` is saved for inspection but is NEVER a model feature (avoids leakage).

Everything is seeded for reproducibility.
"""

import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Reproducibility + configuration
# --------------------------------------------------------------------------------------
SEED = 42
N_TRAINING = 1500          # rows in the labelled training set
N_LISTINGS = 20            # current receivables on the marketplace
TARGET_DEFAULT_RATE = 0.13  # aim for ~12-15% overall default rate
DATA_DIR = "data"

rng = np.random.default_rng(SEED)

# The nine credit features, with their plausible ranges (from CLAUDE.md / business plan).
# Direction notes are encoded later in LABEL_WEIGHTS.
FEATURE_RANGES = {
    "wage_to_revenue": (0.45, 1.10),            # higher worse; sharp above 0.70 (FSR cap)
    "operating_cash_flow_eur_m": (-30.0, 80.0),  # negative worse
    "debt_to_assets": (0.10, 0.90),             # higher worse
    "regulatory_headroom_eur_m": (-20.0, 60.0),  # negative is high risk
    "uefa_coefficient_trend": (-15.0, 15.0),    # declining (negative) worse
    "league_position_trend": (-10.0, 10.0),     # declining (negative) worse
    "broadcast_revenue_share": (0.20, 0.70),    # higher mildly worse
    "payment_timeliness_score": (0.0, 1.0),     # higher = pays on time = better (STRONG)
    "management_tenure_years": (0.0, 12.0),     # higher = stability = mildly better
}
FEATURE_COLS = list(FEATURE_RANGES.keys())

# Fictional clubs placed in the real rollout leagues. Names are invented on purpose.
LEAGUE_CLUBS = {
    "Eredivisie": [
        "AFC Kanaalstad", "SC Lindehoven", "FC Maasdijk", "VV Polderkamp",
        "Sparta Veenendaal", "RKC Zandvoort-Noord", "FC Hertenbos", "Olympia Dollard",
    ],
    "Bundesliga": [
        "SV Rheinfelden", "FC Waldkirchen", "TSV Mooshausen", "Borussia Talheim",
        "SC Eichberg", "VfL Donaustein", "FC Nordhafen", "SpVgg Lindenau",
    ],
    "Belgian Pro League": [
        "KV Scheldebrug", "Royal Ardennes FC", "SK Meerhout", "RFC Lysveld",
        "KAS Dendermonde-West", "Union Kempen", "FC Walloon Rovers", "KVC Houtland",
    ],
}


def _scale_unit_to_range(unit_values, low, high):
    """Map values in [0, 1] onto [low, high]."""
    return low + unit_values * (high - low)


def sample_clubs(n):
    """
    Draw `n` synthetic clubs as a DataFrame of raw (un-standardised) features.

    We start from independent draws, then induce a few MILD, realistic correlations so
    the data isn't artificially clean: bigger-revenue clubs tend to have healthier cash
    flow, lower leverage and more regulatory headroom; clubs that overspend on wages tend
    to run down their headroom. Correlations are deliberately weak so no single feature
    trivially determines the others.
    """
    # annual_revenue_eur_m is context (not a credit feature). Use a 0..1 "size" factor to
    # gently drive the financial features that realistically scale with club size.
    size = rng.beta(2.0, 3.0, n)                      # skewed toward smaller clubs
    annual_revenue = _scale_unit_to_range(size, 20.0, 300.0)

    # Independent baselines in [0, 1] for each feature, then nudge by `size` where sensible.
    u = {f: rng.random(n) for f in FEATURE_COLS}

    # Wage-to-revenue: smaller clubs more likely to overstretch -> mild negative tie to size.
    wage_unit = np.clip(u["wage_to_revenue"] * 0.8 + (1 - size) * 0.2, 0, 1)

    # Operating cash flow: scales up with size (bigger clubs generate more cash).
    ocf_unit = np.clip(u["operating_cash_flow_eur_m"] * 0.7 + size * 0.3, 0, 1)

    # Debt-to-assets: bigger clubs slightly less leveraged.
    debt_unit = np.clip(u["debt_to_assets"] * 0.8 + (1 - size) * 0.2, 0, 1)

    # Regulatory headroom: shrinks when wages eat the budget (negative tie to wage_unit).
    head_unit = np.clip(u["regulatory_headroom_eur_m"] * 0.75 + (1 - wage_unit) * 0.25, 0, 1)

    df = pd.DataFrame({
        "annual_revenue_eur_m": np.round(annual_revenue, 1),
        "wage_to_revenue": np.round(_scale_unit_to_range(wage_unit, *FEATURE_RANGES["wage_to_revenue"]), 3),
        "operating_cash_flow_eur_m": np.round(_scale_unit_to_range(ocf_unit, *FEATURE_RANGES["operating_cash_flow_eur_m"]), 1),
        "debt_to_assets": np.round(_scale_unit_to_range(debt_unit, *FEATURE_RANGES["debt_to_assets"]), 3),
        "regulatory_headroom_eur_m": np.round(_scale_unit_to_range(head_unit, *FEATURE_RANGES["regulatory_headroom_eur_m"]), 1),
        "uefa_coefficient_trend": np.round(_scale_unit_to_range(u["uefa_coefficient_trend"], *FEATURE_RANGES["uefa_coefficient_trend"]), 2),
        "league_position_trend": np.round(_scale_unit_to_range(u["league_position_trend"], *FEATURE_RANGES["league_position_trend"]), 2),
        "broadcast_revenue_share": np.round(_scale_unit_to_range(u["broadcast_revenue_share"], *FEATURE_RANGES["broadcast_revenue_share"]), 3),
        "payment_timeliness_score": np.round(_scale_unit_to_range(u["payment_timeliness_score"], *FEATURE_RANGES["payment_timeliness_score"]), 3),
        "management_tenure_years": np.round(_scale_unit_to_range(u["management_tenure_years"], *FEATURE_RANGES["management_tenure_years"]), 1),
    })
    return df


# Weighted contribution of each STANDARDISED feature to the default log-odds.
# Sign encodes direction (positive => raises default risk). payment_timeliness carries the
# heaviest weight, as the business plan says payment history is the strongest predictor.
LABEL_WEIGHTS = {
    "wage_to_revenue": 0.80,
    "operating_cash_flow_eur_m": -0.70,
    "debt_to_assets": 0.75,
    "regulatory_headroom_eur_m": -0.85,
    "uefa_coefficient_trend": -0.40,
    "league_position_trend": -0.45,
    "broadcast_revenue_share": 0.25,        # only mildly worse
    "payment_timeliness_score": -1.60,      # STRONGEST predictor
    "management_tenure_years": -0.30,       # stability mildly reduces risk
}


def build_label(df):
    """
    Turn raw club features into a true PD and a drawn binary `defaulted` label.

    Steps (matching the CLAUDE.md spec):
      1. Standardise the nine features to z-scores.
      2. Weighted linear combination in the business-plan directions (LABEL_WEIGHTS).
      3. Threshold penalty for breaching the ~0.70 wage-to-revenue FSR cap.
      4. A couple of mild feature interactions.
      5. Add gaussian noise.
      6. Solve for an intercept so the mean PD hits the target default rate.
      7. Sigmoid -> true_pd, then draw `defaulted` ~ Bernoulli(true_pd).
    """
    # 1. Standardise (z-score) each feature.
    z = (df[FEATURE_COLS] - df[FEATURE_COLS].mean()) / df[FEATURE_COLS].std(ddof=0)

    # 2. Weighted linear combination.
    logit = sum(LABEL_WEIGHTS[f] * z[f] for f in FEATURE_COLS)

    # 3. Threshold effect: breaching the ~0.70 FSR wage cap is sharply worse. The penalty
    #    grows with how far over the cap the club is.
    over_cap = np.clip(df["wage_to_revenue"] - 0.70, 0, None)
    logit = logit + 3.0 * over_cap

    # 4. Mild interactions:
    #    - High leverage AND weak cash flow compounds distress.
    #    - Overspending wages AND thin regulatory headroom compounds distress.
    logit = logit + 0.30 * (z["debt_to_assets"] * (-z["operating_cash_flow_eur_m"]))
    logit = logit + 0.30 * (z["wage_to_revenue"] * (-z["regulatory_headroom_eur_m"]))

    # 5. Gaussian noise so the outcome isn't a deterministic function of features.
    logit = logit + rng.normal(0.0, 0.6, len(df))

    # 6. Find the intercept b such that mean(sigmoid(logit + b)) == TARGET_DEFAULT_RATE.
    #    Simple bisection — keeps the realised default rate on target regardless of weights.
    logit_vals = logit.to_numpy()

    def mean_pd_for(b):
        return float(np.mean(1.0 / (1.0 + np.exp(-(logit_vals + b)))))

    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if mean_pd_for(mid) > TARGET_DEFAULT_RATE:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2.0

    # 7. True PD and the drawn binary label.
    true_pd = 1.0 / (1.0 + np.exp(-(logit_vals + intercept)))
    defaulted = (rng.random(len(df)) < true_pd).astype(int)

    out = df.copy()
    out["true_pd"] = np.round(true_pd, 4)   # kept for inspection only — NOT a model feature
    out["defaulted"] = defaulted
    return out


def assign_club_identities(n, unique=False):
    """
    Attach a fictional club_name + its league to `n` rows.

    For training we sample with replacement (a club can recur across seasons). For the
    marketplace listings we want distinct paying clubs, so `unique=True` samples without
    replacement.
    """
    all_pairs = [(club, league) for league, clubs in LEAGUE_CLUBS.items() for club in clubs]
    idx = rng.choice(len(all_pairs), size=n, replace=not unique)
    names = [all_pairs[i][0] for i in idx]
    leagues = [all_pairs[i][1] for i in idx]
    return names, leagues


def build_training_set():
    """~1500 labelled club-seasons for training the credit model."""
    df = sample_clubs(N_TRAINING)
    df = build_label(df)
    names, leagues = assign_club_identities(N_TRAINING, unique=False)
    df.insert(0, "club_name", names)
    df.insert(1, "league", leagues)
    return df


def build_listings_set():
    """
    ~20 CURRENT receivables for the marketplace (no label — the future is unknown).

    Each listing is one transfer where a `selling_club` is owed money by a `paying_club`.
    The credit features describe the PAYING club (whose default risk we price). We add a
    face value and a simple 3-installment annual schedule. This stands in for the stubbed
    Layer-1 contract-parser output.
    """
    df = sample_clubs(N_LISTINGS)

    # Distinct paying clubs (the obligors we score).
    paying_names, paying_leagues = assign_club_identities(N_LISTINGS, unique=True)

    # Selling clubs: pick a different fictional club for each row (can be from any league).
    all_clubs = [c for clubs in LEAGUE_CLUBS.values() for c in clubs]
    selling_names = []
    for payer in paying_names:
        choice = payer
        while choice == payer:  # ensure seller != payer
            choice = all_clubs[rng.integers(0, len(all_clubs))]
        selling_names.append(choice)

    # Face value of the receivable: EUR 5m - 40m.
    face_value = np.round(rng.uniform(5_000_000, 40_000_000, N_LISTINGS), -3)

    # Simple 3-installment annual schedule: split the face value into three yearly chunks.
    # Slightly uneven (40% / 35% / 25%) to look like a real deferred-payment deal.
    inst_1 = np.round(face_value * 0.40, -3)
    inst_2 = np.round(face_value * 0.35, -3)
    inst_3 = np.round(face_value - inst_1 - inst_2, -3)  # remainder so the three sum exactly

    out = pd.DataFrame({
        "listing_id": [f"RCV-{i+1:03d}" for i in range(N_LISTINGS)],
        "selling_club": selling_names,
        "paying_club": paying_names,
        "paying_club_league": paying_leagues,
        "face_value_eur": face_value.astype(np.int64),
        "installment_1_eur": inst_1.astype(np.int64),
        "installment_2_eur": inst_2.astype(np.int64),
        "installment_3_eur": inst_3.astype(np.int64),
    })
    # Append the paying club's credit features (the inputs the model will score).
    out = pd.concat([out, df.reset_index(drop=True)], axis=1)
    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    training = build_training_set()
    listings = build_listings_set()

    training_path = os.path.join(DATA_DIR, "synthetic_training.csv")
    listings_path = os.path.join(DATA_DIR, "synthetic_listings.csv")
    training.to_csv(training_path, index=False)
    listings.to_csv(listings_path, index=False)

    # ---- Report (all data is SYNTHETIC) -------------------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("=" * 90)
    print("RECEIV synthetic data generator — ALL DATA IS SYNTHETIC (fictional clubs).")
    print("=" * 90)
    print(f"\nWrote {len(training)} rows -> {training_path}")
    print(f"Wrote {len(listings)} rows -> {listings_path}")

    print("\n--- synthetic_training.csv (5-row sample) ---")
    print(training.head().to_string(index=False))

    print("\n--- synthetic_listings.csv (5-row sample) ---")
    print(listings.head().to_string(index=False))

    realised_rate = training["defaulted"].mean()
    print(f"\nRealised default rate: {realised_rate:.1%}  (target ~{TARGET_DEFAULT_RATE:.0%})")

    # Sanity check: defaulted clubs should look worse on the fundamentals.
    print("\n--- Mean fundamentals: defaulted vs non-defaulted ---")
    check_cols = [
        "wage_to_revenue", "operating_cash_flow_eur_m", "debt_to_assets",
        "regulatory_headroom_eur_m", "payment_timeliness_score", "true_pd",
    ]
    summary = training.groupby("defaulted")[check_cols].mean().round(3)
    summary.index = summary.index.map({0: "non_defaulted", 1: "defaulted"})
    print(summary.to_string())

    print(
        "\nExpected pattern: defaulted clubs have HIGHER wage_to_revenue & debt_to_assets, "
        "LOWER cash flow, headroom & payment timeliness, and a HIGHER true_pd."
    )


if __name__ == "__main__":
    main()
