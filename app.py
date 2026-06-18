"""
app.py — RECEIV MVP, Layer 3 UI (Streamlit).  PASS 1 of 2: structure + Club view.

Run with:  streamlit run app.py

This is PURE UI on top of the existing layers — it imports and calls credit_model (Layer 2)
and marketplace (Layer 3) and never re-implements scoring or auction logic.

Pass 1 delivers the full Club side (the selling club that wants cash now for a receivable it
is owed): pick a demo listing OR score a brand-new receivable, see the explainable A-E credit
decision, and list a new receivable on the marketplace. The Investor side is a placeholder
that pass 2 will implement.

All data is SYNTHETIC (see generate_data.py).
"""

import json
import uuid

import streamlit as st

import credit_model as cm
import marketplace as mp

# --------------------------------------------------------------------------------------
# Page config + one-time bootstrap
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="RECEIV", page_icon="⚽", layout="wide")


@st.cache_resource
def bootstrap():
    """Create the DB once and seed the 20 demo listings if the marketplace is empty."""
    mp.init_db()
    if not mp.get_open_listings():
        mp.seed_listings()
    return True


bootstrap()


# --------------------------------------------------------------------------------------
# Presentation helpers (rating colours, value formatting) — kept tiny and native
# --------------------------------------------------------------------------------------
# A->E mapped to a green->red scale. Emoji gives the dot; the name is a Streamlit markdown
# colour for the big letter. (Streamlit markdown supports green/blue/orange/red/violet/gray.)
RATING_STYLE = {
    "A": ("🟢", "green"),
    "B": ("🔵", "blue"),
    "C": ("🟡", "orange"),
    "D": ("🟠", "orange"),
    "E": ("🔴", "red"),
}

# Per-feature input config for the manual-entry form: (min, max, step, default, help text).
# Ranges mirror the synthetic data generator so inputs stay in a realistic envelope.
FEATURE_INPUTS = {
    "wage_to_revenue": (0.45, 1.10, 0.01, 0.60,
        "Wages as a share of revenue. Above 0.70 breaches the FSR cap; above 0.85 trips the hard filter."),
    "operating_cash_flow_eur_m": (-30.0, 80.0, 0.5, 20.0,
        "Annual operating cash flow (€m). Negative trips the hard filter."),
    "debt_to_assets": (0.10, 0.90, 0.01, 0.40,
        "Total debt over total assets. Above 0.80 trips the hard filter."),
    "regulatory_headroom_eur_m": (-20.0, 60.0, 0.5, 20.0,
        "Spare room under financial-sustainability rules (€m). Negative trips the hard filter."),
    "uefa_coefficient_trend": (-15.0, 15.0, 0.1, 0.0,
        "Trend in the club's UEFA coefficient. Negative = declining."),
    "league_position_trend": (-10.0, 10.0, 0.1, 0.0,
        "Trend in league position. Negative = sliding down the table."),
    "broadcast_revenue_share": (0.20, 0.70, 0.01, 0.40,
        "Share of revenue that comes from broadcast deals."),
    "payment_timeliness_score": (0.0, 1.0, 0.01, 0.70,
        "Historical payment timeliness (1 = always on time). Below 0.30 trips the hard filter. Strongest predictor."),
    "management_tenure_years": (0.0, 12.0, 0.5, 5.0,
        "Years of stability in the current management. Longer mildly reduces risk."),
}

LEAGUES = ["Eredivisie", "Bundesliga", "Belgian Pro League"]


def _eur(x):
    return f"€{x:,.0f}"


def rating_to_features(listing):
    """Pull the nine model features out of a stored listing dict."""
    return {f: listing[f] for f in cm.FEATURES}


# --------------------------------------------------------------------------------------
# Shared component: render a full, explainable credit decision
# --------------------------------------------------------------------------------------
def render_credit_decision(decision):
    """
    Show a score_club() result clearly: prominent A-E rating, PD + floor metrics, a hard-filter
    warning when tripped, and the top risk drivers as readable lines (the explainability story).
    """
    emoji, colour = RATING_STYLE[decision["rating"]]

    top = st.columns([1.1, 1, 1])
    with top[0]:
        st.caption("Credit rating")
        # Big, colour-coded letter — the headline of the whole decision.
        st.markdown(f"# {emoji} :{colour}[{decision['rating']}]")
        grade = "Investment grade" if decision["investment_grade"] else "Sub-investment grade"
        st.caption(grade)
    with top[1]:
        st.metric("Probability of default", f"{decision['probability_of_default'] * 100:.1f}%")
    with top[2]:
        st.metric("Recommended floor rate", f"{decision['floor_rate_pct']:.1f}%",
                  help="Minimum discount rate RECEIV recommends; bids below this are rejected.")

    # Hard-filter banner: structural red flags that capped the rating out of investment grade.
    if decision["hard_filter_triggered"]:
        reasons = "\n".join(f"- {r}" for r in decision["hard_filter_reasons"])
        st.error(
            f"**Hard filter triggered — capped to {decision['rating']} "
            f"(model alone said {decision['model_rating']}).**\n\n"
            f"Structurally distressed; blocked from investment grade:\n\n{reasons}"
        )

    # Top risk drivers — the per-prediction explanation, in plain language.
    st.markdown("**Why this rating — top risk drivers**")
    for d in decision["top_drivers"]:
        if d["direction"] == "increases risk":
            arrow, dcol = "🔺", "red"
        else:
            arrow, dcol = "🔻", "green"
        st.markdown(f"{arrow} **{d['label']}** = {d['value']:,.2f}  —  :{dcol}[{d['direction']}]")
    st.caption("Drivers come from the model's own SHAP contributions for this specific club.")


def render_deal_summary(listing):
    """Show the receivable's cash structure (face value + the 3 annual installments)."""
    st.markdown(
        f"**{listing['selling_club']}** is owed **{_eur(listing['face_value_eur'])}** by "
        f"**{listing['paying_club']}** ({listing['paying_club_league']})."
    )
    c = st.columns(4)
    c[0].metric("Face value", _eur(listing["face_value_eur"]))
    c[1].metric("Installment 1", _eur(listing["installment_1_eur"]))
    c[2].metric("Installment 2", _eur(listing["installment_2_eur"]))
    c[3].metric("Installment 3", _eur(listing["installment_3_eur"]))


# --------------------------------------------------------------------------------------
# Persistence: write a newly scored receivable to the store as an OPEN listing.
# We deliberately use marketplace's public connection helper + schema rather than adding a
# function to marketplace.py (this pass keeps that module unchanged — UI only).
# --------------------------------------------------------------------------------------
def persist_new_listing(meta, features, decision):
    """Insert a freshly scored receivable as an open listing; return its new listing_id."""
    listing_id = f"NEW-{uuid.uuid4().hex[:6].upper()}"
    cols = (
        ["listing_id", "selling_club", "paying_club", "paying_club_league", "face_value_eur",
         "installment_1_eur", "installment_2_eur", "installment_3_eur"]
        + cm.FEATURES
        + ["probability_of_default", "rating", "model_rating", "floor_rate_pct",
           "hard_filter_triggered", "hard_filter_reasons", "status"]
    )
    values = (
        [listing_id, meta["selling_club"], meta["paying_club"], meta["league"],
         int(meta["face_value"]), int(meta["inst1"]), int(meta["inst2"]), int(meta["inst3"])]
        + [features[f] for f in cm.FEATURES]
        + [decision["probability_of_default"], decision["rating"], decision["model_rating"],
           decision["floor_rate_pct"], int(decision["hard_filter_triggered"]),
           json.dumps(decision["hard_filter_reasons"]), "open"]
    )
    placeholders = ", ".join("?" for _ in cols)
    with mp.get_conn() as conn:
        conn.execute(f"INSERT INTO listings ({', '.join(cols)}) VALUES ({placeholders})", values)
    return listing_id


# --------------------------------------------------------------------------------------
# Sidebar: role switcher, synthetic-data note, reset control
# --------------------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("RECEIV")
    role = st.sidebar.radio("I am a…", ["Club", "Investor"], index=0)

    st.sidebar.divider()
    st.sidebar.metric("Open listings", len(mp.get_open_listings()))

    st.sidebar.divider()
    st.sidebar.caption("**Reset demo** — wipe all bids/settlements and reseed the 20 demo deals.")
    if st.sidebar.button("🔄 Reset demo to clean state", use_container_width=True):
        mp.seed_listings(reset=True)            # reset + reseed (Layer 3)
        st.session_state.pop("new_score", None)  # drop any in-progress scoring
        st.session_state.pop("last_listed", None)
        st.sidebar.success("Demo reset.")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.info("⚠️ Academic MVP — all clubs, financials and ratings are **synthetic**.")
    return role


# --------------------------------------------------------------------------------------
# CLUB VIEW
# --------------------------------------------------------------------------------------
def club_view():
    st.subheader("Club — turn a transfer receivable into cash now")
    st.write(
        "You are owed a deferred transfer fee. Get it rated, then list it so institutional "
        "investors can bid to fund it today (you receive cash up front, at a discount)."
    )

    tab_view, tab_score = st.tabs(["📂 View a demo listing", "🧮 Score a new receivable"])

    # --- (a) Inspect an existing, already-scored demo listing -----------------------------
    with tab_view:
        listings = mp.get_open_listings()
        if not listings:
            st.warning("No open listings. Use the sidebar to reset the demo.")
        else:
            labels = {
                f"{l['listing_id']} — {l['paying_club']} (pays) · rating {l['rating']}": l["listing_id"]
                for l in listings
            }
            chosen = st.selectbox("Pick a demo receivable to inspect", list(labels.keys()))
            listing = mp.get_listing(labels[chosen])

            render_deal_summary(listing)
            st.divider()
            # Re-score the stored features so we also get the live top-driver explanation.
            decision = cm.score_club(rating_to_features(listing))
            render_credit_decision(decision)

    # --- (b) Score a brand-new receivable via the manual-entry form -----------------------
    with tab_score:
        st.caption(
            "📝 This form stands in for the **stubbed Layer-1 contract parser**. In production "
            "these fields are auto-extracted from the uploaded transfer-agreement PDF; here we "
            "enter the structured object by hand."
        )

        with st.form("score_form"):
            st.markdown("**Deal**")
            d1, d2, d3 = st.columns(3)
            selling_club = d1.text_input("Selling club (you)", "FC Demo United")
            paying_club = d2.text_input("Paying club (obligor)", "Real Synthetica")
            league = d3.selectbox("Paying club league", LEAGUES)

            face_value = st.number_input(
                "Face value owed (€)", min_value=1_000_000, max_value=80_000_000,
                value=20_000_000, step=500_000,
                help="Total still owed across the installments.",
            )
            i1, i2, i3 = st.columns(3)
            inst1 = i1.number_input("Installment 1 (€)", min_value=0, value=8_000_000, step=250_000)
            inst2 = i2.number_input("Installment 2 (€)", min_value=0, value=7_000_000, step=250_000)
            inst3 = i3.number_input("Installment 3 (€)", min_value=0, value=5_000_000, step=250_000)

            st.markdown("**Paying club credit features**")
            feat_inputs = {}
            grid = st.columns(3)
            for idx, feature in enumerate(cm.FEATURES):
                lo, hi, step, default, helptext = FEATURE_INPUTS[feature]
                feat_inputs[feature] = grid[idx % 3].number_input(
                    cm.FEATURE_LABELS[feature], min_value=lo, max_value=hi,
                    value=default, step=step, help=helptext,
                )

            submitted = st.form_submit_button("🧮 Score receivable", use_container_width=True)

        if submitted:
            # Gentle validation: installments should add up to the face value.
            if abs((inst1 + inst2 + inst3) - face_value) > 1:
                st.warning(
                    f"Installments sum to {_eur(inst1 + inst2 + inst3)} but face value is "
                    f"{_eur(face_value)}. Scoring anyway; settlement uses the installments."
                )
            decision = cm.score_club(feat_inputs)  # Layer-2 call
            # Stash the full result so it survives the rerun caused by the "List" button.
            st.session_state["new_score"] = {
                "meta": {
                    "selling_club": selling_club, "paying_club": paying_club, "league": league,
                    "face_value": face_value, "inst1": inst1, "inst2": inst2, "inst3": inst3,
                },
                "features": feat_inputs,
                "decision": decision,
            }
            st.session_state.pop("last_listed", None)

        # Show the scored decision + the "List on marketplace" action (outside the form).
        if st.session_state.get("new_score"):
            scored = st.session_state["new_score"]
            st.divider()
            render_credit_decision(scored["decision"])

            st.divider()
            if st.button("📢 List on marketplace", type="primary", use_container_width=True):
                new_id = persist_new_listing(scored["meta"], scored["features"], scored["decision"])
                st.session_state["last_listed"] = new_id
                st.session_state.pop("new_score", None)
                st.rerun()

        # Confirmation after a successful listing.
        if st.session_state.get("last_listed"):
            st.success(
                f"✅ Listed as **{st.session_state['last_listed']}** — it's now an open listing "
                "investors can browse and bid on (Investor view arrives in pass 2)."
            )


# --------------------------------------------------------------------------------------
# INVESTOR VIEW (placeholder for pass 2)
# --------------------------------------------------------------------------------------
def investor_view():
    st.subheader("Investor")
    st.info(
        "🚧 **Coming in pass 2.** The investor side will browse live listings, open one to see "
        "its credit breakdown and feature importances, place a sealed bid (the discount rate it "
        "requires), and the auction will match the lowest qualifying bid and show the settlement."
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    st.title("RECEIV — Transfer Receivables Marketplace")
    st.caption("Turning illiquid football transfer receivables into rated, tradeable claims. "
               "Demo on synthetic data.")

    role = render_sidebar()
    st.divider()

    if role == "Club":
        club_view()
    else:
        investor_view()


main()
