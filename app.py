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
        # Colour the arrow AND the label together so they agree: red = raises risk, green =
        # lowers it. Use a plain ▲/▼ glyph wrapped in Streamlit markdown colour (the fixed-
        # colour 🔺/🔻 emoji always render red, which is what made "reduces risk" look wrong).
        increases = d["direction"] == "increases risk"
        arrow = "▲" if increases else "▼"
        dcol = "red" if increases else "green"
        st.markdown(f":{dcol}[{arrow}] **{d['label']}** = {d['value']:,.2f}  —  :{dcol}[{d['direction']}]")
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
        st.session_state.pop("inv_selected", None)  # drop investor navigation
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
                "investors can browse and bid on (switch to the **Investor** role in the sidebar)."
            )


# --------------------------------------------------------------------------------------
# INVESTOR VIEW — three stages driven by listing status: browse -> detail+bid -> reveal.
# Navigation state lives in st.session_state["inv_selected"] (the listing being viewed).
# --------------------------------------------------------------------------------------
# A->E sort order (A best). Built from the model's own band order.
RATING_ORDER = {r: i for i, (r, _) in enumerate(cm.RATING_BANDS)}


def render_settlement(listing):
    """
    Stage 3 reveal for a MATCHED listing: unseal + rank every bid, then show the settlement
    economics with cash-to-club as the headline (the club's payoff).
    """
    bids = mp.get_bids(listing["listing_id"])
    settlement = mp.get_settlement(listing["listing_id"])

    st.success("🔓 Auction run — bids revealed below.")

    # Rank bids: winner first, then beaten (active->lost) cheapest-first, then rejected.
    def sort_key(b):
        priority = {"winning": 0, "lost": 1, "rejected": 2}.get(b["status"], 3)
        return (priority, b["bid_rate_pct"])

    st.markdown("**Bids (unsealed)**")
    for b in sorted(bids, key=sort_key):
        if b["status"] == "winning":
            st.markdown(f"🏆 :green[**{b['investor_name']} — {b['bid_rate_pct']:.2f}%**] · "
                        ":green[winning bid (lowest qualifying rate)]")
        elif b["status"] == "lost":
            # Greyed: qualified but outbid by a cheaper rate.
            st.markdown(f":gray[▫️ {b['investor_name']} — {b['bid_rate_pct']:.2f}% · outbid]")
        else:  # rejected
            st.markdown(f":red[🚫 {b['investor_name']} — {b['bid_rate_pct']:.2f}% · "
                        f"rejected (below {listing['floor_rate_pct']:.1f}% floor)]")

    st.divider()

    # --- Settlement economics -----------------------------------------------------------
    st.markdown("**Settlement**")
    face = listing["face_value_eur"]
    s = st.columns(3)
    s[0].metric("Accepted discount rate", f"{settlement['accepted_rate_pct']:.2f}%")
    s[1].metric("Present value of installments", _eur(settlement["present_value_eur"]),
                help="The three installments discounted at the accepted rate. Less than face "
                     "value because the money arrives over three years.")
    s[2].metric(f"RECEIV fee ({mp.RECEIV_FEE_PCT}% of face)", _eur(settlement["receiv_fee_eur"]))

    # Headline outcome for the club.
    st.metric(
        "💰 Cash to club today", _eur(settlement["cash_to_club_eur"]),
        delta=f"{_eur(settlement['cash_to_club_eur'] - face)} vs {_eur(face)} face value",
        delta_color="off",
    )
    st.caption("Cash to club = present value − RECEIV fee. The club gets paid now instead of "
               "waiting three years for the installments.")


def investor_detail(listing):
    """Stage 2/3: one listing — deal + credit breakdown, then either bidding or the reveal."""
    if st.button("← Back to listings"):
        st.session_state.pop("inv_selected", None)
        st.rerun()

    st.markdown(f"### {listing['listing_id']} — {listing['paying_club']}")
    render_deal_summary(listing)
    st.divider()

    # Same shared credit-decision component the club saw (re-scored for the live drivers).
    decision = cm.score_club(rating_to_features(listing))
    render_credit_decision(decision)
    st.divider()

    # If already matched, show the reveal instead of the bid form.
    if listing["status"] != "open":
        render_settlement(listing)
        return

    # --- Stage 2: place a sealed bid ----------------------------------------------------
    st.markdown("**Place a sealed bid**")
    st.caption("Bid the annual discount rate you require. Bids are SEALED — you cannot see "
               "others' bids (or their values) until the auction is run.")

    with st.form("bid_form"):
        b1, b2 = st.columns([2, 1])
        investor_name = b1.text_input("Investor name", "Acme Credit Partners")
        bid_rate = b2.number_input("Required discount rate (%)", min_value=0.0, max_value=40.0,
                                   value=round(listing["floor_rate_pct"] + 1.0, 1), step=0.1)
        placed = st.form_submit_button("🔒 Submit sealed bid", use_container_width=True)

    if placed:
        result = mp.place_bid(listing["listing_id"], investor_name, bid_rate)  # Layer-3 call
        if result["status"] == "rejected":
            # Honest feedback: below-floor bids underprice the risk and are rejected.
            st.error(
                f"Bid **{bid_rate:.2f}%** is below the **{listing['floor_rate_pct']:.1f}% floor** "
                "for this rating — rejected as underpricing the risk. Bid at or above the floor."
            )
        else:
            st.success(f"Sealed bid of **{bid_rate:.2f}%** recorded for **{investor_name}**.")

    # Sealed status: investors may see THAT bids exist, never their values.
    active = [b for b in mp.get_bids(listing["listing_id"]) if b["status"] == "active"]
    st.divider()
    if active:
        st.info(f"🔒 {len(active)} sealed bid(s) on this listing — values hidden until the auction runs.")
        if st.button("🔓 Run auction & reveal", type="primary", use_container_width=True):
            mp.run_auction(listing["listing_id"])  # lowest qualifying rate wins
            st.rerun()
    else:
        st.warning("No qualifying bids yet. Place at least one bid at/above the floor to run the auction.")


def investor_browse():
    """Stage 1: scannable grid of all OPEN listings, openable into the detail view."""
    st.write(
        "Fund a receivable today and collect the installments over the next three years. "
        "You bid the **discount rate you require**; the lowest qualifying bid wins."
    )

    listings = mp.get_open_listings()
    if not listings:
        st.info("No open listings right now. The club side can list new receivables, "
                "or reset the demo from the sidebar.")
        return

    # Let the user sort the board.
    sort = st.selectbox("Sort by", [
        "Rating (best first)", "Rating (worst first)",
        "Face value (high → low)", "Face value (low → high)",
    ])
    if sort == "Rating (best first)":
        listings.sort(key=lambda l: RATING_ORDER[l["rating"]])
    elif sort == "Rating (worst first)":
        listings.sort(key=lambda l: RATING_ORDER[l["rating"]], reverse=True)
    elif sort == "Face value (high → low)":
        listings.sort(key=lambda l: l["face_value_eur"], reverse=True)
    else:
        listings.sort(key=lambda l: l["face_value_eur"])

    # Header + one row per listing.
    cols_spec = [3, 2, 1.3, 1.2, 1.6]
    h = st.columns(cols_spec)
    for col, label in zip(h, ["Paying club", "Face value", "Rating", "Floor", ""]):
        col.caption(label)

    for l in listings:
        emoji, colour = RATING_STYLE[l["rating"]]
        row = st.columns(cols_spec, vertical_alignment="center")
        row[0].markdown(f"**{l['paying_club']}**  \n:gray[{l['paying_club_league']}]")
        row[1].markdown(_eur(l["face_value_eur"]))
        row[2].markdown(f"{emoji} :{colour}[**{l['rating']}**]")
        row[3].markdown(f"{l['floor_rate_pct']:.1f}%")
        if row[4].button("View & bid", key=f"view_{l['listing_id']}", use_container_width=True):
            st.session_state["inv_selected"] = l["listing_id"]
            st.rerun()


def investor_view():
    st.subheader("Investor — fund a receivable, collect the installments later")

    selected = st.session_state.get("inv_selected")
    if selected:
        listing = mp.get_listing(selected)
        if listing is None:  # e.g. demo was reset while viewing
            st.session_state.pop("inv_selected", None)
            st.rerun()
        investor_detail(listing)
    else:
        investor_browse()


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
