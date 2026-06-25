"""
marketplace.py — RECEIV MVP, Layer 3 (store + auction logic).

This is the shared persistence + marketplace layer the Streamlit app (step 4) sits on. It
keeps all marketplace state in a single local SQLite file (Python's built-in sqlite3) so
the club side and the investor side read/write the same data.

Responsibilities:
  - Define the schema (listings, bids, settlements).
  - Seed listings from data/synthetic_listings.csv, scoring each paying club through the
    Layer-2 credit model (credit_model.score_club) and storing the full credit decision.
  - Record sealed bids (rejecting any that underprice the risk below the rating floor).
  - Run the sealed-bid auction: the LOWEST qualifying rate wins, then compute the
    settlement economics (present value, RECEIV fee, cash to the club).

We import and call credit_model — we never re-implement scoring here (clean layer boundary).
All underlying data is SYNTHETIC (see generate_data.py).
"""

import json
import os
import sqlite3

from credit_model import FEATURES, load_model, score_club

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DB_PATH = "receiv.db"
LISTINGS_CSV = os.path.join("data", "synthetic_listings.csv")

# RECEIV's transaction fee, as a % of face value. The business plan allows 1.0-1.5%; we use
# 1.25% (the midpoint). Exposed as a named constant so the demo/UI can reference it.
RECEIV_FEE_PCT = 1.25

# Listing columns copied verbatim from the CSV (identity + deal structure).
_DEAL_COLS = [
    "player_name", "selling_club", "paying_club", "paying_club_league", "face_value_eur",
    "installment_1_eur", "installment_2_eur", "installment_3_eur",
]


# --------------------------------------------------------------------------------------
# Connection helper
# --------------------------------------------------------------------------------------
def get_conn(db_path=DB_PATH):
    """Open a SQLite connection with dict-like rows and foreign keys enforced."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------
def init_db(db_path=DB_PATH):
    """Create the tables if they don't already exist."""
    import os

    # Schema-upgrade guard: if the DB exists but predates the player_name column, wipe it.
    # Must explicitly close the connection before os.remove — on Windows, SQLite holds a
    # file lock until close() is called, so the delete would otherwise silently fail.
    if os.path.exists(db_path):
        _chk = get_conn(db_path)
        try:
            existing = [r[1] for r in _chk.execute("PRAGMA table_info(listings)").fetchall()]
        finally:
            _chk.close()
        if existing and "player_name" not in existing:
            os.remove(db_path)

    # The nine credit-feature columns are added dynamically so the schema always matches
    # whatever credit_model.FEATURES declares (single source of truth).
    feature_cols = ",\n            ".join(f"{f} REAL" for f in FEATURES)

    with get_conn(db_path) as conn:
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS listings (
            listing_id          TEXT PRIMARY KEY,
            player_name         TEXT,
            selling_club        TEXT,
            paying_club         TEXT,
            paying_club_league  TEXT,
            face_value_eur      INTEGER,
            installment_1_eur   INTEGER,
            installment_2_eur   INTEGER,
            installment_3_eur   INTEGER,
            {feature_cols},
            probability_of_default REAL,
            rating              TEXT,
            model_rating        TEXT,
            floor_rate_pct      REAL,
            hard_filter_triggered INTEGER,
            hard_filter_reasons TEXT,          -- JSON list of strings
            status              TEXT DEFAULT 'open',   -- 'open' | 'matched' | 'settled'
            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS bids (
            bid_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id    TEXT NOT NULL,
            investor_name TEXT NOT NULL,
            bid_rate_pct  REAL NOT NULL,
            status        TEXT DEFAULT 'active',  -- 'active'|'winning'|'lost'|'rejected'
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (listing_id) REFERENCES listings (listing_id)
        )
        """)

        # Settlement economics for a matched listing (one row per matched listing).
        conn.execute("""
        CREATE TABLE IF NOT EXISTS settlements (
            listing_id        TEXT PRIMARY KEY,
            winning_bid_id    INTEGER,
            accepted_rate_pct REAL,
            present_value_eur REAL,
            receiv_fee_eur    REAL,
            cash_to_club_eur  REAL,
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (listing_id) REFERENCES listings (listing_id),
            FOREIGN KEY (winning_bid_id) REFERENCES bids (bid_id)
        )
        """)



def reset_db(db_path=DB_PATH):
    """Clear all marketplace state so the demo can start from a clean, known baseline."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM settlements")
        conn.execute("DELETE FROM bids")
        conn.execute("DELETE FROM listings")
        # Reset the bid_id autoincrement counter (best-effort; table may not exist on a
        # brand-new db, hence the guard).
        conn.execute("DELETE FROM sqlite_sequence WHERE name='bids'")


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------
def seed_listings(db_path=DB_PATH, reset=False, csv_path=LISTINGS_CSV):
    """
    Load the synthetic listings, score each paying club through the credit model, and insert
    them as 'open' listings with their full credit decision.

    Idempotent: listing_id is the primary key and we INSERT OR IGNORE, so re-running never
    duplicates rows. Pass reset=True to wipe the marketplace (listings, bids, settlements)
    and reseed from scratch.
    """
    import pandas as pd  # local import keeps pandas optional for the rest of the module

    init_db(db_path)
    if reset:
        reset_db(db_path)

    df = pd.read_csv(csv_path)
    model = load_model()  # load once, reuse for every club

    inserted = 0
    with get_conn(db_path) as conn:
        for _, row in df.iterrows():
            features = {f: float(row[f]) for f in FEATURES}
            decision = score_club(features, model=model)

            cols = (
                ["listing_id"]
                + _DEAL_COLS
                + FEATURES
                + ["probability_of_default", "rating", "model_rating",
                   "floor_rate_pct", "hard_filter_triggered", "hard_filter_reasons",
                   "status"]
            )
            values = (
                [row["listing_id"]]
                + [row[c] for c in _DEAL_COLS]
                + [features[f] for f in FEATURES]
                + [
                    decision["probability_of_default"],
                    decision["rating"],
                    decision["model_rating"],
                    decision["floor_rate_pct"],
                    int(decision["hard_filter_triggered"]),
                    json.dumps(decision["hard_filter_reasons"]),
                    "open",
                ]
            )
            placeholders = ", ".join("?" for _ in cols)
            cur = conn.execute(
                f"INSERT OR IGNORE INTO listings ({', '.join(cols)}) VALUES ({placeholders})",
                values,
            )
            inserted += cur.rowcount

    return inserted


# --------------------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------------------
def _listing_row_to_dict(row):
    """Convert a listings Row to a dict, parsing the JSON hard_filter_reasons back to a list."""
    if row is None:
        return None
    d = dict(row)
    d["hard_filter_triggered"] = bool(d["hard_filter_triggered"])
    d["hard_filter_reasons"] = json.loads(d["hard_filter_reasons"] or "[]")
    return d


def get_open_listings(db_path=DB_PATH):
    """All listings still available to bid on, as a list of dicts."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM listings WHERE status = 'open' ORDER BY listing_id"
        ).fetchall()
    return [_listing_row_to_dict(r) for r in rows]


def get_listing(listing_id, db_path=DB_PATH):
    """One listing by id, or None."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM listings WHERE listing_id = ?", (listing_id,)
        ).fetchone()
    return _listing_row_to_dict(row)


def get_bids(listing_id, db_path=DB_PATH):
    """
    All bids on a listing, as a list of dicts (oldest first).

    NB: bids are SEALED — this returns every bid regardless of status. Hiding bids from
    other investors before the match is the app's responsibility; the store just records them.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM bids WHERE listing_id = ? ORDER BY bid_id", (listing_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_settlement(listing_id, db_path=DB_PATH):
    """The settlement for a matched listing, or None."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM settlements WHERE listing_id = ?", (listing_id,)
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------------------
# Bidding
# --------------------------------------------------------------------------------------
def place_bid(listing_id, investor_name, bid_rate_pct, db_path=DB_PATH):
    """
    Record a sealed bid (the discount rate the investor requires).

    A bid BELOW the listing's floor_rate_pct underprices the credit risk and is stored with
    status 'rejected'. A bid at or above the floor is stored 'active' and competes in the
    auction. Returns the stored bid as a dict.
    """
    listing = get_listing(listing_id, db_path)
    if listing is None:
        raise ValueError(f"Unknown listing_id: {listing_id}")

    # Below the rating floor => rejected (does not adequately price the risk).
    status = "rejected" if bid_rate_pct < listing["floor_rate_pct"] else "active"

    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO bids (listing_id, investor_name, bid_rate_pct, status) "
            "VALUES (?, ?, ?, ?)",
            (listing_id, investor_name, float(bid_rate_pct), status),
        )
        bid_id = cur.lastrowid
        row = conn.execute("SELECT * FROM bids WHERE bid_id = ?", (bid_id,)).fetchone()
    return dict(row)


# --------------------------------------------------------------------------------------
# Auction + settlement
# --------------------------------------------------------------------------------------
def _present_value(installments, rate_pct):
    """
    Present value of the three annual installments discounted at the accepted rate.

        PV = inst_1/(1+r)^1 + inst_2/(1+r)^2 + inst_3/(1+r)^3,   r = rate_pct / 100

    A higher accepted rate => more discounting => lower PV => less cash for the club. This is
    the core economic trade-off the marketplace prices.
    """
    r = rate_pct / 100.0
    return sum(inst / (1 + r) ** year for year, inst in enumerate(installments, start=1))


def run_auction(listing_id, db_path=DB_PATH):
    """
    Resolve the sealed-bid auction for one listing.

    Among 'active' bids (those at/above the floor), the LOWEST rate wins — that is the
    cheapest financing for the club that still prices the risk adequately. The winning bid is
    marked 'winning', the other active bids 'lost' (rejected bids stay 'rejected'). The
    listing flips to 'matched' and the settlement economics are computed and persisted.

    Returns a settlement dict, or None if there were no qualifying bids.
    """
    listing = get_listing(listing_id, db_path)
    if listing is None:
        raise ValueError(f"Unknown listing_id: {listing_id}")

    with get_conn(db_path) as conn:
        # Active (qualifying) bids, cheapest first. Ties broken by earliest bid.
        active = conn.execute(
            "SELECT * FROM bids WHERE listing_id = ? AND status = 'active' "
            "ORDER BY bid_rate_pct ASC, bid_id ASC",
            (listing_id,),
        ).fetchall()

        if not active:
            return None  # no qualifying bids -> nothing to match

        winner = active[0]
        winner_id = winner["bid_id"]
        accepted_rate = winner["bid_rate_pct"]

        # Mark winner / losers (rejected bids are left untouched).
        conn.execute(
            "UPDATE bids SET status = 'lost' WHERE listing_id = ? AND status = 'active'",
            (listing_id,),
        )
        conn.execute("UPDATE bids SET status = 'winning' WHERE bid_id = ?", (winner_id,))

        # --- Settlement economics -----------------------------------------------------
        installments = [
            listing["installment_1_eur"],
            listing["installment_2_eur"],
            listing["installment_3_eur"],
        ]
        present_value = _present_value(installments, accepted_rate)
        receiv_fee = listing["face_value_eur"] * (RECEIV_FEE_PCT / 100.0)
        cash_to_club = present_value - receiv_fee

        conn.execute("UPDATE listings SET status = 'matched' WHERE listing_id = ?",
                     (listing_id,))

        conn.execute(
            "INSERT OR REPLACE INTO settlements "
            "(listing_id, winning_bid_id, accepted_rate_pct, present_value_eur, "
            " receiv_fee_eur, cash_to_club_eur) VALUES (?, ?, ?, ?, ?, ?)",
            (listing_id, winner_id, accepted_rate,
             round(present_value, 2), round(receiv_fee, 2), round(cash_to_club, 2)),
        )

    return {
        "listing_id": listing_id,
        "winning_bid_id": winner_id,
        "accepted_rate_pct": accepted_rate,
        "present_value_eur": round(present_value, 2),
        "receiv_fee_eur": round(receiv_fee, 2),
        "cash_to_club_eur": round(cash_to_club, 2),
    }


# --------------------------------------------------------------------------------------
# End-to-end smoke test
# --------------------------------------------------------------------------------------
def _eur(x):
    return f"€{x:,.2f}"


def main():
    # Use a throwaway db file so the smoke test never clobbers a real demo state.
    demo_db = "receiv_demo.db"
    if os.path.exists(demo_db):
        os.remove(demo_db)

    print("=" * 80)
    print("RECEIV marketplace — end-to-end smoke test (SYNTHETIC data)")
    print("=" * 80)

    init_db(demo_db)
    n = seed_listings(demo_db, reset=True)
    print(f"\nSeeded {n} open listings.")

    # Show a couple of open listings with their rating + floor.
    open_listings = get_open_listings(demo_db)
    print("\nSample open listings:")
    for lst in open_listings[:3]:
        print(f"  {lst['listing_id']}  {lst['paying_club']:<22} "
              f"rating {lst['rating']}  floor {lst['floor_rate_pct']:.1f}%  "
              f"face {_eur(lst['face_value_eur'])}")

    # Pick one listing to auction. Choose a non-hard-filtered one so the floor is moderate.
    target = next(l for l in open_listings if not l["hard_filter_triggered"])
    floor = target["floor_rate_pct"]
    print(f"\nBidding on {target['listing_id']} ({target['paying_club']}) — "
          f"rating {target['rating']}, floor {floor:.1f}%")

    # Three sealed bids: one below the floor (rejected) and two above (active).
    bids_to_place = [
        ("Cautious Capital",  floor - 1.0),   # below floor -> rejected
        ("Mid Pension Fund",  floor + 2.5),   # qualifies
        ("Sharp Credit Fund", floor + 1.0),   # qualifies AND cheaper -> should win
    ]
    print("\nPlacing bids:")
    for name, rate in bids_to_place:
        b = place_bid(target["listing_id"], name, rate, demo_db)
        print(f"  {name:<18} {rate:5.2f}%  -> {b['status']}")

    # Resolve the auction.
    settlement = run_auction(target["listing_id"], demo_db)

    print("\n--- Auction settlement ---")
    print(f"  Accepted rate : {settlement['accepted_rate_pct']:.2f}%  (lowest qualifying bid)")
    print(f"  Present value : {_eur(settlement['present_value_eur'])}")
    print(f"  RECEIV fee    : {_eur(settlement['receiv_fee_eur'])}  "
          f"({RECEIV_FEE_PCT}% of {_eur(target['face_value_eur'])} face value)")
    print(f"  Cash to club  : {_eur(settlement['cash_to_club_eur'])}")
    print(f"  Face value    : {_eur(target['face_value_eur'])}  "
          f"(installments {_eur(target['installment_1_eur'])} / "
          f"{_eur(target['installment_2_eur'])} / {_eur(target['installment_3_eur'])})")

    # Confirm state transitions.
    print("\n--- State after auction ---")
    matched = get_listing(target["listing_id"], demo_db)
    print(f"  Listing status: {matched['status']}")
    print("  Bid statuses:")
    for b in get_bids(target["listing_id"], demo_db):
        print(f"    bid {b['bid_id']}  {b['investor_name']:<18} "
              f"{b['bid_rate_pct']:5.2f}%  -> {b['status']}")

    os.remove(demo_db)


if __name__ == "__main__":
    main()
