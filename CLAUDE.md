# CLAUDE.md — RECEIV MVP

## What this project is
RECEIV is a marketplace MVP for European football transfer receivables, built for a
FinTech course assignment (build the MVP of our business model). It converts illiquid,
deferred transfer payments into rated, tradeable claims by (1) scoring the paying club's
credit risk and (2) running a sealed-bid auction that matches club listings with
institutional investors. RECEIV is a pure marketplace intermediary — it never holds
receivables on its balance sheet and carries no credit risk.

This is an academic MVP, not production software. The goal is a working, demonstrable
end-to-end flow that maps the business plan's key features to code, plus a clean repo
and git history. It is graded on: features actually working, code being runnable with
zero extra setup, AI instructions matching the architecture, and git collaboration history.

## Architecture — three layers from the business plan for now
1. Contract parsing (Layer 1) — STUBBED / NOT IMPLEMENTED. In production: a PDF transfer
   agreement → LayoutLMv3 + an LLM → structured JSON (parties, total fee, installment
   schedule, add-ons, jurisdiction). For this MVP we start from that structured object
   directly, via fixtures or a manual-entry form. Keep a clean function boundary so a
   real parser could drop in later. This is deliberately out of scope and shown as
   "future work" in the demo.
2. Credit risk model (Layer 2) — CORE, IMPLEMENTED. Club features → XGBoost → probability
   of default → A–E rating → recommended minimum discount rate (the floor). Must expose
   feature importances so each rating is explainable. A rules-based hard filter runs
   BEFORE the model and blocks structurally distressed clubs from investment-grade ratings.
3. Marketplace + sealed-bid auction (Layer 3) — CORE, IMPLEMENTED, HIGHEST PRIORITY.
   A genuine two-sided flow:
   - Club side: provide/select a receivable → see the auto-generated rating + floor rate
     → list it on the marketplace.
   - Investor side: browse live listings → open one → see the credit breakdown and feature
     importances → place a sealed bid (the discount rate they require).
   - Matching: the lowest bid above the floor wins → a settlement view shows the accepted
     rate, the cash the club receives, and RECEIV's 1.0–1.5% transaction fee.

The two-sided flow is the whole point. A static dashboard of numbers is a FAILURE MODE:
the investor must be able to actually evaluate and fund a listing, and the club must see
the result.

## Tech stack — keep it simple, all local, no external services
- Python 3.12
- Streamlit for the UI (pure Python; no JavaScript or HTML build step)
- XGBoost for the credit model; pandas / numpy / scikit-learn as needed
- SQLite (Python's built-in sqlite3) for shared state — listings and bids — so both
  marketplace sides read the same data
- NO external API calls, NO API keys, NO network dependencies. Everything runs offline
  on synthetic data.

## Data
All data is synthetic and clearly labelled as such. A generator script creates plausible
clubs with the business plan's feature set (wage-to-revenue ratio, operating cash flow,
debt-to-asset ratio, regulatory headroom, UEFA coefficient trend, league-position trend,
broadcast-revenue share, historical payment timeliness) plus a default label, with
defaults correlated to weak fundamentals so the model learns a real pattern.

## Build order
1. Synthetic data generator
2. Credit model (train + score → rating + floor + feature importances + hard filter)
3. SQLite store
4. Streamlit app (club view, investor view, matching + settlement)
5. README, a flowchart of implemented-vs-stubbed features, polish

## Conventions
- Favour clarity over cleverness — this code gets read and explained on camera. Comment
  the non-obvious logic, especially the scoring and the auction matching.
- Small, single-purpose modules with clean boundaries between the three layers.
- One-command run, documented in the README (target: `streamlit run app.py` after
  `pip install -r requirements.txt`).
- Don't over-engineer. No extra frameworks, services, or abstractions beyond what the
  MVP needs.

## What NOT to do
- Do not add external API calls, cloud services, API keys, or anything needing the network.
- Do not implement real contract parsing — keep Layer 1 stubbed behind a clean interface.
- Do not invent business-plan facts: the fee is 1.0–1.5%, ratings are A–E, breakeven is
  ~10 deals/year. Ask if unsure rather than guessing.
- Do not build the investor side as a read-only dashboard — bidding and matching must work.
