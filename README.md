# RECEIV — Transfer Receivables Marketplace

> **Academic MVP** for a FinTech course assignment. Converts illiquid, deferred football
> transfer fees into rated, tradeable claims via a sealed-bid auction.
> All club names and financial figures are **illustrative / synthetic**.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app seeds 20 demo listings automatically on first run. No API keys, no network, no
external services, everything runs fully offline.

---

## What RECEIV does

European football clubs often receive transfer fees in annual installments spread over
three years. RECEIV is a **two-sided marketplace** that:

1. **Scores the paying club's credit risk** (XGBoost → probability of default → A–E rating → recommended discount-rate floor).
2. **Lists the receivable** so institutional investors can browse and evaluate it.
3. **Runs a sealed-bid auction**: investors bid the discount rate they require; the
   lowest qualifying bid wins. The club receives cash today; the investor collects the
   installments over three years.

RECEIV earns a **1.25% transaction fee** on the face value of each matched deal.

---

## Architecture — three layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Contract parser              [STUBBED / FUTURE]  │
│  PDF transfer agreement → structured JSON (parties, fees,   │
│  installment schedule). In this MVP: manual-entry form or   │
│  pre-seeded synthetic fixtures stand in for the parser.     │
│  Production target: LayoutLMv3 + LLM extraction pipeline.  │
└────────────────────────────┬────────────────────────────────┘
                             │ structured receivable object
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2 — Credit risk model            [IMPLEMENTED ✓]     │
│  9 club features → XGBoost → P(default) → A–E rating        │
│  → recommended floor rate (€STR + credit spread)           │
│  → SHAP-based feature-importance explanation                │
│  Hard-filter rules block structurally distressed clubs       │
│  from investment-grade ratings before the model runs.       │
│  Files: generate_data.py · train_model.py · credit_model.py │
└────────────────────────────┬────────────────────────────────┘
                             │ credit decision (rating, floor, drivers)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — Marketplace + sealed-bid auction [IMPLEMENTED ✓] │
│  SQLite store (listings · bids · settlements)               │
│  Club side  : list a receivable → see the credit decision   │
│  Investor side: browse listings → bid discount rate (sealed)│
│  Matching   : lowest bid above the floor wins               │
│  Settlement : present value of installments − RECEIV fee    │
│  Files: marketplace.py · app.py (Streamlit UI)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature status

| Feature | Status | File(s) |
|---|---|---|
| Synthetic data generation | ✅ Implemented | `generate_data.py` |
| XGBoost credit model (train + score) | ✅ Implemented | `train_model.py`, `credit_model.py` |
| A–E rating + discount-rate floor | ✅ Implemented | `credit_model.py` |
| SHAP-based explainability (top drivers) | ✅ Implemented | `credit_model.py` |
| Hard-filter rules (structural red flags) | ✅ Implemented | `credit_model.py` |
| SQLite marketplace store | ✅ Implemented | `marketplace.py` |
| Club view — score + list a receivable | ✅ Implemented | `app.py` |
| Investor view — browse + sealed bid | ✅ Implemented | `app.py` |
| Sealed-bid auction + settlement | ✅ Implemented | `marketplace.py`, `app.py` |
| PDF contract parser (Layer 1) | 🔲 Stubbed | Manual form in `app.py` |
| Live ECB rate feed (€STR) | 🔲 Stubbed | Hard-coded 2.4% in `credit_model.py` |

---

## End-to-end demo flow

### Club side
1. Open the app and select **Club** in the sidebar.
2. **View a demo listing** to see a pre-scored receivable with its rating and risk drivers.
3. Or switch to **Score a new receivable**, fill in the paying club's financials, and hit
   *Score receivable* — you get an instant A–E rating with SHAP-explained drivers.
4. Click **List on marketplace** — the receivable is now live for investors.

### Investor side
1. Switch role to **Investor** in the sidebar.
2. Browse the open listings board (sortable by rating or face value).
3. Click **View & bid** on any listing to see the full credit breakdown.
4. Enter your name and required discount rate, then **Submit sealed bid** (bids are hidden
   from other investors until the auction is run).
5. Once at least one qualifying bid is in, click **Run auction & reveal** — the lowest
   bid above the floor wins and the settlement economics are shown.

---

## Repository layout

```
fintech-receiv/
├── app.py                   # Streamlit UI (Layer 3 front-end)
├── marketplace.py           # SQLite store + auction logic (Layer 3)
├── credit_model.py          # Scoring interface + hard filter (Layer 2)
├── train_model.py           # Train + save the XGBoost model (Layer 2)
├── generate_data.py         # Synthetic data generator (Layer 1 → 2 feed)
├── requirements.txt
├── CLAUDE.md                # AI agent instructions (Claude Code)
├── .claude/
│   └── settings.json        # Claude Code project settings
├── data/
│   ├── synthetic_training.csv   # 1 500 labelled club-seasons (SYNTHETIC)
│   └── synthetic_listings.csv   # 20 current marketplace listings (SYNTHETIC)
└── models/
    ├── credit_model.json        # Trained XGBoost booster (committed artifact)
    └── credit_model_meta.json   # Feature list, rating bands, spreads
```

---

## Regenerating data and model

The committed `data/` CSVs and `models/` artifacts are ready to use. To regenerate:

```bash
python generate_data.py   # re-creates data/synthetic_training.csv + synthetic_listings.csv
python train_model.py     # re-trains the model and writes models/
```

---

## Credit model — key design choices

| Choice | Rationale |
|---|---|
| XGBoost (n=250 trees, max_depth=3) | Calibrated probabilities needed — deeper trees overfit and inflate PDs, distorting the floor rate |
| No `scale_pos_weight` | Reweighting inflates predicted PDs; since the floor is derived directly from PD, the model must be calibrated, not just rank-ordered |
| SHAP contributions (built-in XGBoost) | Per-prediction explanation without a separate library |
| Hard-filter rules run before the model | Structural red flags (negative cash flow, breached FSR cap) should block investment grade regardless of the model score |
| €STR hard-coded at 2.4% | Offline MVP; in production this is pulled live from the ECB |

---

## Tech stack

| Component | Library / tool |
|---|---|
| UI | Streamlit |
| Credit model | XGBoost |
| Data wrangling | pandas, numpy |
| Shared state | SQLite (Python built-in `sqlite3`) |
| Explainability | XGBoost native SHAP contributions |
| Leakage guard | scikit-learn train/test split |

---

## AI agent orchestration

This MVP was built with **Claude Code** (Anthropic) as the primary coding agent.

Agent instructions live in [`CLAUDE.md`](CLAUDE.md) (project-level) and
[`.claude/settings.json`](.claude/settings.json) (tool permissions). The instructions
encode:
- The three-layer architecture and which layers are in scope vs stubbed.
- The exact business-plan constants (fee = 1.25%, ratings A–E, break-even ~10 deals/year).
- Hard constraints: no external API calls, no network, one-command run, SQLite only.
- Coding conventions: clarity over cleverness, comments on non-obvious logic only,
  single-purpose modules with clean layer boundaries.

Claude Code was chosen because it understands the full codebase across files in a single
context window, can make multi-file coordinated edits, and produces verifiable diffs —
well suited to an MVP that needs coherent architecture across five Python modules.

---

## Scaling prerequisites and risks

| Concern | Notes |
|---|---|
| Contract parser (Layer 1) | Requires fine-tuned LayoutLMv3 + LLM; the MVP's clean function boundary allows drop-in replacement |
| Model retraining | Synthetic data must be replaced with real default histories from UEFA / club filings |
| Live rate feed | €STR hard-coded; production needs an ECB API or Bloomberg feed |
| SQLite → production DB | SQLite does not support concurrent writes; replace with PostgreSQL for multi-user load |
| Regulatory (MiFID II / AIFMD) | Institutional investors and rated instruments trigger licensing requirements |

---

*All club names, transfer values, and financial figures in this MVP are entirely
fictional and generated programmatically. No real club data was used.*
