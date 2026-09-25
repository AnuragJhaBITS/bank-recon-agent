# Bank Reconciliation Agent

An AI-powered bank reconciliation engine that matches bank-statement transactions to general-ledger entries using a three-layer approach: deterministic exact matching, fuzzy scoring, and LLM-based exception classification.

## Architecture

```
Bank Statement (500+ txns)       General Ledger (500+ entries)
         │                                │
         └──────────┬─────────────────────┘
                    ▼
        ┌───────────────────────┐
        │  Layer 1: Exact Match │  amount + reference
        │  (deterministic)      │  → ~60% matched
        └──────────┬────────────┘
                   ▼
        ┌───────────────────────┐
        │  Layer 2: Fuzzy Match │  amount proximity (40%)
        │  (scoring engine)     │  + date window (30%)
        │                       │  + description sim. (30%)
        │                       │  → ~25% matched
        └──────────┬────────────┘
                   ▼
        ┌───────────────────────┐
        │  Layer 3: LLM Agent   │  classifies remaining
        │  (OpenAI gpt-4o-mini) │  exceptions, suggests
        │                       │  correcting entries
        └──────────┬────────────┘
                   ▼
        ┌───────────────────────┐
        │  REST API + Dashboard │  FastAPI backend
        │                       │  HTML/JS frontend
        └───────────────────────┘
```

### Design principles

1. **Numbers never come from the LLM.** Every amount, date, and metric originates from a SQL query. The LLM receives read-only row data and returns only a classification label and free-text explanation. This eliminates hallucinated figures.

2. **Known ground truth.** The synthetic data generator plants errors deliberately (timing offsets, amount typos, description mismatches, missing entries, duplicates) and records the true pairing. This lets us compute real precision and recall, not just "it looks right."

3. **Graceful degradation.** If no OpenAI key is configured, Layer 3 falls back to a rule-based heuristic. Layers 1 and 2 need no external API.

## Error types planted in synthetic data

| Error | % of data | What happens |
|---|---|---|
| Clean | 60% | Identical amount, date, reference, description |
| Timing | 10% | GL date shifted ±1–4 days from bank date |
| Amount typo | 8% | GL amount differs by a small delta or transposed digits |
| Description mismatch | 7% | Bank uses abbreviation, GL uses full vendor name |
| Missing in GL | 5% | Bank transaction exists, no GL counterpart |
| Missing in bank | 5% | GL entry exists, no bank counterpart |
| Duplicate | 5% | Same bank transaction appears twice |

## Setup

```bash
# clone and enter
git clone <your-repo-url>
cd bank-recon-agent

# create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# configure (optional — Layer 3 falls back to heuristics without a key)
cp .env.example .env
# edit .env and add your OpenAI API key

# run
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

1. Click **Generate test data** to seed 500 transactions with planted errors.
2. Click **Run reconciliation** to execute all three layers.
3. Inspect matches and exceptions in the dashboard tabs.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate-data?n=500` | Seed fresh synthetic data |
| `POST` | `/reconcile` | Run three-layer matching |
| `GET` | `/matches?match_type=exact` | Accepted matches (filterable) |
| `GET` | `/exceptions` | Unmatched rows from both sides |
| `GET` | `/stats` | Summary statistics and accuracy |

## Tech stack

- **Backend:** Python, FastAPI, SQLAlchemy, SQLite (swap to PostgreSQL via `DATABASE_URL`)
- **Matching:** thefuzz (Levenshtein), custom scoring weights
- **LLM:** OpenAI API (gpt-4o-mini), structured JSON output
- **Frontend:** Vanilla HTML/CSS/JS — no build step

## Switching to PostgreSQL

```bash
# in .env
DATABASE_URL=postgresql://user:pass@localhost:5432/recon
pip install psycopg2-binary
```


