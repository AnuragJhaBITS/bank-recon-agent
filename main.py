"""
FastAPI application — REST API for the reconciliation engine.

Endpoints:
  POST /generate-data       seed fresh synthetic data
  POST /reconcile           run the three-layer matcher
  GET  /matches             all accepted matches with details
  GET  /exceptions          unmatched rows from both sides
  GET  /stats               summary statistics and accuracy
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import func

from models import (
    SessionLocal, BankTransaction, GLEntry, Match,
    MatchType, init_db
)
from seed_data import generate
from matcher import run_reconciliation

app = FastAPI(title="Bank Reconciliation Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

init_db()


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────

@app.post("/generate-data")
def generate_data(n: int = 500):
    result = generate(n)
    return {"status": "ok", "summary": result}


# ──────────────────────────────────────────────
# Reconciliation
# ──────────────────────────────────────────────

@app.post("/reconcile")
def reconcile():
    result = run_reconciliation()
    return {"status": "ok", "result": result}


# ──────────────────────────────────────────────
# Matches
# ──────────────────────────────────────────────

@app.get("/matches")
def get_matches(match_type: str | None = None):
    session = SessionLocal()

    query = session.query(Match)
    if match_type:
        query = query.filter(Match.match_type == match_type)

    matches = query.order_by(Match.id).all()

    results = []
    for m in matches:
        bank = session.query(BankTransaction).get(m.bank_id)
        gl = session.query(GLEntry).get(m.gl_id)
        results.append({
            "match_id": m.id,
            "match_type": m.match_type.value,
            "confidence": m.confidence,
            "explanation": m.explanation,
            "bank": {
                "id": bank.id,
                "date": str(bank.date),
                "description": bank.description,
                "amount": bank.amount,
                "reference": bank.reference,
            } if bank else None,
            "gl": {
                "id": gl.id,
                "date": str(gl.date),
                "description": gl.description,
                "amount": gl.amount,
                "reference": gl.reference,
                "account_code": gl.account_code,
            } if gl else None,
        })

    session.close()
    return results


# ──────────────────────────────────────────────
# Exceptions (unmatched rows)
# ──────────────────────────────────────────────

@app.get("/exceptions")
def get_exceptions():
    session = SessionLocal()

    matched_bank_ids = {
        m.bank_id for m in session.query(Match.bank_id).all()
    }
    matched_gl_ids = {
        m.gl_id for m in session.query(Match.gl_id).all()
    }

    unmatched_bank = (
        session.query(BankTransaction)
        .filter(BankTransaction.id.notin_(matched_bank_ids))
        .all()
    )
    unmatched_gl = (
        session.query(GLEntry)
        .filter(GLEntry.id.notin_(matched_gl_ids))
        .all()
    )

    session.close()

    return {
        "unmatched_bank": [
            {
                "id": b.id,
                "date": str(b.date),
                "description": b.description,
                "amount": b.amount,
                "reference": b.reference,
                "planted_error": b.planted_error.value,
            }
            for b in unmatched_bank
        ],
        "unmatched_gl": [
            {
                "id": g.id,
                "date": str(g.date),
                "description": g.description,
                "amount": g.amount,
                "reference": g.reference,
                "account_code": g.account_code,
                "planted_error": g.planted_error.value,
            }
            for g in unmatched_gl
        ],
    }


# ──────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────

@app.get("/stats")
def get_stats():
    session = SessionLocal()

    total_bank = session.query(BankTransaction).count()
    total_gl = session.query(GLEntry).count()
    total_matches = session.query(Match).count()

    by_type = {}
    for mt in MatchType:
        by_type[mt.value] = (
            session.query(Match).filter(Match.match_type == mt).count()
        )

    matched_bank_ids = {
        m.bank_id for m in session.query(Match.bank_id).all()
    }
    matched_gl_ids = {
        m.gl_id for m in session.query(Match.gl_id).all()
    }

    unmatched_bank = total_bank - len(matched_bank_ids)
    unmatched_gl = total_gl - len(matched_gl_ids)

    # accuracy against planted ground truth
    matches = session.query(Match).all()
    correct = 0
    incorrect = 0
    for m in matches:
        bank = session.query(BankTransaction).get(m.bank_id)
        gl = session.query(GLEntry).get(m.gl_id)
        if bank and gl and bank.true_pair_id and gl.true_pair_id:
            if bank.true_pair_id == gl.true_pair_id:
                correct += 1
            else:
                incorrect += 1

    session.close()

    return {
        "total_bank_transactions": total_bank,
        "total_gl_entries": total_gl,
        "total_matches": total_matches,
        "matches_by_type": by_type,
        "unmatched_bank": unmatched_bank,
        "unmatched_gl": unmatched_gl,
        "auto_match_rate_pct": round(
            len(matched_bank_ids) / max(total_bank, 1) * 100, 1
        ),
        "correct_matches": correct,
        "incorrect_matches": incorrect,
        "precision_pct": round(
            correct / max(correct + incorrect, 1) * 100, 1
        ),
    }
