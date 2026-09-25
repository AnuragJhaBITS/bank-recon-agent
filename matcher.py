"""
Three-layer reconciliation engine.

Layer 1 – Exact:   match on (amount AND reference).  Fast, deterministic.
Layer 2 – Fuzzy:   for remaining rows, score on amount proximity, date
                   window, and description similarity.  Accept if the
                   weighted score exceeds a threshold.
Layer 3 – LLM:     hand the leftover exceptions to an OpenAI agent that
                   classifies each one (timing / duplicate / missing /
                   amount discrepancy) and writes a human-readable
                   explanation.  The LLM never invents numbers.

All data flows through SQL.  The LLM receives read-only context and
returns only a classification label + free-text explanation.
"""

from datetime import timedelta
from sqlalchemy import and_
from thefuzz import fuzz

from models import (
    SessionLocal, BankTransaction, GLEntry, Match, MatchType
)
from config import (
    FUZZY_AMOUNT_TOLERANCE,
    FUZZY_DATE_WINDOW_DAYS,
    FUZZY_DESCRIPTION_THRESHOLD,
    FUZZY_COMBINED_SCORE_CUTOFF,
)
from agent import classify_exceptions


# ──────────────────────────────────────────────
# Layer 1: exact match on amount + reference
# ──────────────────────────────────────────────

def _exact_pass(session) -> int:
    """Match bank rows to GL rows where amount AND reference are identical."""
    matched_bank_ids = {
        m.bank_id for m in session.query(Match.bank_id).all()
    }
    matched_gl_ids = {
        m.gl_id for m in session.query(Match.gl_id).all()
    }

    bank_rows = (
        session.query(BankTransaction)
        .filter(BankTransaction.id.notin_(matched_bank_ids))
        .all()
    )
    gl_rows = (
        session.query(GLEntry)
        .filter(GLEntry.id.notin_(matched_gl_ids))
        .all()
    )

    # index GL by (amount, reference) for O(1) lookup
    gl_index: dict[tuple, list[GLEntry]] = {}
    for gl in gl_rows:
        key = (round(gl.amount, 2), (gl.reference or "").strip())
        gl_index.setdefault(key, []).append(gl)

    count = 0
    for bank in bank_rows:
        key = (round(bank.amount, 2), (bank.reference or "").strip())
        candidates = gl_index.get(key, [])
        if candidates:
            gl = candidates.pop(0)
            session.add(Match(
                bank_id=bank.id,
                gl_id=gl.id,
                match_type=MatchType.EXACT,
                confidence=1.0,
                explanation="Exact match on amount and reference number.",
            ))
            count += 1

    session.commit()
    return count


# ──────────────────────────────────────────────
# Layer 2: fuzzy scoring
# ──────────────────────────────────────────────

def _score_pair(bank: BankTransaction, gl: GLEntry) -> tuple[float, str]:
    """
    Return (score, explanation) for a candidate bank-GL pair.

    Weights:
      amount proximity  40%
      date proximity    30%
      description sim.  30%
    """
    reasons = []

    # --- amount ---
    if bank.amount == 0 and gl.amount == 0:
        amt_score = 1.0
    elif bank.amount == 0 or gl.amount == 0:
        amt_score = 0.0
    else:
        pct_diff = abs(bank.amount - gl.amount) / max(
            abs(bank.amount), abs(gl.amount)
        )
        if pct_diff <= FUZZY_AMOUNT_TOLERANCE:
            amt_score = 1.0 - (pct_diff / FUZZY_AMOUNT_TOLERANCE)
        else:
            amt_score = 0.0
    if amt_score < 1.0:
        reasons.append(
            f"Amount differs: bank {bank.amount:,.2f} vs GL {gl.amount:,.2f}"
        )

    # --- date ---
    day_diff = abs((bank.date - gl.date).days)
    if day_diff <= FUZZY_DATE_WINDOW_DAYS:
        date_score = 1.0 - (day_diff / FUZZY_DATE_WINDOW_DAYS)
    else:
        date_score = 0.0
    if day_diff > 0:
        reasons.append(f"Date offset: {day_diff} day(s)")

    # --- description ---
    desc_score_raw = fuzz.token_sort_ratio(
        bank.description.lower(), gl.description.lower()
    )
    if desc_score_raw >= FUZZY_DESCRIPTION_THRESHOLD:
        desc_score = desc_score_raw / 100.0
    else:
        desc_score = 0.0
    if desc_score < 1.0:
        reasons.append(
            f"Description similarity: {desc_score_raw}% "
            f"('{bank.description}' vs '{gl.description}')"
        )

    combined = 0.40 * amt_score + 0.30 * date_score + 0.30 * desc_score
    explanation = "Fuzzy match. " + "; ".join(reasons) if reasons else "Fuzzy match."
    return combined, explanation


def _fuzzy_pass(session) -> int:
    """Score every unmatched bank×GL pair; accept best mutual matches."""
    matched_bank_ids = {
        m.bank_id for m in session.query(Match.bank_id).all()
    }
    matched_gl_ids = {
        m.gl_id for m in session.query(Match.gl_id).all()
    }

    bank_rows = (
        session.query(BankTransaction)
        .filter(BankTransaction.id.notin_(matched_bank_ids))
        .all()
    )
    gl_rows = (
        session.query(GLEntry)
        .filter(GLEntry.id.notin_(matched_gl_ids))
        .all()
    )

    # score every candidate pair
    scored: list[tuple[float, str, int, int]] = []
    for bank in bank_rows:
        for gl in gl_rows:
            score, explanation = _score_pair(bank, gl)
            if score >= FUZZY_COMBINED_SCORE_CUTOFF:
                scored.append((score, explanation, bank.id, gl.id))

    # greedy best-first assignment (no bank or GL used twice)
    scored.sort(key=lambda x: x[0], reverse=True)
    used_bank = set()
    used_gl = set()
    count = 0

    for score, explanation, bank_id, gl_id in scored:
        if bank_id in used_bank or gl_id in used_gl:
            continue
        session.add(Match(
            bank_id=bank_id,
            gl_id=gl_id,
            match_type=MatchType.FUZZY,
            confidence=round(score, 4),
            explanation=explanation,
        ))
        used_bank.add(bank_id)
        used_gl.add(gl_id)
        count += 1

    session.commit()
    return count


# ──────────────────────────────────────────────
# Layer 3: LLM exception handling
# ──────────────────────────────────────────────

def _llm_pass(session) -> dict:
    """
    Send remaining unmatched rows to the LLM agent.
    The agent classifies each exception and writes an explanation.
    It does NOT produce any numbers — those come from the rows themselves.
    """
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

    if not unmatched_bank and not unmatched_gl:
        return {"llm_matches": 0, "exceptions_flagged": 0}

    results = classify_exceptions(unmatched_bank, unmatched_gl)

    llm_matches = 0
    exceptions_flagged = 0

    for r in results:
        if r["action"] == "match":
            session.add(Match(
                bank_id=r["bank_id"],
                gl_id=r["gl_id"],
                match_type=MatchType.LLM,
                confidence=r.get("confidence", 0.6),
                explanation=r["explanation"],
            ))
            llm_matches += 1
        else:
            exceptions_flagged += 1

    session.commit()
    return {
        "llm_matches": llm_matches,
        "exceptions_flagged": exceptions_flagged,
    }


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

def run_reconciliation() -> dict:
    """Run all three layers in sequence and return summary stats."""
    session = SessionLocal()

    # clear previous matches
    session.query(Match).delete()
    session.commit()

    exact_count = _exact_pass(session)
    fuzzy_count = _fuzzy_pass(session)
    llm_result = _llm_pass(session)

    total_bank = session.query(BankTransaction).count()
    total_gl = session.query(GLEntry).count()
    total_matched = session.query(Match).count()

    # --- Accuracy vs planted ground truth ---
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
        "layer_1_exact_matches": exact_count,
        "layer_2_fuzzy_matches": fuzzy_count,
        "layer_3_llm_matches": llm_result["llm_matches"],
        "layer_3_exceptions_flagged": llm_result["exceptions_flagged"],
        "total_matched": total_matched,
        "auto_match_rate": round(total_matched / max(total_bank, 1) * 100, 1),
        "correct_matches": correct,
        "incorrect_matches": incorrect,
        "match_precision": round(
            correct / max(correct + incorrect, 1) * 100, 1
        ),
    }
