"""
Layer 3: LLM-powered exception classifier.

The agent receives unmatched bank and GL rows as read-only context
and returns structured classifications.  It NEVER generates amounts
or dates — those come from the database rows it was given.

If the OpenAI key is missing or the call fails, a rule-based
fallback runs instead so the rest of the pipeline still works.
"""

import json
import os
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL

SYSTEM_PROMPT = """You are a bank reconciliation assistant.
You will receive two lists:
  1. UNMATCHED BANK TRANSACTIONS – rows from the bank statement with no GL match.
  2. UNMATCHED GL ENTRIES – rows from the general ledger with no bank match.

Your job:
  • Identify any pairs that SHOULD be matched despite surface differences
    (e.g. different descriptions for the same vendor, small amount rounding,
    date offsets, or duplicates).
  • For rows with no plausible counterpart, classify the exception.

RULES:
  • Never invent or modify any amounts or dates.  Cite the exact figures
    from the rows you were given.
  • Return ONLY a JSON array.  No markdown, no commentary.

Each element must be one of:

  {"action": "match",
   "bank_id": <int>,
   "gl_id": <int>,
   "confidence": <float 0-1>,
   "explanation": "<why these belong together>"}

  {"action": "flag",
   "side": "bank" | "gl",
   "row_id": <int>,
   "category": "missing_counterpart" | "suspected_duplicate" | "requires_review",
   "explanation": "<what the issue is and suggested correcting entry>"}
"""


def _build_user_message(
    unmatched_bank: list, unmatched_gl: list
) -> str:
    bank_lines = []
    for b in unmatched_bank:
        bank_lines.append(
            f"  id={b.id}  date={b.date}  amount={b.amount:,.2f}  "
            f"ref={b.reference}  desc=\"{b.description}\""
        )

    gl_lines = []
    for g in unmatched_gl:
        gl_lines.append(
            f"  id={g.id}  date={g.date}  amount={g.amount:,.2f}  "
            f"ref={g.reference}  acct={g.account_code}  desc=\"{g.description}\""
        )

    return (
        "UNMATCHED BANK TRANSACTIONS:\n"
        + ("\n".join(bank_lines) if bank_lines else "  (none)")
        + "\n\nUNMATCHED GL ENTRIES:\n"
        + ("\n".join(gl_lines) if gl_lines else "  (none)")
    )


def _parse_response(text: str) -> list[dict]:
    """Strip markdown fences if present, then parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def classify_exceptions(
    unmatched_bank: list, unmatched_gl: list
) -> list[dict]:
    """
    Call the LLM to classify leftover exceptions.
    Falls back to a simple rule-based classifier if the API key
    is missing or the call fails.
    """
    if not OPENAI_API_KEY:
        return _fallback_classifier(unmatched_bank, unmatched_gl)

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_msg = _build_user_message(unmatched_bank, unmatched_gl)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4096,
        )
        raw = response.choices[0].message.content
        return _parse_response(raw)

    except Exception as e:
        print(f"[agent] LLM call failed ({e}), using fallback classifier")
        return _fallback_classifier(unmatched_bank, unmatched_gl)


# ──────────────────────────────────────────────
# Rule-based fallback (no API key needed)
# ──────────────────────────────────────────────

def _fallback_classifier(
    unmatched_bank: list, unmatched_gl: list
) -> list[dict]:
    """
    Simple heuristic: try to pair by closest amount within a date
    window.  Anything still unpaired is flagged as missing.
    """
    results = []
    used_gl_ids = set()

    for bank in unmatched_bank:
        best_gl = None
        best_diff = float("inf")

        for gl in unmatched_gl:
            if gl.id in used_gl_ids:
                continue
            day_gap = abs((bank.date - gl.date).days)
            if day_gap > 7:
                continue
            amt_diff = abs(bank.amount - gl.amount)
            if amt_diff < best_diff:
                best_diff = amt_diff
                best_gl = gl

        if best_gl and best_diff < abs(bank.amount) * 0.05:
            results.append({
                "action": "match",
                "bank_id": bank.id,
                "gl_id": best_gl.id,
                "confidence": round(max(0.5, 1.0 - best_diff / max(abs(bank.amount), 1)), 2),
                "explanation": (
                    f"Fallback heuristic: closest amount within date window. "
                    f"Amount difference: {best_diff:,.2f}, "
                    f"date gap: {abs((bank.date - best_gl.date).days)} day(s)."
                ),
            })
            used_gl_ids.add(best_gl.id)
        else:
            results.append({
                "action": "flag",
                "side": "bank",
                "row_id": bank.id,
                "category": "missing_counterpart",
                "explanation": (
                    f"No GL entry found for bank transaction "
                    f"{bank.description} on {bank.date} "
                    f"for {bank.amount:,.2f}."
                ),
            })

    for gl in unmatched_gl:
        if gl.id not in used_gl_ids:
            results.append({
                "action": "flag",
                "side": "gl",
                "row_id": gl.id,
                "category": "missing_counterpart",
                "explanation": (
                    f"No bank transaction found for GL entry "
                    f"{gl.description} on {gl.date} "
                    f"for {gl.amount:,.2f}."
                ),
            })

    return results
