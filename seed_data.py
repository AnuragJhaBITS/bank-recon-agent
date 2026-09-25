"""
Generate synthetic bank-statement and general-ledger data with
deliberately planted reconciliation errors.

Error distribution (approximate):
  60%  clean          – exact date, amount, reference, description
  10%  timing         – GL date shifted ±1-4 days from bank date
   8%  amount typo    – GL amount differs by a small random delta
   7%  desc mismatch  – bank uses abbreviation, GL uses full name
   5%  missing in GL  – bank txn exists, no GL counterpart
   5%  missing in bank– GL entry exists, no bank counterpart
   5%  duplicate      – same bank txn appears twice

Because *we* plant the errors, we know the ground-truth pairing.
This lets us compute precision, recall, and accuracy after matching.
"""

import random
from datetime import date, timedelta
from models import (
    BankTransaction, GLEntry, ErrorType,
    SessionLocal, reset_db
)

random.seed(42)

# --- Realistic transaction templates ---

VENDORS = [
    ("Amazon Web Services", "AWS", "6100"),
    ("Google Cloud Platform", "GCP", "6100"),
    ("WeWork Office Rent", "WEWORK", "6200"),
    ("Zomato Corporate Meals", "ZOMATO", "6300"),
    ("Airtel Telecom", "AIRTEL TEL", "6400"),
    ("HDFC Bank Charges", "HDFC CHRG", "6500"),
    ("Swiggy Instamart", "SWIGGY", "6300"),
    ("Adobe Creative Cloud", "ADOBE CC", "6100"),
    ("Slack Technologies", "SLACK TECH", "6100"),
    ("Urban Company Services", "URBAN CO", "6600"),
    ("Flipkart Business", "FLIPKART", "6700"),
    ("Tata Power Electricity", "TATA ELEC", "6400"),
    ("Reliance Jio Data", "JIO DATA", "6400"),
    ("Zoho Subscriptions", "ZOHO SUB", "6100"),
    ("MakeMyTrip Travel", "MMT TRAVEL", "6800"),
]

CUSTOMERS = [
    ("Infosys Ltd", "INFY"),
    ("Wipro Technologies", "WIPRO"),
    ("HCL Technologies", "HCL TECH"),
    ("Bajaj Finance", "BAJFIN"),
    ("Tata Consultancy", "TCS"),
    ("Tech Mahindra", "TECHMAH"),
    ("L&T Infotech", "LTI"),
    ("Mindtree Ltd", "MINDTREE"),
]

SALARY_NAMES = [
    "Aarav Sharma", "Priya Patel", "Rohan Gupta", "Sneha Reddy",
    "Vikram Singh", "Ananya Iyer", "Karthik Nair", "Divya Joshi",
    "Arjun Mehta", "Neha Kapoor", "Siddharth Das", "Pooja Rao",
]


def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _ref_number() -> str:
    return f"REF{random.randint(100000, 999999)}"


def generate(n_transactions: int = 500) -> dict:
    """
    Generate n_transactions base pairs, then apply error transformations.
    Returns summary stats of what was planted.
    """
    reset_db()
    session = SessionLocal()

    start = date(2025, 1, 1)
    end = date(2025, 12, 31)

    # decide how many of each type
    n_clean = int(n_transactions * 0.60)
    n_timing = int(n_transactions * 0.10)
    n_amount = int(n_transactions * 0.08)
    n_desc = int(n_transactions * 0.07)
    n_missing_gl = int(n_transactions * 0.05)
    n_missing_bank = int(n_transactions * 0.05)
    n_duplicate = n_transactions - (
        n_clean + n_timing + n_amount + n_desc + n_missing_gl + n_missing_bank
    )

    error_plan = (
        [ErrorType.NONE] * n_clean
        + [ErrorType.TIMING] * n_timing
        + [ErrorType.AMOUNT_TYPO] * n_amount
        + [ErrorType.DESCRIPTION_MISMATCH] * n_desc
        + [ErrorType.MISSING_GL] * n_missing_gl
        + [ErrorType.MISSING_BANK] * n_missing_bank
        + [ErrorType.DUPLICATE] * n_duplicate
    )
    random.shuffle(error_plan)

    pair_id = 0
    stats = {e.value: 0 for e in ErrorType}

    for error_type in error_plan:
        pair_id += 1

        # pick a random transaction type
        txn_kind = random.choice(["vendor", "customer", "salary", "bank_charge"])

        if txn_kind == "vendor":
            full_name, abbr, acct = random.choice(VENDORS)
            amount = -round(random.uniform(500, 80000), 2)  # outflow
        elif txn_kind == "customer":
            full_name, abbr = random.choice(CUSTOMERS)
            acct = "4100"
            amount = round(random.uniform(10000, 500000), 2)  # inflow
        elif txn_kind == "salary":
            name = random.choice(SALARY_NAMES)
            full_name = f"Salary - {name}"
            abbr = f"SAL {name.split()[0].upper()}"
            acct = "6000"
            amount = -round(random.uniform(30000, 120000), 2)
        else:
            full_name = "Bank Service Charge"
            abbr = "BANK CHRG"
            acct = "6500"
            amount = -round(random.uniform(50, 2000), 2)

        txn_date = _random_date(start, end)
        ref = _ref_number()

        # -- Base bank row (uses abbreviation) --
        bank_desc = abbr
        bank_amount = amount
        bank_date = txn_date

        # -- Base GL row (uses full name) -- but for clean matches,
        # we use the SAME description so exact matching works.
        gl_desc = abbr  # same as bank for clean pairs
        gl_amount = amount
        gl_date = txn_date
        gl_ref = ref

        # --- Apply the planted error ---

        if error_type == ErrorType.TIMING:
            shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            gl_date = txn_date + timedelta(days=shift)

        elif error_type == ErrorType.AMOUNT_TYPO:
            # small perturbation: swap two digits or add/subtract a small %
            if random.random() < 0.5:
                gl_amount = amount + random.choice([-1, 1]) * round(
                    abs(amount) * random.uniform(0.001, 0.015), 2
                )
            else:
                # transpose last two digits of the integer part
                s = str(int(abs(amount)))
                if len(s) >= 2:
                    s = s[:-2] + s[-1] + s[-2]
                gl_amount = float(s) * (1 if amount > 0 else -1)

        elif error_type == ErrorType.DESCRIPTION_MISMATCH:
            # GL uses full vendor/customer name; bank uses abbreviation
            gl_desc = full_name

        elif error_type == ErrorType.MISSING_GL:
            # bank txn exists, no GL entry
            bank = BankTransaction(
                date=bank_date, description=bank_desc,
                amount=bank_amount, reference=ref,
                planted_error=error_type, true_pair_id=pair_id,
            )
            session.add(bank)
            stats[error_type.value] += 1
            continue

        elif error_type == ErrorType.MISSING_BANK:
            # GL entry exists, no bank txn
            gl = GLEntry(
                date=gl_date, description=gl_desc,
                amount=gl_amount, reference=gl_ref,
                account_code=acct,
                planted_error=error_type, true_pair_id=pair_id,
            )
            session.add(gl)
            stats[error_type.value] += 1
            continue

        elif error_type == ErrorType.DUPLICATE:
            # add the normal pair, then add a second bank row
            bank1 = BankTransaction(
                date=bank_date, description=bank_desc,
                amount=bank_amount, reference=ref,
                planted_error=ErrorType.NONE, true_pair_id=pair_id,
            )
            gl1 = GLEntry(
                date=gl_date, description=gl_desc,
                amount=gl_amount, reference=gl_ref,
                account_code=acct,
                planted_error=ErrorType.NONE, true_pair_id=pair_id,
            )
            # the duplicate bank row
            dup_ref = ref + "D"
            bank2 = BankTransaction(
                date=bank_date + timedelta(days=random.randint(0, 2)),
                description=bank_desc,
                amount=bank_amount,
                reference=dup_ref,
                planted_error=ErrorType.DUPLICATE,
                true_pair_id=pair_id,
            )
            session.add_all([bank1, gl1, bank2])
            stats[error_type.value] += 1
            continue

        # --- Create the pair ---
        bank = BankTransaction(
            date=bank_date, description=bank_desc,
            amount=bank_amount, reference=ref,
            planted_error=error_type, true_pair_id=pair_id,
        )
        gl = GLEntry(
            date=gl_date, description=gl_desc,
            amount=gl_amount, reference=gl_ref,
            account_code=acct,
            planted_error=error_type, true_pair_id=pair_id,
        )
        session.add_all([bank, gl])
        stats[error_type.value] += 1

    session.commit()

    total_bank = session.query(BankTransaction).count()
    total_gl = session.query(GLEntry).count()
    session.close()

    return {
        "planted_errors": stats,
        "total_bank_rows": total_bank,
        "total_gl_rows": total_gl,
    }


if __name__ == "__main__":
    result = generate(500)
    print("Data generated:")
    for k, v in result.items():
        print(f"  {k}: {v}")
