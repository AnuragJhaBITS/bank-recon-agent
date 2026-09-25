import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recon.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# --- Matching thresholds ---
FUZZY_AMOUNT_TOLERANCE = 0.02        # 2% tolerance on amounts
FUZZY_DATE_WINDOW_DAYS = 5           # bank vs GL date can differ by up to 5 days
FUZZY_DESCRIPTION_THRESHOLD = 65     # minimum fuzz ratio for description similarity
FUZZY_COMBINED_SCORE_CUTOFF = 0.70   # weighted score must exceed this to accept
