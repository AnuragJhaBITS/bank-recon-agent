"""
Database models for the reconciliation engine.

Three core tables:
  - bank_transactions: rows from the bank statement
  - gl_entries: rows from the general ledger
  - matches: links a bank row to a GL row with match metadata

Design rule: every number that reaches the frontend originates from
a SQL query, never from the LLM.  The LLM only classifies and explains.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Date, DateTime,
    ForeignKey, Enum as SQLEnum, Text, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import enum

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class MatchType(str, enum.Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    LLM = "llm"


class ErrorType(str, enum.Enum):
    """Labels planted during data generation so we can score accuracy."""
    NONE = "none"
    TIMING = "timing"
    AMOUNT_TYPO = "amount_typo"
    DESCRIPTION_MISMATCH = "description_mismatch"
    DUPLICATE = "duplicate"
    MISSING_GL = "missing_gl"
    MISSING_BANK = "missing_bank"


class BankTransaction(Base):
    __tablename__ = "bank_transactions"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    description = Column(String(300), nullable=False)
    amount = Column(Float, nullable=False)
    reference = Column(String(50))
    planted_error = Column(SQLEnum(ErrorType), default=ErrorType.NONE)
    # the GL entry this was generated from (ground truth for scoring)
    true_pair_id = Column(Integer, nullable=True)

    match = relationship("Match", back_populates="bank_txn", uselist=False)


class GLEntry(Base):
    __tablename__ = "gl_entries"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    description = Column(String(300), nullable=False)
    amount = Column(Float, nullable=False)
    reference = Column(String(50))
    account_code = Column(String(20), nullable=False)
    planted_error = Column(SQLEnum(ErrorType), default=ErrorType.NONE)
    true_pair_id = Column(Integer, nullable=True)

    match = relationship("Match", back_populates="gl_entry", uselist=False)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    bank_id = Column(Integer, ForeignKey("bank_transactions.id"), nullable=False)
    gl_id = Column(Integer, ForeignKey("gl_entries.id"), nullable=False)
    match_type = Column(SQLEnum(MatchType), nullable=False)
    confidence = Column(Float, nullable=False)
    explanation = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    bank_txn = relationship("BankTransaction", back_populates="match")
    gl_entry = relationship("GLEntry", back_populates="match")


def init_db():
    Base.metadata.create_all(engine)


def reset_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
