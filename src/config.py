"""Central configuration and LLM factory for the Sprints AI Orchestrator.

Curriculum mapping: Module 4-5 (LangChain & prompting) for model setup,
Module 10-12 (LangGraph orchestration) for the limits that control loops.
All limits come from the Sprints quality bar in the client kickoff deck.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CURRICULUM_DIR = DATA_DIR / "curriculum_docs"
LEARNER_CSV = DATA_DIR / "sprints_learner_db.csv"
LEARNER_DB = DATA_DIR / "sprints_learner.db"
CHROMA_DIR = DATA_DIR / "chroma_store"

CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
RETRIEVAL_TIMEOUT_MS = 1000
MAX_REFINEMENT_ITERATIONS = 3
MAX_RAG_REFORMULATIONS = 2
PASS_THRESHOLD = 70
UNABLE_TO_VERIFY = "Unable to verify"
TIER1_PARTNERS = frozenset({"fawry"})

CHAT_MODEL = os.getenv("SPRINTS_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("SPRINTS_EMBED_MODEL", "text-embedding-3-small")


def llm_available() -> bool:
    """Return True when an OpenAI key is configured, so LLM nodes can run."""
    return bool(os.getenv("OPENAI_API_KEY"))


def get_llm(temperature: float = 0.0):
    """Create the chat model used by every agent (Module 4: LangChain basics).

    Temperature 0 keeps routing and grading repeatable. Imported lazily so the
    framework-free parts of the package work without LangChain installed.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature, timeout=30, max_retries=1)
