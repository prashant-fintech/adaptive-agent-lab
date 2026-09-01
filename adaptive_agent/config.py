"""Project configuration.

Every setting is a plain module-level constant read from the environment
(via a `.env` file if present), so the resolved values can be inspected in
a REPL:

    >>> from adaptive_agent import config
    >>> config.NEO4J_URI
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is a convenience - plain env vars work without it
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# --- Neo4j ----------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "please-change-me")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- Chat LLM (any OpenAI-compatible endpoint; defaults to local Ollama) --
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")

# --- Embeddings (local, via sentence-transformers) ------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Fine-tuning ----------------------------------------------------------
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-0.6B")
ADAPTER_DIR = ARTIFACTS_DIR / "polite_adapter"
POLITE_PAIRS_PATH = ARTIFACTS_DIR / "polite_pairs.jsonl"

ARTIFACTS_DIR.mkdir(exist_ok=True)
