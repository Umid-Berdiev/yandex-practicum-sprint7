"""Settings of the bot. Every value can be overridden with an environment variable or .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "faiss_index"))

# Embeddings: the same model must be used for indexing and for queries.
EMBEDDING_MODEL = "BAAI/bge-m3"
# Local copy of the model (scripts/download_model.sh). If the folder is missing,
# the model is downloaded from the Hugging Face Hub by name.
EMBEDDING_MODEL_DIR = Path(os.environ.get("EMBEDDING_MODEL_DIR", ROOT / "models" / "bge-m3"))

# LLM: any server with an OpenAI-compatible API (OpenAI, Ollama, vLLM).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed-for-local-llm")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
# "none" switches off hidden thinking of reasoning models; empty = provider default.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "none")

# Retrieval.
TOP_K = int(os.environ.get("TOP_K", "5"))
# Minimal cosine similarity of a chunk to be used as context. Measured on the index:
# questions outside the knowledge base score below 0.47, questions about it - above 0.55.
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.5"))
