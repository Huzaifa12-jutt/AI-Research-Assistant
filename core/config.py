# ==========================
# core/config.py
# Central configuration for the AI Research Assistant.
# Nothing here raises on import — a missing API key is handled
# gracefully inside the Streamlit UI instead of crashing the app.
# ==========================

import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ==============================
# API
# ==============================

# Groq API key (get one free at https://console.groq.com/keys)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ==============================
# Model settings
# ==============================

# openai/gpt-oss-120b is Groq's current flagship production model.
# (llama-3.3-70b-versatile is being retired by Groq on 08/16/2026,
# so it is kept only as a legacy option in the UI dropdown.)
DEFAULT_MODEL = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
]

DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

# ==============================
# RAG settings
# ==============================

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 3))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight, no GPU needed

# ==============================
# Paths
# ==============================

DOCUMENTS_PATH = "documents"

# For FAISS - works on Streamlit Cloud (in-memory)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./faiss_index")

# Create directory if it doesn't exist
if not os.path.exists(VECTOR_DB_PATH):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

MAX_FILE_SIZE_MB = 25
MAX_FILES = 5

# ==============================
# ChromaDB fallback (if needed)
# ==============================

# ChromaDB requires a writable path. On Streamlit Cloud, use tempdir.
CHROMA_DB_PATH = tempfile.mkdtemp(prefix="chroma_db_")