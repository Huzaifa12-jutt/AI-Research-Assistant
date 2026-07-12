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

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 4))

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ==============================
# Paths
# ==============================

DOCUMENTS_PATH = "documents"

# Streamlit Community Cloud mounts the repo as read-only — only /tmp
# is writable. So the Chroma vector DB must live in a temp directory,
# not inside the repo folder.
VECTOR_DB_PATH = tempfile.mkdtemp(prefix="chroma_db_")

MAX_FILE_SIZE_MB = 25
MAX_FILES = 5
