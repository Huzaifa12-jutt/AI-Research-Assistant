# ==========================
# rag/vector_store.py
# ==========================

import logging
import shutil

from langchain_chroma import Chroma

from core.config import VECTOR_DB_PATH

# Chroma's telemetry client has a known harmless bug where it logs
# "Failed to send telemetry event ..." errors on every call. It doesn't
# affect functionality, but it looks like a real error in the console,
# so it's silenced here.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def reset_vector_store():
    """
    Wipe any existing vector database so a fresh one can be built
    (used when the user uploads a new set of documents).
    """

    shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)


def create_vector_store(chunks, embedding_model):
    """
    Build a new Chroma vector database from document chunks and
    persist it to disk.
    """

    reset_vector_store()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH,
    )

    return vector_db


def load_vector_store(embedding_model):
    """
    Load an existing Chroma vector database from disk.
    """

    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model,
    )

    return vector_db
