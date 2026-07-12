# ==========================
# rag/chunker.py
# ==========================

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def split_documents(documents, chunk_size=None, chunk_overlap=None):
    """
    Split documents into smaller, overlapping chunks so retrieval
    can find precise, relevant passages instead of whole pages.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or DEFAULT_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return chunks
