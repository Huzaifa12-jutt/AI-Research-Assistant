# ==========================
# rag/retriever.py
# ==========================

from core.config import DEFAULT_TOP_K


def get_retriever(vector_db, top_k=None):
    """
    Turn a vector store into a retriever that returns the top_k
    most relevant chunks for a given question.
    """

    retriever = vector_db.as_retriever(
        search_kwargs={"k": top_k or DEFAULT_TOP_K}
    )

    return retriever
