# ==========================
# rag/retriever.py
# ==========================

from core.config import DEFAULT_TOP_K
import os
from langchain_community.vectorstores import FAISS
from core.config import VECTOR_DB_PATH


def get_retriever(vector_db=None, top_k=None, embedding_model=None):
    """
    Turn a vector store into a retriever that returns the top_k
    most relevant chunks for a given question.
    
    Args:
        vector_db: FAISS vector store instance (optional)
        top_k: Number of chunks to retrieve (default from config)
        embedding_model: Embedding model for loading from disk (optional)
    
    Returns:
        Retriever object
    """
    # If vector_db is provided, use it directly
    if vector_db is not None:
        retriever = vector_db.as_retriever(
            search_kwargs={"k": top_k or DEFAULT_TOP_K}
        )
        print(f"✅ Retriever created with top_k={top_k or DEFAULT_TOP_K}")
        return retriever
    
    # If no vector_db but embedding_model provided, try loading from disk
    if embedding_model is not None and os.path.exists(VECTOR_DB_PATH):
        vector_db = FAISS.load_local(
            folder_path=VECTOR_DB_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        retriever = vector_db.as_retriever(
            search_kwargs={"k": top_k or DEFAULT_TOP_K}
        )
        print(f"✅ Retriever loaded from disk with top_k={top_k or DEFAULT_TOP_K}")
        return retriever
    
    raise ValueError("No vector store found. Please process documents first or provide a vector_db.")