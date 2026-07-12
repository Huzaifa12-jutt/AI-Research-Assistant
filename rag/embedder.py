# ==========================
# rag/embedder.py
#
# Uses FastEmbed (ONNX runtime) for lightweight embeddings.
# Uses a model that is actually supported by FastEmbed.
# ==========================

from langchain_community.embeddings import FastEmbedEmbeddings


def get_embedding_model():
    """
    Load (and lazily cache-download) the embedding model.
    FastEmbed supports these models:
    - BAAI/bge-small-en-v1.5 (recommended, ~33MB)
    - BAAI/bge-base-en-v1.5
    - BAAI/bge-large-en-v1.5
    - sentence-transformers/all-MiniLM-L6-v2 (NOT supported by FastEmbed!)
    """
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"  # ✅ FastEmbed supported
    )
    
    print("✅ FastEmbed Model Loaded (BAAI/bge-small-en-v1.5)")
    return embeddings