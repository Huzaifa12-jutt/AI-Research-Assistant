# ==========================
# rag/embedder.py
#
# Uses FastEmbed (ONNX runtime) instead of sentence-transformers/torch.
# Same embedding quality for this use case, but no CUDA/torch downloads,
# which keeps the app light enough to deploy comfortably on free-tier
# hosting (Streamlit Community Cloud, Render, etc).
# ==========================

from langchain_community.embeddings import FastEmbedEmbeddings

from core.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Load (and lazily cache-download) the embedding model.
    """

    embeddings = FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings
