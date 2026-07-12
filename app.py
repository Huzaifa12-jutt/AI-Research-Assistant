# ==========================================================
# AI Research Assistant
# Retrieval-Augmented Generation over your own PDFs, powered
# by Groq (LPU inference) + LangChain + FAISS + Streamlit.
# ==========================================================

import os
import tempfile
import traceback

import streamlit as st

from core.config import (
    GROQ_API_KEY,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_TEMPERATURE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    MAX_FILE_SIZE_MB,
    MAX_FILES,
)
from rag.loader import load_multiple_pdfs
from rag.chunker import split_documents
from rag.embedder import get_embedding_model
from rag.vector_store import create_vector_store, reset_vector_store
from rag.retriever import get_retriever
from rag.chatbot import create_chatbot
from utils.helpers import (
    format_file_size,
    chat_history_to_markdown,
    looks_like_valid_groq_key,
    SAMPLE_QUESTIONS,
)


# ----------------------------------------------------------
# Page config (must be the first Streamlit call)
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# ----------------------------------------------------------
# Cached, expensive resources
# ----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embedding_model()


# ----------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------

defaults = {
    "chatbot": None,
    "kb_ready": False,
    "messages": [],
    "doc_stats": {},
    "total_chunks": 0,
    "manual_api_key": "",
    "active_model": DEFAULT_MODEL,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_api_key():
    return GROQ_API_KEY or st.session_state.manual_api_key


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔬 AI Research Assistant")
    st.caption("RAG over your PDFs — powered by Groq")

    st.divider()

    # ---- API key ----
    st.markdown("**🔑 Groq API Key**")

    if GROQ_API_KEY:
        st.success("Loaded from environment (.env)", icon="✅")
    else:
        st.session_state.manual_api_key = st.text_input(
            "Enter your Groq API key",
            type="password",
            value=st.session_state.manual_api_key,
            placeholder="gsk_...",
            help="Get a free key at console.groq.com/keys",
            label_visibility="collapsed",
        )
        if st.session_state.manual_api_key and not looks_like_valid_groq_key(
            st.session_state.manual_api_key
        ):
            st.warning("That doesn't look like a typical Groq key (usually starts with `gsk_`).")
        st.caption("🔗 [Get a free key →](https://console.groq.com/keys)")

    st.divider()

    # ---- Model & settings ----
    st.markdown("**⚙️ Model**")
    model_name = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        label_visibility="collapsed",
    )
    if model_name == "llama-3.3-70b-versatile":
        st.caption("⚠️ Groq is retiring this model on 08/16/2026 — prefer gpt-oss-120b.")

    with st.expander("Advanced settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
        chunk_size = st.slider("Chunk size", 300, 2000, DEFAULT_CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, 25)
        top_k = st.slider("Chunks retrieved per question (top-k)", 1, 10, DEFAULT_TOP_K, 1)

    st.divider()

    # ---- Upload ----
    st.markdown("**📄 Documents**")
    uploaded_files = st.file_uploader(
        f"Upload up to {MAX_FILES} PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process = st.button("🚀 Process documents", use_container_width=True)

    if st.session_state.doc_stats:
        st.caption("**Loaded documents:**")
        for name, pages in st.session_state.doc_stats.items():
            st.caption(f"• {name} — {pages} pages")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💬 New chat", use_container_width=True, disabled=not st.session_state.kb_ready):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🗑️ Reset all", use_container_width=True, disabled=not st.session_state.kb_ready):
            reset_vector_store()
            st.session_state.chatbot = None
            st.session_state.kb_ready = False
            st.session_state.messages = []
            st.session_state.doc_stats = {}
            st.session_state.total_chunks = 0
            st.rerun()

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export chat (.md)",
            data=chat_history_to_markdown(st.session_state.messages),
            file_name="research_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown(
        '<div class="ara-footer">Built with Streamlit · LangChain · FAISS · Groq</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------
# Process documents
# ----------------------------------------------------------

if process:
    api_key = get_api_key()

    if not api_key:
        st.error("Please add your Groq API key in the sidebar first.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    if len(uploaded_files) > MAX_FILES:
        st.error(f"Please upload at most {MAX_FILES} files at a time.")
        st.stop()

    oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE_MB * 1024 * 1024]
    if oversized:
        st.error(f"These files exceed {MAX_FILE_SIZE_MB} MB: {', '.join(oversized)}")
        st.stop()

    try:
        with st.status("Processing documents...", expanded=True) as status:
            temp_dir = tempfile.mkdtemp()
            saved = []

            st.write("📥 Saving uploads...")
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved.append((path, f.name))

            st.write("📄 Reading PDF pages...")
            documents, page_stats = load_multiple_pdfs(saved)

            if not documents:
                status.update(label="No readable text found.", state="error")
                st.error(
                    "No extractable text was found in these PDFs. "
                    "They may be scanned images without a text layer."
                )
                st.stop()

            st.write("✂️ Splitting into chunks...")
            chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            st.write("🧠 Loading embedding model (first run downloads ~130MB, then it's cached)...")
            embeddings = load_embeddings()

            st.write("🗂️ Building vector index...")
            vector_db = create_vector_store(chunks, embeddings)

            st.write("🔗 Connecting retriever + Groq model...")
            retriever = get_retriever(vector_db=vector_db, top_k=top_k)
            chatbot = create_chatbot(
                retriever=retriever,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
            )

            st.session_state.chatbot = chatbot
            st.session_state.kb_ready = True
            st.session_state.doc_stats = page_stats
            st.session_state.total_chunks = len(chunks)
            st.session_state.active_model = model_name
            st.session_state.messages = []

            status.update(label="✅ Documents processed successfully!", state="complete")

        st.rerun()

    except Exception as e:
        st.error(f"Something went wrong while processing your documents: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()


# ----------------------------------------------------------
# Hero header
# ----------------------------------------------------------

st.markdown(
    """
    <div class="ara-hero">
        <span class="eyebrow">Retrieval-Augmented Generation</span>
        <h1>🔬 AI Research Assistant</h1>
        <p>Upload your PDFs and ask questions in plain language. Every answer is
        grounded in your documents, with sources you can check.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# Main area
# ----------------------------------------------------------

if not st.session_state.kb_ready:
    st.markdown("#### How it works")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📤", "1. Upload", "Add one or more PDFs — papers, reports, notes, textbooks."),
        ("🧩", "2. Index", "Your documents are chunked and embedded locally with FastEmbed."),
        ("💬", "3. Ask", "Chat naturally. Groq's LPU inference keeps answers fast."),
        ("📚", "4. Verify", "Every answer links back to the exact source passage."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""<div class="ara-card">
                        <span class="icon">{icon}</span>
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.info("👈 Add your Groq API key and upload a PDF from the sidebar to get started.")

else:
    # ---- Stats row ----
    total_pages = sum(st.session_state.doc_stats.values())
    st.markdown(
        f"""
        <div class="ara-stats">
            <div class="ara-stat"><span class="num">{len(st.session_state.doc_stats)}</span><span class="label">Documents</span></div>
            <div class="ara-stat"><span class="num">{total_pages}</span><span class="label">Pages</span></div>
            <div class="ara-stat"><span class="num">{st.session_state.total_chunks}</span><span class="label">Chunks indexed</span></div>
            <div class="ara-stat"><span class="num">{st.session_state.active_model}</span><span class="label">Model</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Chat history ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} source(s)"):
                    for doc in msg["sources"]:
                        page = doc.metadata.get("page")
                        page_label = f"page {page + 1}" if isinstance(page, int) else "unknown page"
                        source_name = doc.metadata.get("source", "document")
                        snippet = doc.page_content[:280].strip().replace("\n", " ")
                        st.markdown(
                            f"""<div class="ara-source">
                                    <span class="tag">{source_name} · {page_label}</span>
                                    <span class="snippet">{snippet}...</span>
                                </div>""",
                            unsafe_allow_html=True,
                        )

    # ---- Sample questions (only before first message) ----
    if not st.session_state.messages:
        st.caption("Try asking:")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, q in zip(cols, SAMPLE_QUESTIONS):
            with col:
                if st.button(q, use_container_width=True, key=f"sample_{q}"):
                    st.session_state["pending_question"] = q
                    st.rerun()

    question = st.chat_input("Ask anything about your documents...")

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(st.session_state.chatbot.stream(question))
                sources = list(st.session_state.chatbot.last_sources)

                if sources:
                    with st.expander(f"📚 {len(sources)} source(s)"):
                        for doc in sources:
                            page = doc.metadata.get("page")
                            page_label = f"page {page + 1}" if isinstance(page, int) else "unknown page"
                            source_name = doc.metadata.get("source", "document")
                            snippet = doc.page_content[:280].strip().replace("\n", " ")
                            st.markdown(
                                f"""<div class="ara-source">
                                        <span class="tag">{source_name} · {page_label}</span>
                                        <span class="snippet">{snippet}...</span>
                                    </div>""",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                error_text = str(e)
                if "401" in error_text or "invalid_api_key" in error_text.lower():
                    friendly = "Your Groq API key looks invalid or expired. Please check it in the sidebar."
                elif "429" in error_text:
                    friendly = "Rate limit reached on Groq's free tier. Wait a moment and try again."
                else:
                    friendly = f"I hit an error talking to the model: {error_text}"

                st.error(friendly)
                st.session_state.messages.append({"role": "assistant", "content": friendly, "sources": []})