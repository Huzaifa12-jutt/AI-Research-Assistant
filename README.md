# 🔬 AI Research Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own PDF
documents — with visible source citations, streamed answers, and a fast Groq-hosted LLM.

Built with **Streamlit · LangChain · Chroma · FastEmbed · Groq**.

---

## ✨ Features

- 📄 Upload multiple PDFs at once and chat across all of them
- ⚡ Fast, streamed answers via Groq's LPU inference (`openai/gpt-oss-120b` by default)
- 📚 Every answer shows the exact source passages (file + page number) it was grounded in
- 🪶 Lightweight embeddings (FastEmbed / ONNX) — no GPU or heavy `torch` install needed
- 🎛️ Adjustable model, temperature, chunk size, and retrieval depth from the sidebar
- 💾 Export any conversation as a Markdown file
- 🎨 Custom, responsive dark UI (no default Streamlit look)
- 🛡️ Friendly error handling — bad keys, rate limits, and empty PDFs never crash the app

---

## 🧰 Requirements

- Python 3.10–3.12
- A free Groq API key → https://console.groq.com/keys

---

## 🚀 Run it locally

```bash
# 1. Go into the project folder
cd AI-Research-Assistant

# 2. Create and activate a virtual environment
python -m venv venv_ai
# Windows:
venv_ai\Scripts\activate
# macOS/Linux:
source venv_ai/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
copy .env.example .env      # Windows
cp .env.example .env         # macOS/Linux
# then open .env and paste your GROQ_API_KEY

# 5. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

> No `.env` file? No problem — you can also paste your Groq API key directly into the
> sidebar at runtime.

The first time you process a PDF, the embedding model (~130 MB) downloads once and is
cached locally for every run after that.

---

## 📁 Project structure

```
AI-Research-Assistant/
├── app.py                  # Streamlit UI + app flow
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml         # Theme
├── core/
│   ├── config.py           # Settings, env vars, model list
│   └── prompts.py          # RAG prompt template
├── rag/
│   ├── loader.py           # PDF loading (multi-file)
│   ├── chunker.py          # Text splitting
│   ├── embedder.py         # FastEmbed embedding model
│   ├── vector_store.py     # Chroma vector DB
│   ├── retriever.py        # Retriever builder
│   └── chatbot.py          # Groq LLM + RAG chain, streaming
├── utils/
│   └── helpers.py          # Formatting, export, small utilities
├── assets/
│   └── style.css           # Custom UI theme
└── documents/               # (empty — scratch space for uploads)
```

---

## ☁️ Deploying

### Option A — Streamlit Community Cloud (easiest, free)

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick your repo, branch, and
   `app.py` as the entry point.
3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy. That's it — no separate frontend/backend to manage.

### Option B — Render / Railway

1. Push to GitHub.
2. Create a new **Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add an environment variable `GROQ_API_KEY` in the dashboard.

Both options work with this repo as-is — no code changes needed.

---

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `Please add your Groq API key` | Paste a key in the sidebar, or set `GROQ_API_KEY` in `.env` / host secrets |
| "That doesn't look like a Groq key" warning | Groq keys start with `gsk_` — double check you copied the whole thing |
| First PDF process is slow | Normal — the embedding model downloads once (~130 MB), then it's cached |
| "No extractable text was found" | The PDF is likely scanned images with no text layer (OCR isn't included) |
| 429 rate limit error while chatting | Groq's free tier has request limits — wait a few seconds and retry |

---

## 🗺️ Notes on the model

Groq is retiring `llama-3.3-70b-versatile` on **August 16, 2026**. This project defaults
to `openai/gpt-oss-120b`, Groq's current recommended production model, and you can switch
models anytime from the sidebar dropdown.
