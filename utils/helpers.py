# ==========================
# utils/helpers.py
# ==========================

from datetime import datetime


def format_file_size(size_bytes):
    """Human-readable file size, e.g. 2.3 MB."""

    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} TB"


def chat_history_to_markdown(messages, title="AI Research Assistant - Chat Export"):
    """Turn the session chat history into a downloadable markdown transcript."""

    lines = [
        f"# {title}",
        f"_Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]

    for msg in messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"**{role}:**\n\n{msg['content']}\n")

    return "\n".join(lines)


def looks_like_valid_groq_key(key):
    """Light sanity check, not a real validation call."""

    if not key:
        return False

    return key.strip().startswith("gsk_") and len(key.strip()) > 20


SAMPLE_QUESTIONS = [
    "Summarize this document in 5 bullet points",
    "What are the key findings or conclusions?",
    "Are there any numbers, dates, or statistics mentioned?",
    "Explain the methodology used, if any",
]
