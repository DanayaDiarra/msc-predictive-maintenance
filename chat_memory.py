"""
chat_memory.py — OrchestrAI NOC
Per-user SQLite conversation persistence (no RunnableWithMessageHistory).

Public API
----------
get_history(user_id)              → SQLChatMessageHistory  (full log, read/write)
get_windowed_messages(user_id)    → list[BaseMessage]       (last WINDOW msgs, read-only)
save_turn(user_id, question, ans) → None                    (append one Q+A pair)
clear_history(user_id)            → None
load_history_for_display(user_id) → list[dict]              (for Streamlit UI)
"""

import re
from pathlib import Path
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

HISTORY_DIR = Path(__file__).resolve().parent / "data" / "chat_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

WINDOW     = 10       # messages (= 5 turns) sent to the LLM per request
SESSION_ID = "main"   # one thread per user DB


# ─────────────────────────────────────────────────────────────────────────────
#  CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _db_url(user_id: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", user_id)
    return f"sqlite:///{HISTORY_DIR / safe}.db"


def get_history(user_id: str) -> SQLChatMessageHistory:
    """Full per-user SQLChatMessageHistory (all messages, read + write)."""
    return SQLChatMessageHistory(session_id=SESSION_ID, connection=_db_url(user_id))


def get_windowed_messages(user_id: str) -> list[BaseMessage]:
    """
    Return the last WINDOW messages from the user's history as a plain list.
    These are injected into the LLM's chat_history slot — no subclassing needed.
    """
    return get_history(user_id).messages[-WINDOW:]


def save_turn(user_id: str, question: str, answer: str) -> None:
    """Append one human+AI turn to the user's persistent history."""
    get_history(user_id).add_messages([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])


def clear_history(user_id: str) -> None:
    """Wipe all messages for this user."""
    get_history(user_id).clear()


def load_history_for_display(user_id: str) -> list[dict]:
    """
    Return the full history as [{role, content}] dicts for the Streamlit UI.
    engine/tools_used fields are not stored in the DB — the caller may add them
    from session_state if available.
    """
    result = []
    for msg in get_history(user_id).messages:
        role = "user" if msg.type == "human" else "assistant"
        result.append({"role": role, "content": msg.content})
    return result
