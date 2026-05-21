"""
chat_memory.py — OrchestrAI NOC
LangChain-backed conversation memory with per-user SQLite persistence.

Usage:
    from chat_memory import build_chain, get_history, clear_history, load_history_for_display

Each user gets their own DB at data/chat_history/{user_id}.db.
Messages window: last 10 messages (5 turns) sent to the LLM per request.
"""

import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

HISTORY_DIR = Path(__file__).resolve().parent / "data" / "chat_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 10          # number of past messages passed to the LLM
SESSION_ID = "main"  # one conversation thread per user

SYSTEM_PROMPT = (
    "You are an expert telecom BTS maintenance engineer for the OrchestrAI NOC system "
    "covering 25 West African stations (Senegal and Mali). "
    "You help NOC engineers diagnose faults, interpret RUL predictions, and plan maintenance. "
    "Answer alarm codes, procedures, RUL interpretation concisely and actionably. "
    "Cite [DOC-ID] when referencing SOPs or manuals. "
    "When you don't know, say so clearly rather than guessing."
)


# ─────────────────────────────────────────────────────────────────────────────
#  PER-USER HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _db_url(user_id: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", user_id)
    return f"sqlite:///{HISTORY_DIR / safe}.db"


def get_history(user_id: str) -> SQLChatMessageHistory:
    """Return the SQLChatMessageHistory for this user (creates DB on first use)."""
    return SQLChatMessageHistory(
        session_id=SESSION_ID,
        connection=_db_url(user_id),
    )


def clear_history(user_id: str) -> None:
    """Wipe all messages for this user."""
    get_history(user_id).clear()


def load_history_for_display(user_id: str) -> list[dict]:
    """
    Return chat history as a list of {'role','content'} dicts
    suitable for direct rendering in the Streamlit chat UI.
    """
    history = get_history(user_id)
    result = []
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        result.append({"role": role, "content": msg.content})
    return result


def _windowed_history(user_id: str) -> SQLChatMessageHistory:
    """
    Returns a SQLChatMessageHistory whose get_messages() is transparently
    windowed to the last WINDOW messages.  We achieve this by subclassing
    inline so the chain only ever sees the tail.
    """
    class WindowedHistory(SQLChatMessageHistory):
        def get_messages(self):          # LangChain < 0.3 API
            return super().messages[-WINDOW:]
        @property
        def messages(self):              # LangChain >= 0.3 API
            return super().messages[-WINDOW:]

    return WindowedHistory(
        session_id=SESSION_ID,
        connection=_db_url(user_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHAIN BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])


def build_groq_chain(api_key: str) -> RunnableWithMessageHistory:
    """Build a RunnableWithMessageHistory backed by ChatGroq (LLaMA 3.3 70B)."""
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        temperature=0.7,
    )
    chain = _make_prompt() | llm
    return chain


def build_anthropic_chain(api_key: str) -> RunnableWithMessageHistory:
    """Build a RunnableWithMessageHistory backed by ChatAnthropic (Claude Haiku)."""
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(
        api_key=api_key,
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        temperature=0.7,
    )
    chain = _make_prompt() | llm
    return chain


def invoke_with_memory(chain, user_id: str, question: str) -> str:
    """
    Invoke `chain` with the windowed history for `user_id`.
    Saves the new human+AI messages to the persistent DB automatically.
    Returns the assistant's text response.
    """
    runnable = RunnableWithMessageHistory(
        chain,
        lambda sid: _windowed_history(user_id),
        input_messages_key="input",
        history_messages_key="history",
    )
    response = runnable.invoke(
        {"input": question},
        config={"configurable": {"session_id": SESSION_ID}},
    )
    # response is an AIMessage; extract text
    return response.content if hasattr(response, "content") else str(response)
