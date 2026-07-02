"""
Streamlit frontend for the History-Aware RAG Chatbot.

Run with:
    .\\venv\\Scripts\\streamlit.exe run streamlit_app.py
"""

import os
import sys
import time

# Fix encoding for Windows terminals (not needed for Streamlit, kept for safety)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #131a3a 50%, #1a1a2e 100%);
    color: #e0e6f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] * { color: #c9d4e8; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 8px;
    margin-bottom: 4px;
    animation: fadeInUp 0.3s ease both;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(16, 185, 129, 0.07);
    border: 1px solid rgba(16, 185, 129, 0.18);
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 12px !important;
    color: #e0e6f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.9) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Source expander ── */
details {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 6px 12px;
    margin-top: 6px;
}
summary { cursor: pointer; font-size: 0.78rem; color: #8892b0; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 0.75rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35) !important;
}

/* ── Sliders / selects ── */
[data-testid="stSlider"] .stMarkdown { color: #8892b0; }
.stSelectbox label, .stSlider label { color: #8892b0 !important; font-size: 0.82rem; }

/* ── Status badges ── */
.badge-ok   { color: #10b981; font-weight: 600; }
.badge-err  { color: #f43f5e; font-weight: 600; }

/* ── Fade-in animation ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Header ── */
.rag-header {
    text-align: center;
    padding: 20px 0 8px 0;
}
.rag-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.rag-header p {
    color: #8892b0;
    font-size: 0.95rem;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07); }

/* ── Thinking spinner ── */
.thinking-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #a78bfa;
    animation: pulse 1.4s ease infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}

/* Scrollable source block */
.source-block {
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #8892b0;
    max-height: 180px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.5;
    border-left: 3px solid #6366f1;
    margin-top: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

load_dotenv()

# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],          # list of {"role": "user"|"assistant", "content": str, "sources": list}
        "lc_history": [],        # LangChain message objects for the model
        "total_questions": 0,
        "db": None,
        "model": None,
        "embeddings": None,
        "top_k": 3,
        "max_history_turns": 6,
        "chat_model": DEFAULT_CHAT_MODEL,
        "ready": False,
        "init_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Backend initialisation (cached per session) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources(chat_model: str):
    """Load embeddings, vector store, and Groq model once per session."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    model = ChatGroq(model=chat_model, temperature=0)
    return embeddings, db, model


def ensure_ready(chat_model: str) -> bool:
    """Initialise resources if not already done. Returns True if OK."""
    if not os.getenv("GROQ_API_KEY"):
        st.session_state.init_error = "🔑 `GROQ_API_KEY` is missing. Add it to your `.env` file."
        st.session_state.ready = False
        return False
    if not os.path.exists(PERSIST_DIRECTORY):
        st.session_state.init_error = (
            f"📂 Vector database not found at `{PERSIST_DIRECTORY}`. "
            "Run **1_ingestion_pipeline.py** first."
        )
        st.session_state.ready = False
        return False
    try:
        embeddings, db, model = load_resources(chat_model)
        st.session_state.embeddings = embeddings
        st.session_state.db = db
        st.session_state.model = model
        st.session_state.ready = True
        st.session_state.init_error = None
        return True
    except Exception as exc:
        st.session_state.init_error = f"⚠️ Failed to initialise: {exc}"
        st.session_state.ready = False
        return False


# ── Core RAG logic ────────────────────────────────────────────────────────────
def rag_answer(user_question: str, top_k: int, max_history_turns: int):
    """Run history-aware retrieval + generation. Returns (answer, docs, search_q)."""
    db: Chroma = st.session_state.db
    model: ChatGroq = st.session_state.model
    lc_history: list = st.session_state.lc_history

    # Step 1 — Rewrite question as standalone if history exists
    if lc_history:
        rewrite_msgs = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question to be "
                    "standalone and searchable. Return only the rewritten question."
                )
            ),
        ] + lc_history[-max_history_turns:] + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        search_question = model.invoke(rewrite_msgs).content.strip()
    else:
        search_question = user_question

    # Step 2 — Retrieve docs
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(search_question)

    # Step 3 — Build context
    context_blocks = [
        f"[Document {i} | {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    ]
    context = "\n\n".join(context_blocks)

    combined_input = (
        f"Based on the following documents, answer this question: {user_question}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer clearly using only the provided documents. "
        "If the answer isn't in the documents, say: "
        '"I don\'t have enough information to answer that question based on the provided documents."'
    )

    # Step 4 — Generate answer
    answer_msgs = [
        SystemMessage(
            content="You are a helpful assistant answering questions from company documents."
        ),
    ] + lc_history[-max_history_turns:] + [
        HumanMessage(content=combined_input)
    ]
    answer = model.invoke(answer_msgs).content.strip()

    # Step 5 — Update LangChain history
    st.session_state.lc_history.append(HumanMessage(content=user_question))
    st.session_state.lc_history.append(AIMessage(content=answer))

    return answer, docs, search_question


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    selected_model = st.selectbox(
        "🧠 Groq Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.chat_model),
        key="model_select",
    )
    if selected_model != st.session_state.chat_model:
        st.session_state.chat_model = selected_model
        st.session_state.ready = False   # force re-init with new model
        load_resources.clear()

    top_k = st.slider(
        "📄 Documents to retrieve (Top-K)",
        min_value=1, max_value=8,
        value=st.session_state.top_k,
        key="top_k_slider",
    )
    st.session_state.top_k = top_k

    max_turns = st.slider(
        "💬 Max history turns",
        min_value=2, max_value=12, step=2,
        value=st.session_state.max_history_turns,
        key="max_turns_slider",
    )
    st.session_state.max_history_turns = max_turns

    st.markdown("---")
    st.markdown("## 📊 Session Stats")

    col1, col2 = st.columns(2)
    col1.metric("Questions", st.session_state.total_questions)
    col2.metric("History msgs", len(st.session_state.lc_history))

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.total_questions = 0
        st.rerun()

    st.markdown("---")
    # System status
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    db_ok = os.path.exists(PERSIST_DIRECTORY)
    st.markdown(
        f"**GROQ_API_KEY** {'<span class=\"badge-ok\">✓ found</span>' if groq_ok else '<span class=\"badge-err\">✗ missing</span>'}<br>"
        f"**Vector DB** {'<span class=\"badge-ok\">✓ found</span>' if db_ok else '<span class=\"badge-err\">✗ missing</span>'}",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4a5568;text-align:center'>"
        "Powered by LangChain · Groq · ChromaDB<br>HuggingFace Embeddings"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="rag-header">
  <h1>🤖 RAG Chatbot</h1>
  <p>History-aware retrieval · Groq LLM · ChromaDB</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# Initialise resources
if not st.session_state.ready:
    with st.spinner("Loading models and vector database…"):
        ensure_ready(st.session_state.chat_model)

if st.session_state.init_error:
    st.error(st.session_state.init_error)
    st.stop()

# ── Render existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) used"):
                for doc in msg["sources"]:
                    src = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:400].replace("\n", " ")
                    st.markdown(
                        f"**{os.path.basename(src)}**  \n"
                        f"<div class='source-block'>{preview}…</div>",
                        unsafe_allow_html=True,
                    )

        if msg["role"] == "assistant" and msg.get("search_q") and msg["search_q"] != msg.get("original_q"):
            st.caption(f"🔍 Searched as: *{msg['search_q']}*")


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant", avatar="🤖"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            "<span class='thinking-pill'>⏳ Thinking…</span>",
            unsafe_allow_html=True,
        )

        try:
            t0 = time.time()
            answer, docs, search_q = rag_answer(
                prompt,
                top_k=st.session_state.top_k,
                max_history_turns=st.session_state.max_history_turns,
            )
            elapsed = time.time() - t0
        except Exception as exc:
            thinking_placeholder.error(f"❌ Error: {exc}")
            st.stop()

        thinking_placeholder.empty()
        st.markdown(answer)

        # Sources expander
        if docs:
            with st.expander(f"📚 {len(docs)} source(s) used  ·  _{elapsed:.1f}s_"):
                for doc in docs:
                    src = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:400].replace("\n", " ")
                    st.markdown(
                        f"**{os.path.basename(src)}**  \n"
                        f"<div class='source-block'>{preview}…</div>",
                        unsafe_allow_html=True,
                    )

        if search_q != prompt:
            st.caption(f"🔍 Searched as: *{search_q}*")

    # Persist in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
        "search_q": search_q,
        "original_q": prompt,
    })
    st.session_state.total_questions += 1
