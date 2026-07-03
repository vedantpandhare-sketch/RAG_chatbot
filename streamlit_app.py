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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import traceable

from rag_hybrid import (
    HybridRetriever,
    build_marathi_rewrite_prompt,
    extractive_marathi_answer_strict,
)

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
EMBEDDING_MODEL = "bge-m3"
DEFAULT_CHAT_MODEL = "llama2:latest"
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # Lower for multilingual (Marathi) text matching
AVAILABLE_MODELS = [
    "llama2:latest",
    "qwen2.5:3b-instruct",
    "llama3.2:latest",
    "qwen3:latest",
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
        "hybrid_index": None,
        "top_k": DEFAULT_TOP_K,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
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
    """Load embeddings, vector store, and local Ollama model once per session."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    model = ChatOllama(model=chat_model, temperature=0, num_predict=256)
    return embeddings, db, model


def ollama_up() -> bool:
    """Return True if a local Ollama server is reachable."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_ready(chat_model: str) -> bool:
    """Initialise resources if not already done. Returns True if OK."""
    if not ollama_up():
        st.session_state.init_error = (
            "🦙 Ollama server not reachable at `localhost:11434`. "
            "Start Ollama and make sure the models are pulled."
        )
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
        st.session_state.hybrid_index = HybridRetriever.from_vector_store(db)
        st.session_state.ready = True
        st.session_state.init_error = None
        return True
    except Exception as exc:
        st.session_state.init_error = f"⚠️ Failed to initialise: {exc}"
        st.session_state.ready = False
        return False


# ── Core RAG logic ────────────────────────────────────────────────────────────
@traceable(run_type="chain", name="rag_answer")
def rag_answer(user_question: str, top_k: int, max_history_turns: int):
    """Run history-aware retrieval + generation. Returns (answer, docs, search_q)."""
    db: Chroma = st.session_state.db
    model: ChatOllama = st.session_state.model
    hybrid_index: HybridRetriever = st.session_state.hybrid_index
    lc_history: list = st.session_state.lc_history

    # Step 1 — Rewrite question as standalone if history exists
    if False and lc_history:
        conversation_history = "\n".join(
            f"वापरकर्ता: {m.content}" if isinstance(m, HumanMessage) else f"सहाय्यक: {m.content}"
            for m in lc_history[-max_history_turns:]
        )
        rewrite_msgs = [
            SystemMessage(
                content="तुम्ही फक्त प्रश्न पुनर्लेखन करणारे सहाय्यक आहात. उत्तरात फक्त पुनर्लिखित प्रश्न द्या."
            ),
            HumanMessage(content=build_marathi_rewrite_prompt(user_question, conversation_history)),
        ] + lc_history[-max_history_turns:] + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        search_question = model.invoke(rewrite_msgs).content.strip()
    else:
        search_question = user_question

    # Step 2 — Hybrid retrieve + rerank
    docs, debug = hybrid_index.retrieve(
        db,
        search_question,
        top_k=top_k,
        semantic_k=max(12, top_k * 4),
        lexical_k=max(12, top_k * 6),
        min_score=0.12,
    )
    answer = extractive_marathi_answer_strict(user_question, docs)
    st.session_state.lc_history.append(HumanMessage(content=user_question))
    st.session_state.lc_history.append(AIMessage(content=answer))
    return answer, docs, search_question

    if debug:
        st.session_state.retrieval_debug = debug

    # Step 3 — Build context
    context = format_context(docs, max_chars_per_doc=1200)

    # Step 4 — Generate answer in Marathi
    combined_input = build_marathi_answer_prompt(user_question, context)
    answer_msgs = [
        SystemMessage(
            content=(
                "तू एक काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. "
                "फक्त दिलेल्या दस्तऐवजांवर आधारित उत्तर दे. "
                "उत्तर मराठीतच द्या."
            )
        ),
    ] + [
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
        "🧠 Local Model (Ollama)",
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

    similarity_threshold = st.slider(
        "🎯 Similarity threshold",
        min_value=0.0, max_value=1.0, step=0.05,
        value=st.session_state.similarity_threshold,
        key="similarity_slider",
        help="Higher = stricter matching (fewer but better results, faster response)"
    )
    st.session_state.similarity_threshold = similarity_threshold

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
    ollama_ok = ollama_up()
    db_ok = os.path.exists(PERSIST_DIRECTORY)
    st.markdown(
        f"**Ollama server** {'<span class=\"badge-ok\">✓ running</span>' if ollama_ok else '<span class=\"badge-err\">✗ offline</span>'}<br>"
        f"**Vector DB** {'<span class=\"badge-ok\">✓ found</span>' if db_ok else '<span class=\"badge-err\">✗ missing</span>'}",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4a5568;text-align:center'>"
        "Powered by LangChain · Ollama · ChromaDB<br>bge-m3 Embeddings"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="rag-header">
  <h1>🤖 RAG Chatbot</h1>
  <p>History-aware retrieval · Local Ollama LLM · ChromaDB</p>
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
@traceable(run_type="chain", name="rag_answer_v2")
def rag_answer(user_question: str, top_k: int, max_history_turns: int):
    """Run history-aware hybrid retrieval + Marathi answer generation."""
    db: Chroma = st.session_state.db
    model: ChatOllama = st.session_state.model
    hybrid_index: HybridRetriever = st.session_state.hybrid_index
    lc_history: list = st.session_state.lc_history

    if False and lc_history:
        conversation_history = "\n".join(
            f"वापरकर्ता: {m.content}" if isinstance(m, HumanMessage) else f"सहाय्यक: {m.content}"
            for m in lc_history[-max_history_turns:]
        )
        rewrite_msgs = [
            SystemMessage(content="तुम्ही फक्त प्रश्न पुनर्लेखन करणारे सहाय्यक आहात. फक्त पुनर्लिखित प्रश्न द्या."),
            HumanMessage(content=build_marathi_rewrite_prompt(user_question, conversation_history)),
        ]
        search_question = model.invoke(rewrite_msgs).content.strip() or user_question
    else:
        search_question = user_question

    docs, debug = hybrid_index.retrieve(
        db,
        search_question,
        top_k=top_k,
        semantic_k=max(12, top_k * 4),
        lexical_k=max(12, top_k * 6),
        min_score=0.12,
    )
    st.session_state.retrieval_debug = debug
    answer = extractive_marathi_answer_strict(user_question, docs)
    st.session_state.lc_history.append(HumanMessage(content=user_question))
    st.session_state.lc_history.append(AIMessage(content=answer))
    return answer, docs, search_question

    context = format_context(docs, max_chars_per_doc=1200)
    combined_input = build_marathi_answer_prompt(user_question, context)
    answer_msgs = [
        SystemMessage(content="तू काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. उत्तर मराठीतच द्या."),
        HumanMessage(content=combined_input),
    ]
    answer = model.invoke(answer_msgs).content.strip()
    ok, _reason = validate_marathi_answer(answer, context)
    if not ok:
        answer = refusal_message()

    st.session_state.lc_history.append(HumanMessage(content=user_question))
    st.session_state.lc_history.append(AIMessage(content=answer))
    return answer, docs, search_question


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
