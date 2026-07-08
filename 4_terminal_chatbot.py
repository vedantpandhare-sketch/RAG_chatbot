"""
4_terminal_chatbot.py — RAG Terminal Chatbot
---------------------------------------------
Multi-turn CLI chatbot with history-aware retrieval and abstractive
Marathi answer generation via a local Ollama model.

Usage:
    .\\venv\\Scripts\\python.exe 4_terminal_chatbot.py
"""

import os
import sys
import importlib.util

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# ── Import 2_retrieval_pipeline (numeric prefix, use spec loader) ─────────────
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "retrieval_pipeline",
    os.path.join(_here, "2_retrieval_pipeline.py"),
)
retrieval_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval_pipeline)

from rag_utils import (
    build_marathi_rewrite_prompt,
    build_marathi_answer_prompt,
    validate_marathi_answer,
    refusal_message,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL   = "bge-m3"
# llama3.2 / qwen2.5:3b-instruct generate Marathi much better than llama2.
# Change this to whichever model you have pulled with Ollama.
CHAT_MODEL        = "qwen2.5:3b-instruct"
TOP_K             = 5
MAX_HISTORY_TURNS = 3


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_vector_store() -> Chroma:
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector database not found at {PERSIST_DIRECTORY}. "
            "Run 1_ingestion_pipeline.py first."
        )
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    recent = history[-MAX_HISTORY_TURNS:]
    return "\n".join(
        f"वापरकर्ता: {q}\nसहाय्यक: {a}" for q, a in recent
    )


# ── Core RAG logic ─────────────────────────────────────────────────────────────
@traceable(run_type="chain", name="rag_answer_terminal")
def answer_question(question: str, model: ChatOllama, history: list, db: Chroma):
    """
    1. (Optional) Rewrite the question using history for better retrieval.
    2. Retrieve the top-k relevant document chunks (silently).
    3. Generate an abstractive Marathi answer using the local LLM.
    4. Validate that the answer is in Marathi; fall back to a refusal if not.
    """
    search_question = question

    # Step 1 — Silent history-aware query rewriting
    if history:
        rewrite_prompt = build_marathi_rewrite_prompt(question, _format_history(history))
        try:
            rewritten = model.invoke([
                SystemMessage(content=(
                    "तुम्ही फक्त प्रश्न पुनर्लेखन करणारे सहाय्यक आहात. "
                    "फक्त पुनर्लिखित प्रश्न द्या, इतर काही नाही."
                )),
                HumanMessage(content=rewrite_prompt),
            ]).content.strip()
            if rewritten:
                search_question = rewritten
        except Exception:
            pass  # silently fall back to original question

    # Step 2 — Retrieve (verbose=False: no progress prints)
    docs = retrieval_pipeline.retrieve_with_hybrid_search(
        search_question,
        k=TOP_K,
        use_pre_filter=True,
        db=db,
        verbose=False,
    )

    if not docs:
        return refusal_message(), []

    # Step 3 — Build context and generate abstractive Marathi answer
    context = retrieval_pipeline.format_context(docs)
    prompt  = build_marathi_answer_prompt(question, context)

    try:
        result = model.invoke([
            SystemMessage(content=(
                "तू एक अचूक, दस्तऐवज-आधारित मराठी सहाय्यक आहेस. "
                "उत्तर नेहमी मराठी भाषेत आणि देवनागरी लिपीत दे. "
                "दिलेल्या दस्तऐवजांमधील माहितीवर आधारित स्वतःच्या शब्दांत संक्षिप्त उत्तर दे."
            )),
            HumanMessage(content=prompt),
        ])
        answer = result.content.strip()
    except Exception as exc:
        return f"[त्रुटी: {exc}]", []

    # Step 4 — Validate Marathi content; fall back if the model responded in English
    ok, _reason = validate_marathi_answer(answer, context)
    if not ok:
        answer = refusal_message()

    sources = sorted({
        os.path.basename(doc.metadata.get("source_pdf", doc.metadata.get("source", "unknown")))
        for doc in docs
    })
    return answer, sources


# ── Chat loop ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAG TERMINAL CHATBOT")
    print("=" * 60)
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    try:
        db = load_vector_store()
    except Exception as exc:
        print(f"Error: {exc}")
        print("Tip: run  python 1_ingestion_pipeline.py --rebuild")
        return

    model   = ChatOllama(model=CHAT_MODEL, temperature=0.2, num_predict=256)
    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            answer, sources = answer_question(question, model, history, db)
        except Exception as exc:
            print(f"Bot: Sorry, I ran into an error: {exc}\n")
            continue

        print(f"\nBot: {answer}")
        if sources:
            print(f"     📄 Sources: {', '.join(sources)}")
        print()

        history.append((question, answer))


if __name__ == "__main__":
    main()
