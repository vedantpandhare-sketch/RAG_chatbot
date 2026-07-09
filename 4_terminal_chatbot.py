"""
4_terminal_chatbot.py
---------------------

History-aware RAG chatbot using

• ChromaDB
• BGE-M3 Embeddings
• Ollama
• Qwen2.5
• LangSmith

Optimized for Marathi document QA.
"""

import os
import sys
import importlib.util

from dotenv import load_dotenv

from langsmith import traceable

from langchain_chroma import Chroma

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

# ------------------------------------------------------------------------------
# Load Retrieval Pipeline
# ------------------------------------------------------------------------------

_here = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "retrieval_pipeline",
    os.path.join(
        _here,
        "2_retrieval_pipeline.py",
    ),
)

retrieval_pipeline = importlib.util.module_from_spec(_spec)

_spec.loader.exec_module(retrieval_pipeline)

# ------------------------------------------------------------------------------
# RAG Utilities
# ------------------------------------------------------------------------------

from rag_utils import (
    SYSTEM_PROMPT,
    build_marathi_answer_prompt,
    build_marathi_rewrite_prompt,
    refusal_message,
    validate_marathi_answer,
)

# ------------------------------------------------------------------------------
# Terminal
# ------------------------------------------------------------------------------

sys.stdout.reconfigure(
    encoding="utf-8",
    errors="replace",
)

load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================

PERSIST_DIRECTORY = "db/chroma_db"

EMBEDDING_MODEL = "bge-m3"

CHAT_MODEL = "qwen2.5:7b-instruct"

TOP_K = 4

MAX_HISTORY_TURNS = 3

TEMPERATURE = 0.3

NUM_PREDICT = 384

TOP_P = 0.9

REPEAT_PENALTY = 1.1

# ==============================================================================
# Vector Store
# ==============================================================================

def load_vector_store() -> Chroma:

    if not os.path.exists(PERSIST_DIRECTORY):

        raise FileNotFoundError(
            f"Vector DB not found:\n{PERSIST_DIRECTORY}\n\n"
            "Run the ingestion pipeline first."
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

# ==============================================================================
# Conversation History
# ==============================================================================

def format_history(history):

    if not history:
        return ""

    recent = history[-MAX_HISTORY_TURNS:]

    conversation = []

    for question, answer in recent:

        conversation.append(
            f"वापरकर्ता: {question}"
        )

        conversation.append(
            f"सहाय्यक: {answer}"
        )

    return "\n".join(conversation)

# ==============================================================================
# Core RAG Pipeline
# ==============================================================================

@traceable(
    run_type="chain",
    name="rag_answer_terminal",
)
def answer_question(
    question: str,
    model: ChatOllama,
    history: list,
    db: Chroma,
):
    """
    RAG Pipeline

    User Question
            ↓
    Query Rewrite (optional)
            ↓
    Hybrid Retrieval
            ↓
    Context Construction
            ↓
    Qwen Generation
            ↓
    Marathi Validation
    """

    search_query = question

    # ------------------------------------------------------------------
    # Step 1 : Rewrite follow-up question
    # ------------------------------------------------------------------

    if history:

        try:

            rewrite_prompt = build_marathi_rewrite_prompt(
                question,
                format_history(history),
            )

            rewritten = model.invoke(
                [
                    SystemMessage(
                        content=(
                            "तुम्ही शोध प्रणालीसाठी प्रश्न पुन्हा लिहिणारे सहाय्यक आहात. "
                            "फक्त पुनर्लिखित प्रश्न द्या."
                        )
                    ),
                    HumanMessage(content=rewrite_prompt),
                ]
            ).content.strip()

            if rewritten:
                search_query = rewritten

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 2 : Retrieve documents
    # ------------------------------------------------------------------

    docs = retrieval_pipeline.retrieve_with_hybrid_search(
        search_query,
        k=TOP_K,
        use_pre_filter=True,
        db=db,
        verbose=False,
    )

    if not docs:
        return refusal_message(), []

    # ------------------------------------------------------------------
    # Step 3 : Build Context
    # ------------------------------------------------------------------

    context = retrieval_pipeline.format_context(docs)

    prompt = build_marathi_answer_prompt(
        question,
        context,
    )

    # ------------------------------------------------------------------
    # Step 4 : Generate Answer
    # ------------------------------------------------------------------

    try:

        result = model.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
        )

        answer = result.content.strip()

    except Exception as exc:

        return (
            f"उत्तर तयार करताना त्रुटी आली:\n{exc}",
            [],
        )

    # ------------------------------------------------------------------
    # Step 5 : Validate Marathi
    # ------------------------------------------------------------------

    ok, reason = validate_marathi_answer(
        answer,
        context,
    )

    if not ok:
        return refusal_message(), []

    # ------------------------------------------------------------------
    # Step 6 : Collect Sources
    # ------------------------------------------------------------------

    sources = sorted(
        {
            os.path.basename(
                doc.metadata.get(
                    "source_pdf",
                    doc.metadata.get(
                        "source",
                        "Unknown",
                    ),
                )
            )
            for doc in docs
        }
    )

    return answer, sources

# ==============================================================================
# Terminal Chat
# ==============================================================================

def main():

    print("=" * 70)
    print("🇮🇳 Marathi RAG Chatbot")
    print("=" * 70)
    print(f"Model      : {CHAT_MODEL}")
    print(f"Embeddings : {EMBEDDING_MODEL}")
    print(f"Retriever K: {TOP_K}")
    print("=" * 70)
    print("Type 'exit' to quit.\n")

    try:
        db = load_vector_store()

    except Exception as exc:

        print(exc)
        return

    model = ChatOllama(

        model=CHAT_MODEL,

        temperature=TEMPERATURE,

        num_predict=NUM_PREDICT,

        top_p=TOP_P,

        repeat_penalty=REPEAT_PENALTY,
    )

    history = []

    while True:

        try:

            question = input("\n You : ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\n\nGoodbye ")
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        try:

            answer, sources = answer_question(
                question,
                model,
                history,
                db,
            )

        except Exception as exc:

            print("\n Error")
            print(exc)

            continue

        print("\n Bot\n")

        print(answer)

        if sources:

            print("\n📄 Sources")

            for source in sources:

                print(f"   • {source}")

        history.append(
            (
                question,
                answer,
            )
        )

        if len(history) > MAX_HISTORY_TURNS:

            history = history[-MAX_HISTORY_TURNS:]

        print("\n" + "-" * 70)


if __name__ == "__main__":

    main()
