import os
import sys
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from rag_hybrid import (
    HybridRetriever,
    build_marathi_rewrite_prompt,
    extractive_marathi_answer_strict,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "bge-m3"
CHAT_MODEL = "llama2:latest"
TOP_K = 5
MAX_HISTORY_TURNS = 3


def build_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def load_vector_store():
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector database not found at {PERSIST_DIRECTORY}. Run 1_ingestion_pipeline.py first."
        )

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=build_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def format_context(docs):
    context_blocks = []
    for index, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown source")
        context_blocks.append(f"[Document {index} | {source}]\n{doc.page_content}")
    return "\n\n".join(context_blocks)


def format_history(history: List[Tuple[str, str]]):
    if not history:
        return "No previous conversation."

    recent_history = history[-MAX_HISTORY_TURNS:]
    return "\n".join(
        f"User: {question}\nAssistant: {answer}" for question, answer in recent_history
    )


@traceable(run_type="chain", name="rag_answer_question")
def answer_question(question, retriever, model, history, db):
    search_question = question
    if False and history:
        rewrite_prompt = build_marathi_rewrite_prompt(question, format_history(history))
        rewrite_messages = [
            SystemMessage(content="तुम्ही फक्त प्रश्न पुनर्लेखन करणारे सहाय्यक आहात. फक्त पुनर्लिखित प्रश्न द्या."),
            HumanMessage(content=rewrite_prompt),
        ]
        search_question = model.invoke(rewrite_messages).content.strip() or question

    docs, _ = retriever.retrieve(
        db,
        search_question,
        top_k=TOP_K,
        semantic_k=max(12, TOP_K * 4),
        lexical_k=max(12, TOP_K * 6),
        min_score=0.12,
    )
    answer = extractive_marathi_answer_strict(question, docs)
    sources = sorted({doc.metadata.get("source", "unknown source") for doc in docs})
    return answer, sources
    context = hybrid_format_context(docs, max_chars_per_doc=1200)

    prompt = build_marathi_answer_prompt(question, context)
    messages = [
        SystemMessage(content="तू काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. उत्तर मराठीतच द्या."),
        HumanMessage(content=prompt),
    ]

    result = model.invoke(messages)
    answer = result.content.strip()
    ok, _reason = validate_marathi_answer(answer, context)
    if not ok:
        answer = refusal_message()
    sources = sorted({doc.metadata.get("source", "unknown source") for doc in docs})
    return answer, sources


def main():
    print("Starting RAG terminal chatbot...")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    try:
        db = load_vector_store()
    except Exception as exc:
        print(f"Could not load the vector database: {exc}")
        print("Try running: .\\venv\\Scripts\\python.exe .\\1_ingestion_pipeline.py --rebuild")
        return

    retriever = HybridRetriever.from_vector_store(db)
    model = ChatOllama(model=CHAT_MODEL, temperature=0, num_predict=256)
    history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            answer, sources = answer_question(question, retriever, model, history, db)
        except Exception as exc:
            print(f"Bot: Sorry, I ran into an error: {exc}\n")
            continue

        print(f"\nBot: {answer}")
        if sources:
            print("Sources: " + ", ".join(sources))
        print()

        history.append((question, answer))


if __name__ == "__main__":
    main()
