import os
import sys
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.1-8b-instant"
TOP_K = 4
MAX_HISTORY_TURNS = 3


def build_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )


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


def answer_question(question, retriever, model, history):
    docs = retriever.invoke(question)
    context = format_context(docs)
    conversation_history = format_history(history)

    prompt = f"""Use the retrieved documents to answer the user's question.

Rules:
- Answer only from the retrieved document context.
- If the answer is not in the context, say you do not have enough information in the documents.
- Keep the answer concise and conversational.
- Use the previous conversation only to understand follow-up questions.

Previous conversation:
{conversation_history}

Retrieved context:
{context}

User question:
{question}
"""

    messages = [
        SystemMessage(content="You are a helpful RAG chatbot for company documents."),
        HumanMessage(content=prompt),
    ]

    result = model.invoke(messages)
    sources = sorted({doc.metadata.get("source", "unknown source") for doc in docs})
    return result.content.strip(), sources


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is missing. Add it to your .env file first.")
        return

    print("Starting RAG terminal chatbot...")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    try:
        db = load_vector_store()
    except Exception as exc:
        print(f"Could not load the vector database: {exc}")
        print("Try running: .\\venv\\Scripts\\python.exe .\\1_ingestion_pipeline.py --rebuild")
        return

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    model = ChatGroq(model=CHAT_MODEL, temperature=0)
    history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            answer, sources = answer_question(question, retriever, model, history)
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
