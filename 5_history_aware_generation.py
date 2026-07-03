import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from rag_hybrid import (
    HybridRetriever,
    build_marathi_rewrite_prompt,
    extractive_marathi_answer_strict,
)

# Load environment variables
load_dotenv()

# Constants (aligned with the rest of the project)
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "bge-m3"
CHAT_MODEL = "llama2:latest"
TOP_K = 5  # Retrieve more candidates for multilingual text
SIMILARITY_THRESHOLD = 0.35  # Lower threshold for better Marathi matching
MAX_HISTORY_TURNS = 6  # keep last 6 messages (3 turns)

# Connect to your document database
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# Set up AI model
model = ChatOllama(model=CHAT_MODEL, temperature=0, num_predict=256)

# Store our conversation as messages
chat_history = []
hybrid_index = HybridRetriever.from_vector_store(db)


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if False and chat_history:
        history_text = "\n".join(
            f"वापरकर्ता: {msg.content}" if isinstance(msg, HumanMessage) else f"सहाय्यक: {msg.content}"
            for msg in chat_history[-MAX_HISTORY_TURNS:]
        )
        messages = [
            SystemMessage(content="तुम्ही फक्त प्रश्न पुनर्लेखन करणारे सहाय्यक आहात. फक्त पुनर्लिखित प्रश्न द्या."),
            HumanMessage(content=build_marathi_rewrite_prompt(user_question, history_text)),
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Hybrid retrieval + reranking
    docs, _ = hybrid_index.retrieve(
        db,
        search_question,
        top_k=TOP_K,
        semantic_k=max(12, TOP_K * 4),
        lexical_k=max(12, TOP_K * 6),
        min_score=0.12,
    )

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        source = doc.metadata.get("source", "unknown source")
        print(f"  Doc {i} [{source}]: {preview}...")

    answer = extractive_marathi_answer_strict(user_question, docs)
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    print(f"\nAnswer: {answer}")
    return answer

    # If no documents found with high similarity, inform user
    if not docs:
        print("⚠️  No documents found with sufficient similarity. The answer may not be reliable.")
        answer = "I don't have enough information to answer that question based on the provided documents."
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))
        return answer

    # Step 3: Create final prompt
    context = format_context(docs, max_chars_per_doc=1200)
    combined_input = build_marathi_answer_prompt(user_question, context)

    messages = [
        SystemMessage(content="तू काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. उत्तर मराठीतच द्या."),
    ] + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content.strip()
    ok, _reason = validate_marathi_answer(answer, context)
    if not ok:
        answer = refusal_message()

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer: {answer}")
    return answer


# Simple chat loop
def start_chat():
    if not os.path.exists(PERSIST_DIRECTORY):
        print(
            f"Vector database not found at '{PERSIST_DIRECTORY}'. "
            "Run 1_ingestion_pipeline.py first."
        )
        return

    print("History-Aware RAG Chatbot ready!")
    print("Ask me questions! Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            ask_question(question)
        except Exception as exc:
            print(f"Sorry, I ran into an error: {exc}\n")


if __name__ == "__main__":
    start_chat()
