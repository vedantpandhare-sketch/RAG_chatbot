import os
import sys
import importlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import retrieval pipeline and utility helpers
retrieval_pipeline = importlib.import_module("2_retrieval_pipeline")
from rag_utils import (
    build_marathi_rewrite_prompt,
    build_marathi_answer_prompt,
    extractive_marathi_answer_strict,
    validate_marathi_answer,
    refusal_message,
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


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    search_question = user_question

    # Step 1: Make the question clear using conversation history
    if chat_history:
        history_text = "\n".join(
            f"वापरकर्ता: {msg.content}" if isinstance(msg, HumanMessage) else f"सहाय्यक: {msg.content}"
            for msg in chat_history[-MAX_HISTORY_TURNS:]
        )
        messages = [
            SystemMessage(content="तुम्ही फक्त प्रश्न पुनर्लेखन करणारे सहाय्यक आहात. फक्त पुनर्लिखित प्रश्न द्या."),
            HumanMessage(content=build_marathi_rewrite_prompt(user_question, history_text)),
        ]

        try:
            result = model.invoke(messages)
            rewritten = result.content.strip()
            if rewritten:
                search_question = rewritten
                print(f"Searching for: {search_question}")
        except Exception as exc:
            print(f"Failed to rewrite question using history: {exc}")

    # Step 2: Retrieval using retrieval pipeline
    docs = retrieval_pipeline.retrieve_with_hybrid_search(
        search_question,
        k=TOP_K,
        use_pre_filter=True,
        db=db
    )

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        source = doc.metadata.get("source_pdf", doc.metadata.get("source", "unknown source"))
        print(f"  Doc {i} [{os.path.basename(source)}]: {preview}...")

    # Step 3: Try extractive answer first
    answer = extractive_marathi_answer_strict(user_question, docs)

    # Step 4: Fallback to LLM if extractive search didn't find sufficient match
    if answer == "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही.":
        print("[Extractive search found no sufficient match. Falling back to LLM generation...]")
        
        # If no documents found with high similarity, inform user
        if not docs:
            print("⚠️ No documents found with sufficient similarity.")
            answer = refusal_message()
        else:
            context = retrieval_pipeline.format_context(docs)
            combined_input = build_marathi_answer_prompt(user_question, context)

            messages = [
                SystemMessage(content="तू काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. उत्तर मराठीतच द्या."),
                HumanMessage(content=combined_input)
            ]

            try:
                result = model.invoke(messages)
                gen_answer = result.content.strip()
                ok, _reason = validate_marathi_answer(gen_answer, context)
                if ok:
                    answer = gen_answer
                else:
                    print(f"[Validation Failed: {_reason}]")
                    answer = refusal_message()
            except Exception as exc:
                print(f"[LLM Generation error: {exc}]")
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
        try:
            question = input("Your question: ").strip()
        except EOFError:
            break

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
