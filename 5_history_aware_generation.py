import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Constants (aligned with the rest of the project)
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:3b-instruct"
TOP_K = 2
MAX_HISTORY_TURNS = 6  # keep last 6 messages (3 turns)

# Connect to your document database
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# Set up AI model
model = ChatOllama(model=CHAT_MODEL, temperature=0)

# Store our conversation as messages
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question to be "
                    "standalone and searchable. Just return the rewritten question, "
                    "nothing else."
                )
            ),
        ] + chat_history[-MAX_HISTORY_TURNS:] + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        source = doc.metadata.get("source", "unknown source")
        print(f"  Doc {i} [{source}]: {preview}...")

    # Step 3: Create final prompt
    context_blocks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown source")
        context_blocks.append(f"[Document {idx} | {source}]\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    combined_input = (
        f"Based on the following documents, please answer this question: {user_question}\n\n"
        f"Documents:\n{context}\n\n"
        "Please provide a clear, helpful answer using only the information from these "
        'documents. If you can\'t find the answer in the documents, say '
        '"I don\'t have enough information to answer that question based on the provided documents."'
    )

    # Step 4: Get the answer
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on provided documents and conversation history."
        ),
    ] + chat_history[-MAX_HISTORY_TURNS:] + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content.strip()

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