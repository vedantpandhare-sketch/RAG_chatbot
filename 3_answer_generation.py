import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="bge-m3")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.5  # Only return chunks with high similarity
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

IMPORTANT RULES:
1. ONLY use information from the provided documents above.
2. Do NOT make up, infer, or assume any information not in the documents.
3. If the answer is not explicitly in the documents, you MUST say: "I don't have enough information to answer that question based on the provided documents."
4. Be precise and literal with the facts."""

# Create a local Ollama chat model
model = ChatOllama(model="qwen2.5:3b-instruct", temperature=0)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a strict, factual assistant that answers questions based ONLY on provided documents. You will NOT invent, hallucinate, or infer information beyond what is explicitly stated. You MUST refuse to answer if the document doesn't contain the answer."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)


