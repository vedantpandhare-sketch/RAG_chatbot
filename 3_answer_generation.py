import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rag_hybrid import (
    HybridRetriever,
    extractive_marathi_answer_strict,
)


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

hybrid_index = HybridRetriever.from_vector_store(db)
relevant_docs, debug = hybrid_index.retrieve(
    db,
    query,
    top_k=3,
    semantic_k=12,
    lexical_k=12,
    min_score=0.12,
)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

answer = extractive_marathi_answer_strict(query, relevant_docs)
print("\n--- Generated Response ---")
print("Content only:")
print(answer)
sys.exit(0)

# Combine the query and the relevant document contents
combined_input = build_marathi_answer_prompt(query, format_context(relevant_docs))

# Create a local Ollama chat model
model = ChatOllama(model="llama2:latest", temperature=0, num_predict=256)

# Define the messages for the model
messages = [
    SystemMessage(content=(
        "तू काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. उत्तर मराठीतच द्या."
    )),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)
answer = result.content.strip()
ok, _reason = validate_marathi_answer(answer, format_context(relevant_docs))
if not ok:
    answer = refusal_message()

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(answer)


