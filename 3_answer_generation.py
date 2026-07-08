import sys
import os
import importlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Import retrieval pipeline and utility helpers
retrieval_pipeline = importlib.import_module("2_retrieval_pipeline")
from rag_utils import (
    extractive_marathi_answer_strict,
    build_marathi_answer_prompt,
    validate_marathi_answer,
    refusal_message,
)

load_dotenv()

persistent_directory = "db/chroma_db"

if not os.path.exists(persistent_directory):
    print(f"Error: Vector database not found at {persistent_directory}. Run 1_ingestion_pipeline.py first.")
    sys.exit(1)

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="bge-m3")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "पुणे शहरात पाणीपुरवठ्याच्या समस्या काय आहेत?"

# Use the retrieve_with_hybrid_search function from retrieval pipeline
relevant_docs = retrieval_pipeline.retrieve_with_hybrid_search(
    query,
    k=3,
    use_pre_filter=True,
    db=db
)

print(f"User Query: {query}")
# Display results
print("\n--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} (Source: {doc.metadata.get('source_pdf', doc.metadata.get('source', 'Unknown'))}):\n{doc.page_content}\n")

# Try extractive answer first
answer = extractive_marathi_answer_strict(query, relevant_docs)

# If extractive search didn't find clear answers, generate using Ollama
if answer == "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही.":
    print("\n[Extractive search found no sufficient match. Falling back to Ollama generation...]")
    
    # Format the context for the LLM
    context = retrieval_pipeline.format_context(relevant_docs)
    combined_input = build_marathi_answer_prompt(query, context)
    
    # Create local Ollama chat model
    model = ChatOllama(model="qwen2.5:3b-instruct", temperature=0, num_predict=256)
    
    # Define messages for the model
    messages = [
        SystemMessage(content="तू काटेकोर दस्तऐवज-आधारित सहाय्यक आहेस. उत्तर मराठीतच द्या."),
        HumanMessage(content=combined_input),
    ]
    
    # Invoke model
    try:
        result = model.invoke(messages)
        answer = result.content.strip()
        
        # Validate the answer
        ok, reason = validate_marathi_answer(answer, context)
        if not ok:
            print(f"[Validation Failed: {reason}]")
            answer = refusal_message()
    except Exception as e:
        print(f"[Ollama generation error: {e}]")
        answer = refusal_message()

# Display the generated response
print("\n--- Generated Response ---")
print("Content only:")
print(answer)
