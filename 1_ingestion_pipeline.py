import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def load_json_pages(path):
    """Load a page-structured JSON file into one Document per page.

    Expects the shape produced by the OCR extractor:
        {"source_pdf": ..., "pages": [{"page_number": 1, "text": "..."}, ...]}
    Empty / whitespace-only pages are skipped.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source_name = data.get("source_pdf", str(path))
    docs = []
    for page in data.get("pages", []):
        text = (page.get("text") or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "source_pdf": source_name,
                    "page_number": page.get("page_number"),
                },
            )
        )
    print(f"  Loaded {len(docs)} pages from {path}")
    return docs


def load_documents(docs_path="Docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    documents = []

    # Plain text files -> one Document each.
    for path in sorted(Path(docs_path).glob("*.txt")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8-sig"),
                metadata={"source": str(path)},
            )
        )

    # JSON files (e.g. OCR-extracted newspaper) -> one Document per page.
    for path in sorted(Path(docs_path).glob("*.json")):
        documents.extend(load_json_pages(path))

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt or .json files found in {docs_path}. Please add your documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    # Recursive splitter breaks down to the character level, so chunks actually
    # respect chunk_size (CharacterTextSplitter can't cut below paragraph breaks
    # and produced oversized 1500+ char chunks -> bloated prompts -> slow prefill).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=100):
    """Create and persist ChromaDB vector store.

    Embeddings are added in small batches so the local Ollama embedding
    runner is never handed a huge payload at once (a single giant request
    crashes the runner on low-RAM CPU-only machines).
    """
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="bge-m3")

    # Create an empty ChromaDB vector store, then fill it batch by batch.
    print("--- Creating vector store ---")
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    with tqdm(total=total, unit="chunk", desc="Embedding", ncols=80) as bar:
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            vectorstore.add_documents(batch)
            bar.update(len(batch))

    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    rebuild = "--rebuild" in sys.argv

    if rebuild and os.path.exists(persistent_directory):
        print("Rebuilding vector store from documents...")
        shutil.rmtree(persistent_directory)

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("[OK] Vector store already exists. No need to re-process documents.")

        embedding_model = OllamaEmbeddings(model="bge-m3")
        try:
            vectorstore = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
            return vectorstore
        except Exception as exc:
            raise RuntimeError(
                "The existing vector store could not be loaded. "
                "Run this script again with --rebuild to recreate it."
            ) from exc

    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path)  

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n[OK] Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()




# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]



