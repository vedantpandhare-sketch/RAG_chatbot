"""
Parallel ingestion pipeline (faster rebuild of the Chroma vector store).

Why this exists
---------------
`1_ingestion_pipeline.py` embeds one batch at a time — the CPU sits idle during
each HTTP round-trip to Ollama and doesn't always fill all cores. This script
embeds several batches concurrently with a thread pool, then writes the results
to Chroma serially (ChromaDB / SQLite writes must not run in parallel).

Realistic speedup on a CPU-only box: ~1.3-1.8x. bge-m3 forward passes are the
real bottleneck; concurrent requests share the same cores, so it is NOT linear.
For a bigger win you need OLLAMA_NUM_PARALLEL set on the server (see below),
a smaller embedding model, or a GPU.

Enable server-side concurrency (PowerShell, then restart Ollama):
    setx OLLAMA_NUM_PARALLEL 4
    # restart the Ollama app so it picks up the env var

Run:
    .\\venv\\Scripts\\python.exe ingest_parallel.py            # build if missing
    .\\venv\\Scripts\\python.exe ingest_parallel.py --rebuild  # wipe and rebuild
"""

import importlib.util
import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PERSIST_DIRECTORY = "db/chroma_db"
DOCS_PATH = "docs"
EMBEDDING_MODEL = "bge-m3"
BATCH_SIZE = 64        # chunks per Ollama embed request (smaller = safer for the runner)
MAX_WORKERS = 4        # concurrent embed requests; match OLLAMA_NUM_PARALLEL on the server


# ── Reuse the loaders from 1_ingestion_pipeline.py (module name starts with a
#    digit, so import it by file path instead of a normal import). ─────────────
def _load_ingest_module():
    spec = importlib.util.spec_from_file_location("ingest_mod", "1_ingestion_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_metadata(meta):
    """ChromaDB rejects None metadata values — drop them."""
    return {k: v for k, v in meta.items() if v is not None}


def embed_batch(embeddings, index, batch):
    """Embed one batch of chunks. Runs in a worker thread."""
    texts = [chunk.page_content for chunk in batch]
    vectors = embeddings.embed_documents(texts)
    return index, batch, vectors


def main():
    rebuild = "--rebuild" in sys.argv

    if rebuild and os.path.exists(PERSIST_DIRECTORY):
        print("Rebuilding: removing existing vector store...")
        shutil.rmtree(PERSIST_DIRECTORY)

    if os.path.exists(PERSIST_DIRECTORY) and not rebuild:
        print(f"[!] {PERSIST_DIRECTORY} already exists. Use --rebuild to recreate it.")
        return

    ingest = _load_ingest_module()

    print("=== Parallel RAG Ingestion ===\n")
    documents = ingest.load_documents(DOCS_PATH)
    chunks = ingest.split_documents(documents)
    total = len(chunks)
    print(f"\nTotal chunks to embed: {total}")

    # Split into batches
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"Batches: {len(batches)} x {BATCH_SIZE}  |  workers: {MAX_WORKERS}\n")

    # httpx.Client (inside OllamaEmbeddings) is thread-safe, so one instance is
    # shared across worker threads.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Empty store — written to serially after each batch finishes embedding.
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )
    collection = vectorstore._collection

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(embed_batch, embeddings, i, batch)
            for i, batch in enumerate(batches)
        ]
        # Consume as each embed finishes; write to Chroma on the main thread only.
        for future in as_completed(futures):
            _, batch, vectors = future.result()
            collection.add(
                ids=[str(uuid.uuid4()) for _ in batch],
                embeddings=vectors,
                documents=[c.page_content for c in batch],
                metadatas=[_clean_metadata(c.metadata) for c in batch],
            )
            done += len(batch)
            print(f"  Embedded + stored {done}/{total} chunks")

    print(f"\n[OK] Done. Vector store has {collection.count()} chunks at {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()
