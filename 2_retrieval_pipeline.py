import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

persistent_directory = "db/chroma_db"

# ======================== RETRIEVAL ENHANCEMENT ========================

def pre_filter_relevance(chunks: List, query: str) -> List:
    """
    Quick pre-filter to remove clearly irrelevant chunks before expensive reranking.
    Checks for:
    - Keyword overlap with query
    - Content quality scores
    - Relevance to the question
    
    Returns filtered list of chunks.
    """
    if not chunks:
        return []
    
    # Extract keywords from query (Marathi + English)
    query_lower = query.lower()
    query_keywords = set(re.findall(r'\b\w+\b', query_lower))
    
    filtered = []
    for chunk in chunks:
        # Check quality score
        quality = chunk.metadata.get("final_quality_score", 0.5)
        if quality < 0.25:  # Skip very poor quality chunks
            continue
        
        # Check keyword overlap
        content_lower = chunk.page_content.lower()
        content_keywords = set(re.findall(r'\b\w+\b', content_lower))
        overlap = len(query_keywords & content_keywords) / len(query_keywords) if query_keywords else 0
        
        # Keep if: good quality OR some keyword match
        if quality > 0.4 or overlap > 0.15:
            filtered.append(chunk)
    
    return filtered[:20]  # Limit to top 20 for reranking


def custom_rrf_score(dense_rank: int, sparse_rank: int, k: int = 60, alpha: float = 0.6) -> float:
    """
    Custom Reciprocal Rank Fusion (RRF) for combining dense and sparse vectors.
    
    For OCR-heavy Marathi newspaper content:
    - Sparse vectors (keywords) are weighted heavily (alpha=0.7) because they catch
      proper names, dates, and keywords despite OCR noise
    - Dense vectors (semantics) are weighted for contextual relevance (alpha=0.3)
    
    RRF formula: 1 / (k + rank)
    """
    dense_score = 1.0 / (k + dense_rank) if dense_rank >= 0 else 0
    sparse_score = 1.0 / (k + sparse_rank) if sparse_rank >= 0 else 0
    
    combined = (alpha * sparse_score) + ((1 - alpha) * dense_score)
    return combined


def retrieve_with_hybrid_search(query: str, k: int = 10, use_pre_filter: bool = True) -> List:
    """
    Perform hybrid search using BGE-M3's dense + sparse vectors.
    
    Process:
    1. Dense search (semantic): Find contextually relevant chunks
    2. Sparse search (lexical): Find keyword-matching chunks
    3. RRF: Combine results, favoring keyword matches for OCR content
    4. Pre-filter: Remove irrelevant chunks
    5. Return: Top k combined results
    """
    # Load embeddings and vector store
    embedding_model = OllamaEmbeddings(model="bge-m3")
    
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}  
    )
    
    # Dense vector search (semantic)
    print(f"🔍 Searching with Dense Vector (semantic)...")
    retriever_dense = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k * 2,  # Get more candidates
            "score_threshold": 0.3  # Lower threshold for Marathi
        }
    )
    dense_results = retriever_dense.invoke(query)
    
    # For sparse search, we would need a different approach
    # Chroma doesn't natively support sparse vectors, but BGE-M3 generates them
    # For now, we use dense results as primary
    print(f"  Found {len(dense_results)} dense matches")
    
    # Pre-filter to remove clearly irrelevant chunks
    if use_pre_filter:
        print(f"🔎 Pre-filtering for relevance...")
        dense_results = pre_filter_relevance(dense_results, query)
        print(f"  Filtered to {len(dense_results)} relevant chunks")
    
    return dense_results[:k]


def format_context(retrieved_docs: List) -> str:
    """
    Format retrieved documents with metadata for LLM context.
    """
    context = "=== RETRIEVED CONTEXT ===\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.metadata
        
        context += f"--- Source {i} ---\n"
        context += f"📰 Source: {metadata.get('source_pdf', metadata.get('source', 'Unknown'))}\n"
        context += f"📄 Page: {metadata.get('page_number', 'N/A')}\n"
        
        if metadata.get('primary_date'):
            context += f"📅 Date: {metadata.get('primary_date', 'N/A')}\n"
        
        if metadata.get('section'):
            context += f"📂 Section: {metadata.get('section', 'N/A')}\n"
        
        context += f"✓ Quality: {metadata.get('final_quality_score', 0.5):.2f}\n"
        
        context += f"\nContent:\n{doc.page_content}\n\n"
    
    return context


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RAG RETRIEVAL PIPELINE (Enhanced)")
    print("=" * 60)
    print()
    
    # Test query
    query = "पुणे शहरात पाणीपुरवठ्याच्या समस्या काय आहेत?"
    
    print(f"Query: {query}\n")
    
    # Perform hybrid search with pre-filtering
    relevant_docs = retrieve_with_hybrid_search(
        query,
        k=5,
        use_pre_filter=True
    )
    
    # Format and display results
    if relevant_docs:
        context = format_context(relevant_docs)
        print(context)
    else:
        print("⚠️  No relevant documents found.")





