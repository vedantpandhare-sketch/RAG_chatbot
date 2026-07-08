import json
import os
import shutil
import sys
import re
import unicodedata
from typing import List, Tuple, Dict
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ======================== EXPANDED NOISE PATTERNS ========================
NOISE_PATTERNS = [
    # Website/URL patterns
    r"www\.loksatta\.com",
    r"epaper\.loksatta\.com",
    r"https?://",
    
    # Contact/Legal patterns
    r"\bcontact\b",
    r"\breg\.?\s*no\.?\b",
    r"\bcivil\s*case\b",
    r"\bwrit\s*petition\b",
    r"\bpetition\s*no\.?\b",
    
    # Page markers
    r"\bpage\s*\d+\b",
    r"continued\s+on\s+page",
    r"see\s+page",
    
    # Advertisement keywords
    r"\badvertisement\b",
    r"\bad\s*space\b",
    r"\bfor\s+sale\b",
    r"\brental\b",
    r"\blodge\b",
    r"\brent\b",
]
# NOTE: Removed Marathi month pattern because legitimate news articles contain dates!
# The months are used in actual publication dates, not just ads

# OCR noise indicators (patterns that suggest bad OCR quality)
OCR_NOISE_INDICATORS = [
    (r"[्ंः़ँॆॊ]{2,}", 0.85),  # Multiple diacritics in a row (high weight)
    (r"[|\\\/\-\+\*]{3,}", 0.90),  # Symbol sequences
    (r"[0-9]{8,}", 0.70),  # Long number sequences
    (r"[\u0900-\u097F]{1}[a-z]{1}[\u0900-\u097F]{1}", 0.75),  # Mixed scripts rapidly
]

# Marathi months for date extraction
MARATHI_MONTHS = {
    "जनवरी": 1, "फेब्रुवारी": 2, "मार्च": 3, "एप्रिल": 4,
    "मे": 5, "जून": 6, "जुलै": 7, "ऑगस्ट": 8,
    "सप्टेंबर": 9, "ऑक्टोबर": 10, "नोव्हेंबर": 11, "डिसेंबर": 12
}

# Marathi weekdays
MARATHI_WEEKDAYS = [
    "सोमवार", "मंगळवार", "बुधवार", "गुरुवार", 
    "शुक्रवार", "शनिवार", "रविवार"
]


# ======================== DATE EXTRACTION FUNCTIONS ========================
def extract_marathi_date(text: str) -> Tuple[str, float]:
    """
    Extract Marathi format dates: 'सोमवार, १५ जून २०२६'
    Returns (ISO date string, confidence score)
    """
    # Pattern: optional weekday, day, month, year
    pattern = r"(?:सोमवार|मंगळवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)?[,:]?\s*(\d{1,2})\s+(" + "|".join(MARATHI_MONTHS.keys()) + r")\s+(\d{4})"
    
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        day, month_name, year = match.groups()
        try:
            month = MARATHI_MONTHS.get(month_name, 0)
            if 1 <= int(day) <= 31 and month >= 1 and 1900 <= int(year) <= 2100:
                iso_date = f"{year}-{month:02d}-{int(day):02d}"
                return iso_date, 0.95
        except:
            pass
    return None, 0.0

def extract_english_date(text: str) -> Tuple[str, float]:
    """
    Extract English format dates: '15 June 2026' or '15-06-2026'
    Returns (ISO date string, confidence score)
    """
    english_months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    # Pattern 1: '15 June 2026'
    pattern1 = r"(\d{1,2})\s+(" + "|".join(english_months.keys()) + r")\s+(\d{4})"
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        day, month_name, year = match.groups()
        try:
            month = english_months.get(month_name.lower(), 0)
            if 1 <= int(day) <= 31 and month >= 1 and 1900 <= int(year) <= 2100:
                iso_date = f"{year}-{month:02d}-{int(day):02d}"
                return iso_date, 0.92
        except:
            pass
    
    # Pattern 2: '15-06-2026' or '15/06/2026'
    pattern2 = r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})"
    match = re.search(pattern2, text)
    if match:
        day, month, year = match.groups()
        try:
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1900 <= int(year) <= 2100:
                iso_date = f"{year}-{int(month):02d}-{int(day):02d}"
                return iso_date, 0.85
        except:
            pass
    
    return None, 0.0

def extract_dates(text: str, page_metadata: Dict) -> Dict:
    """
    Extract all dates from text, preferring explicit dates over inferred ones.
    Returns metadata dict with extracted dates and confidence (flattened for ChromaDB compatibility).
    """
    dates_found = []
    
    # Try Marathi format
    marathi_date, marathi_conf = extract_marathi_date(text)
    if marathi_date:
        dates_found.append({"date": marathi_date, "confidence": marathi_conf, "format": "marathi"})
    
    # Try English format
    english_date, english_conf = extract_english_date(text)
    if english_date and english_date not in [d["date"] for d in dates_found]:
        dates_found.append({"date": english_date, "confidence": english_conf, "format": "english"})
    
    # Use extraction date from JSON metadata as fallback
    extraction_date = page_metadata.get("extraction_date")
    
    # Flatten for ChromaDB compatibility (no nested dicts/lists)
    result = {
        "primary_date": dates_found[0]["date"] if dates_found else extraction_date or "unknown",
        "date_confidence": float(dates_found[0]["confidence"]) if dates_found else 0.5,
        "date_format": dates_found[0]["format"] if dates_found else "none",
    }
    
    return result

# ======================== OCR QUALITY SCORING ========================
def calculate_ocr_quality_score(text: str) -> float:
    """
    Calculate OCR quality score (0.0 to 1.0) based on noise indicators.
    Lower score = more noise.
    """
    if not text:
        return 0.0
    
    noise_score = 0.0
    total_weight = len(OCR_NOISE_INDICATORS)
    
    for pattern, weight in OCR_NOISE_INDICATORS:
        if re.search(pattern, text):
            noise_score += weight
    
    # Normalize: higher score means lower quality (more noise)
    quality = 1.0 - (noise_score / (total_weight * 1.0))
    return max(0.0, min(1.0, quality))

# ======================== SECTION DETECTION ========================
def detect_section(text: str) -> str:
    """
    Detect newspaper section from text content.
    """
    text_lower = text.lower()
    
    sections = {
        "political": ["पोलिस", "सरकार", "राज्य", "विधायक", "लोकसभा", "विधानसभा", "चुनाव", "नेता"],
        "business": ["बाजार", "शेयर", "व्यापार", "कंपनी", "आर्थिक", "बैंक", "निवेश"],
        "sports": ["क्रीडा", "खेल", "फुटबॉल", "क्रिकेट", "खेळ", "विजय", "टीम"],
        "health": ["आरोग्य", "स्वास्थ्य", "चिकित्सा", "रोग", "डॉक्टर", "औषध"],
        "technology": ["तंत्रज्ञान", "कंप्यूटर", "मोबाइल", "ऐप", "साइबर"],
        "environment": ["पर्यावरण", "वन", "प्रदूषण", "जलवायु", "ग्रीन"],
        "legal": ["न्यायालय", "कानून", "मामला", "केस", "पेटीशन"],
    }
    
    for section, keywords in sections.items():
        if any(kw in text_lower for kw in keywords):
            return section
    
    return "general"

# ======================== ENHANCED OCR CLEANING ========================
def clean_ocr_text(text: str, min_line_length: 12) -> Tuple[str, float, Dict]:
    """
    Remove obvious OCR garbage while preserving page content.
    Returns (cleaned_text, quality_score, quality_metadata)
    """
    if not text:
        return "", 0.0, {"reason": "empty"}

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)

    cleaned_lines = []
    noise_count = 0
    total_lines = 0
    
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        
        total_lines += 1
        lower = line.lower()
        
        # Check against noise patterns
        if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in NOISE_PATTERNS):
            noise_count += 1
            continue

        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", line))
        latin_chars = len(re.findall(r"[A-Za-z]", line))
        other_chars = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
        total_alpha = devanagari_chars + latin_chars

        if len(line) < min_line_length and total_alpha == 0:
            noise_count += 1
            continue

        # Long lines with no Devanagari are usually ads/headers/garbage
        if len(line) > 25 and devanagari_chars == 0:
            if latin_chars > 0 or other_chars > 0:
                noise_count += 1
                continue

        if len(line) > 40 and total_alpha > 0:
            devanagari_ratio = devanagari_chars / max(1, total_alpha)
            symbol_ratio = other_chars / max(1, len(line))
            if devanagari_ratio < 0.2 and symbol_ratio > 0.35:
                noise_count += 1
                continue

        if other_chars > len(line) * 0.6 and devanagari_chars == 0 and latin_chars == 0:
            noise_count += 1
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip()
    
    # Calculate quality metrics
    noise_ratio = noise_count / total_lines if total_lines > 0 else 0.0
    base_quality = calculate_ocr_quality_score(cleaned)
    final_quality = base_quality * (1.0 - noise_ratio * 0.3)  # Penalize for filtered noise
    
    quality_metadata = {
        "total_lines_processed": total_lines,
        "noise_lines_removed": noise_count,
        "noise_ratio": noise_ratio,
        "ocr_quality_base": base_quality,
        "final_quality_score": final_quality,
    }
    
    return cleaned, final_quality, quality_metadata


# ======================== MULTI-LEVEL CHUNKING ========================
def create_multi_level_chunks(text: str, metadata: Dict) -> List[Document]:
    """
    Create parent-child chunks:
    - Small chunks (150-250 tokens): What gets embedded
    - Large chunks (500-750 tokens): Context for LLM
    
    Returns list of small chunks with parent reference.
    """
    # First split into medium chunks (article-level)
    text_splitter_medium = RecursiveCharacterTextSplitter(
        chunk_size=600,  # ~150-200 tokens in Marathi
        chunk_overlap=100,
        separators=["\n\n", "\n", "।", ".", "?", "!", " ", ""],
    )
    
    medium_chunks = text_splitter_medium.split_text(text)
    
    small_documents = []
    
    for chunk_idx, medium_chunk in enumerate(medium_chunks):
        # Further split medium chunks into small ones for embedding
        text_splitter_small = RecursiveCharacterTextSplitter(
            chunk_size=180,  # ~100-150 tokens
            chunk_overlap=30,
            separators=["\n\n", "\n", "।", ".", " ", ""],
        )
        
        small_chunks = text_splitter_small.split_text(medium_chunk)
        
        for small_idx, small_chunk in enumerate(small_chunks):
            # Create metadata for small chunk
            small_metadata = metadata.copy()
            small_metadata.update({
                "chunk_type": "small",  # What gets embedded
                "medium_chunk_index": chunk_idx,
                "small_chunk_index": small_idx,
                "parent_chunk": medium_chunk,  # Keep parent context
                "chunk_size": len(small_chunk),
            })
            
            small_documents.append(
                Document(
                    page_content=small_chunk,
                    metadata=small_metadata
                )
            )
    
    return small_documents


# ======================== PRE-FILTER QUALITY ========================
def pre_filter_quality(text: str, min_content_length: int = 50) -> bool:
    """
    Quick pre-filter to remove clearly bad chunks before embedding.
    Returns True if chunk should be kept.
    """
    if not text or len(text) < min_content_length:
        return False
    
    # Check for excessive symbols/numbers
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    if symbol_count / len(text) > 0.6:
        return False
    
    # Check for minimum content: at least 20% alphanumeric chars
    alpha_count = sum(1 for ch in text if ch.isalnum())
    if alpha_count / len(text) < 0.2:
        return False
    
    return True




def load_json_pages(path):
    """Load a page-structured JSON file with enhanced metadata extraction.

    Expects the shape produced by the OCR extractor:
        {"source_pdf": ..., "pages": [{"page_number": 1, "text": "..."}, ...]}
    
    Returns documents with enriched metadata including:
    - Extracted and standardized dates
    - OCR quality scores
    - Section detection
    - Multi-level chunking
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source_name = data.get("source_pdf", str(path))
    docs = []
    
    for page in data.get("pages", []):
        raw_text = page.get("text") or ""
        
        # Enhanced cleaning with quality metrics
        text, quality_score, quality_metadata = clean_ocr_text(raw_text, min_line_length=12)
        
        # Lowered threshold to 0.15 for newspaper OCR (which is often noisy)
        # This allows good content through while still filtering garbage
        if not text or quality_score < 0.15:
            continue
        
        # Base metadata
        base_metadata = {
            "source": str(path),
            "source_pdf": source_name,
            "page_number": page.get("page_number"),
            "extraction_date": data.get("extraction_date"),
        }
        
        # Extract dates with confidence scoring
        date_metadata = extract_dates(text, base_metadata)
        base_metadata.update(date_metadata)
        
        # Detect section
        section = detect_section(text)
        base_metadata["section"] = section
        
        # Add OCR quality metrics
        base_metadata.update(quality_metadata)
        
        # Create multi-level chunks (small for embedding, large for context)
        chunk_documents = create_multi_level_chunks(text, base_metadata)
        
        # Pre-filter chunks by quality
        filtered_chunks = [doc for doc in chunk_documents if pre_filter_quality(doc.page_content)]
        
        docs.extend(filtered_chunks)
    
    print(f"  Loaded {len(docs)} chunks from {path}")
    return docs


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory with enhanced processing"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    documents = []
    skipped_txt = []
    skipped_json = []

    # Plain text files -> chunks with basic metadata
    for path in sorted(Path(docs_path).glob("*.txt")):
        raw_content = path.read_text(encoding="utf-8-sig")
        
        # Clean and score quality
        cleaned, quality_score, quality_metadata = clean_ocr_text(raw_content, min_line_length=12)
        
        # Lowered threshold to 0.15 for consistency with JSON processing
        if not cleaned or quality_score < 0.15:
            skipped_txt.append((path.name, quality_score))
            continue
        
        metadata = {
            "source": str(path),
            "file_type": "txt",
            "section": detect_section(cleaned),
        }
        metadata.update(quality_metadata)
        
        # Create multi-level chunks for text files too
        chunk_docs = create_multi_level_chunks(cleaned, metadata)
        filtered_chunks = [doc for doc in chunk_docs if pre_filter_quality(doc.page_content)]
        documents.extend(filtered_chunks)
        print(f"  ✓ Loaded {len(filtered_chunks)} chunks from {path.name}")

    # JSON files (e.g. OCR-extracted newspaper) -> enhanced processing
    for path in sorted(Path(docs_path).glob("*.json")):
        json_docs = load_json_pages(path)
        if json_docs:
            documents.extend(json_docs)
        else:
            skipped_json.append(path.name)

    if len(documents) == 0:
        error_msg = f"No chunks loaded from {docs_path}."
        if skipped_txt:
            error_msg += f"\n  Skipped .txt files (too low quality): {skipped_txt}"
        if skipped_json:
            error_msg += f"\n  Skipped .json files (no valid chunks): {skipped_json}"
        raise FileNotFoundError(error_msg)
    
    # Summary stats
    print(f"\n=== Document Loading Summary ===")
    print(f"Total chunks loaded: {len(documents)}")
    
    # Count by section
    sections = defaultdict(int)
    quality_scores = []
    for doc in documents:
        sections[doc.metadata.get("section", "unknown")] += 1
        quality_scores.append(doc.metadata.get("final_quality_score", 0.5))
    
    print(f"Chunks by section: {dict(sections)}")
    avg_quality = sum(quality_scores)/len(quality_scores) if quality_scores else 0
    print(f"Average quality score: {avg_quality:.2f}")
    
    # Show sample
    if documents:
        print(f"\nSample chunks:")
        for i, doc in enumerate(documents[:2]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"  Source: {doc.metadata['source']}")
            print(f"  Section: {doc.metadata.get('section', 'N/A')}")
            print(f"  Quality: {doc.metadata.get('final_quality_score', 'N/A'):.2f}")
            print(f"  Date: {doc.metadata.get('primary_date', 'N/A')}")
            print(f"  Length: {len(doc.page_content)} chars")
            print(f"  Preview: {doc.page_content[:100]}...")

    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=80):
    """
    Split documents into smaller chunks with overlap.
    
    NOTE: Most documents are already chunked during load_documents() with
    multi-level chunking. This function handles any remaining unsplit content.
    
    For multilingual content like Marathi, larger chunks (800) preserve better context,
    and higher overlap (100) ensures semantic boundaries are respected.
    """
    print("Processing documents for embedding...")
    
    # Filter out chunks that are already too small or already chunked
    documents_to_split = [
        doc for doc in documents 
        if len(doc.page_content) > chunk_size * 1.5 
    ]
    
    chunks = []
    chunks.extend([doc for doc in documents if len(doc.page_content) <= chunk_size * 1.5])
    
    if documents_to_split:
        print(f"  Further splitting {len(documents_to_split)} large documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "।", ".", "?", "!", " ", ""],
        )
        
        split_chunks = text_splitter.split_documents(documents_to_split)
        chunks.extend(split_chunks)
    
    print(f"Total chunks ready for embedding: {len(chunks)}")
    
    # Show statistics
    chunk_sizes = [len(c.page_content) for c in chunks]
    if chunk_sizes:
        print(f"  Avg chunk size: {sum(chunk_sizes)//len(chunk_sizes)} chars")
        print(f"  Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)} chars")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=100):
    """
    Create and persist ChromaDB vector store with BGE-M3 hybrid search support.

    Embeddings are added in small batches so the local Ollama embedding
    runner is never handed a huge payload at once.
    
    Features:
    - Dense embeddings (semantic search)
    - Sparse embeddings (lexical/keyword search) via BGE-M3
    - Cosine similarity for vector comparison
    - Metadata-aware filtering capabilities
    """
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="bge-m3")

    # Create an empty ChromaDB vector store, then fill it batch by batch.
    print("--- Initializing ChromaDB with BGE-M3 embeddings ---")
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},  # Better for BGE-M3
    )

    # Add chunks in batches
    total = len(chunks)
    with tqdm(total=total, unit="chunk", desc="Embedding & Storing", ncols=80) as bar:
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            
            # Filter out any chunks that are still too noisy
            filtered_batch = [
                chunk for chunk in batch 
                if pre_filter_quality(chunk.page_content) and 
                   chunk.metadata.get("final_quality_score", 0.5) > 0.3
            ]
            
            if filtered_batch:
                vectorstore.add_documents(filtered_batch)
                bar.update(len(filtered_batch))
            
            # Skip poor quality chunks
            bar.update(len(batch) - len(filtered_batch))

    print("--- Vector store creation complete ---")
    print(f"Vector store created and persisted to {persist_directory}")
    print(f"Total vectors stored: {vectorstore._collection.count()}")
    
    return vectorstore

def main():
    """
    Main RAG document ingestion pipeline with:
    - Aggressive OCR noise filtering
    - Marathi date extraction
    - Multi-level chunking (small for embedding, large for context)
    - OCR quality scoring
    - Pre-filter quality checks
    - BGE-M3 hybrid search support
    """
    print("=" * 60)
    print("RAG DOCUMENT INGESTION PIPELINE (Enhanced)")
    print("=" * 60)
    print()
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    rebuild = "--rebuild" in sys.argv

    if rebuild and os.path.exists(persistent_directory):
        print("🔄 Rebuilding vector store from documents...")
        shutil.rmtree(persistent_directory)
        print()

    # Check if vector store already exists
    if os.path.exists(persistent_directory) and not rebuild:
        print("✓ Vector store already exists. Loading...")

        embedding_model = OllamaEmbeddings(model="bge-m3")
        try:
            vectorstore = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            count = vectorstore._collection.count()
            print(f"✓ Loaded existing vector store with {count} embedded chunks\n")
            return vectorstore
        except Exception as exc:
            raise RuntimeError(
                "The existing vector store could not be loaded. "
                "Run this script again with --rebuild to recreate it."
            ) from exc

    print("Creating new vector store from documents...\n")
    
    # Step 1: Load documents with enhanced processing
    print("STEP 1: Loading & Cleaning Documents")
    print("-" * 60)
    documents = load_documents(docs_path)
    print()

    # Step 2: Split/prepare chunks for embedding
    print("STEP 2: Preparing Chunks")
    print("-" * 60)
    chunks = split_documents(documents)
    print()
    
    # Step 3: Create vector store with BGE-M3
    print("STEP 3: Creating Vector Store")
    print("-" * 60)
    vectorstore = create_vector_store(chunks, persistent_directory)
    print()
    
    print("=" * 60)
    print("✓ INGESTION PIPELINE COMPLETE")
    print("=" * 60)
    
    return vectorstore


if __name__ == "__main__":
    vectorstore = main()
    print("\n✓ Ingestion complete! Your documents are now ready for RAG queries.")




