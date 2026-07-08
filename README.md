# 🤖 RAG Chatbot

A **History-Aware Retrieval-Augmented Generation (RAG) Chatbot** built with LangChain, ChromaDB, local Ollama LLMs, and a Streamlit web interface. Ask questions about your own documents (optimized for multilingual Marathi newspaper content) and get accurate, context-grounded answers — with full conversation memory.

---

## ✨ Features

- **Document Ingestion** — Loads and chunks documents from a `docs/` directory with aggressive OCR noise filtering, section detection, and Marathi/English date extraction
- **Multi-Level Chunking** — Stores small parent-child chunks to balance precise semantic search with rich context for the LLM
- **Semantic Retrieval** — Retrieves the most relevant document chunks using hybrid dense + sparse retrieval via Ollama's `bge-m3` embedding model
- **History-Aware Q&A** — Rewrites follow-up questions as standalone queries using conversation history before retrieval
- **Local Ollama LLM Inference** — Secure, local inference powered by [Ollama](https://ollama.com/) with multiple local model choices (e.g., Llama 2, Qwen, etc.)
- **Streamlit Web UI** — A polished, dark-themed chat interface with configurable settings, session stats, and source citations
- **Terminal Chatbot** — Lightweight CLI chatbot for quick testing without launching a browser

---

## 🗂️ Project Structure

```
RAG_chatbot/
│
├── docs/                          # Place your documents (.txt or .json structured pages) here
│
├── db/
│   └── chroma_db/                 # Persisted ChromaDB vector store (auto-generated)
│
├── 1_ingestion_pipeline.py        # Step 1: Load docs → clean & extract → chunk → embed (bge-m3) → ChromaDB
├── 2_retrieval_pipeline.py        # Step 2: Test hybrid semantic retrieval from the vector store
├── 3_answer_generation.py         # Step 3: Single-turn Q&A with retrieved context
├── 4_terminal_chatbot.py          # Step 4: Multi-turn CLI chatbot with history
├── 5_history_aware_generation.py  # Step 5: Standalone question rewriting with history
│
├── streamlit_app.py               # Main Streamlit web application
├── rag_utils.py                   # Shared utilities (Marathi answer selection, prompts, tokenization)
├── ingest_parallel.py             # Parallel multi-threaded document ingestion helper
│
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (optional)
├── .gitignore
└── README.md
```

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  History-Aware Question Rewriting (Ollama LLM)      │
│  "तिथे काय समस्या आहेत?" → "पुण्यात काय समस्या आहेत?"│
└──────────────────────┬──────────────────────────────┘
                       │ Rewritten standalone query
                       ▼
┌─────────────────────────────────────────────────────┐
│  Semantic Retrieval  (ChromaDB + Ollama bge-m3)     │
│  Top-K relevant chunks with pre-filtering           │
└──────────────────────┬──────────────────────────────┘
                       │ Retrieved context + metadata
                       ▼
┌─────────────────────────────────────────────────────┐
│  Answer Generation & Validation (Ollama LLM)        │
│  Grounded answer in Marathi from context documents   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
               Final Answer + Sources
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/VedantPandhare/RAG_chatbot.git
cd RAG_chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install & Start Ollama

Ensure you have [Ollama](https://ollama.com/) installed and running locally, and pull the required models:

```bash
ollama pull bge-m3
ollama pull llama2:latest
```

---

## 🚀 Usage

### Step 1 — Add your documents

Place your documents inside the `docs/` directory. The pipeline supports plain `.txt` files and structured `.json` page files.

### Step 2 — Run the ingestion pipeline

This loads, cleans, chunks, embeds, and stores your documents in ChromaDB:

```bash
python 1_ingestion_pipeline.py
```

To force a rebuild of the vector store:

```bash
python 1_ingestion_pipeline.py --rebuild
```

For faster parallel ingestion utilizing multi-threading:

```bash
python ingest_parallel.py --rebuild
```

### Step 3 — Launch the Streamlit app

```bash
venv\Scripts\streamlit.exe run streamlit_app.py
```

The app will open at **http://localhost:8501**.

### Alternative — Use the terminal chatbot

```bash
python 4_terminal_chatbot.py
```

---

## 🖥️ Streamlit UI Overview

| Panel | Description |
|---|---|
| **Sidebar — Model** | Choose from local models running in Ollama (Llama 2, Qwen 2.5, Llama 3.2, etc.) |
| **Sidebar — Top-K** | Control how many document chunks are retrieved per query (1–8) |
| **Sidebar — History Turns** | Limit how many past messages are sent to the LLM for context (2–12) |
| **Sidebar — Stats** | Live count of questions asked and history messages |
| **Sidebar — Status** | Live checks for Ollama server connectivity and vector database availability |
| **Chat Area** | Full conversation history with source document citations (pages, dates, quality score) |

---

## 🔧 Configuration

Key constants in `streamlit_app.py` and pipeline scripts:

| Constant | Default | Description |
|---|---|---|
| `PERSIST_DIRECTORY` | `db/chroma_db` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `bge-m3` | Ollama embedding model |
| `DEFAULT_CHAT_MODEL` | `llama2:latest` | Default local chat model |
| `TOP_K` | `5` | Documents retrieved per query |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` | RAG orchestration & prompt management |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-ollama` | Ollama LLM and embedding integration |
| `langchain-text-splitters` | Document chunking |
| `python-dotenv` | `.env` file loading |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built using LangChain · Ollama · ChromaDB · Streamlit**

</div>
