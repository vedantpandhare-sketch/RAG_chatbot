# 🤖 RAG Chatbot

A **History-Aware Retrieval-Augmented Generation (RAG) Chatbot** built with LangChain, ChromaDB, Groq LLMs, and a Streamlit web interface. Ask questions about your own documents and get accurate, context-grounded answers — with full conversation memory.

---

## ✨ Features

- **📄 Document Ingestion** — Loads and chunks `.txt` files from a `Docs/` directory into a persistent ChromaDB vector store
- **🔍 Semantic Retrieval** — Retrieves the most relevant document chunks using cosine-similarity search via HuggingFace embeddings (`all-MiniLM-L6-v2`)
- **🧠 History-Aware Q&A** — Rewrites follow-up questions as standalone queries using conversation history before retrieval
- **⚡ Groq LLM Inference** — Ultra-fast inference powered by [Groq](https://groq.com/) with multiple model choices
- **🌐 Streamlit Web UI** — A polished, dark-themed chat interface with configurable settings, session stats, and source citation
- **💻 Terminal Chatbot** — Lightweight CLI chatbot for quick testing without launching a browser

---

## 🗂️ Project Structure

```
Chatbot_strlm/
│
├── Docs/                          # Place your .txt documents here
│
├── db/
│   └── chroma_db/                 # Persisted ChromaDB vector store (auto-generated)
│
├── 1_ingestion_pipeline.py        # Step 1: Load docs → chunk → embed → store in ChromaDB
├── 2_retrieval_pipeline.py        # Step 2: Test semantic retrieval from the vector store
├── 3_answer_generation.py         # Step 3: Single-turn Q&A with retrieved context
├── 4_terminal_chatbot.py          # Step 4: Multi-turn CLI chatbot with history
├── 5_history_aware_generation.py  # Step 5: Standalone question rewriting with history
│
├── streamlit_app.py               # Main Streamlit web application
│
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (GROQ_API_KEY) — not committed
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
│  History-Aware Question Rewriting  (Groq LLM)       │
│  "What did he invent?" → "What did Tesla invent?"   │
└──────────────────────┬──────────────────────────────┘
                       │ Rewritten standalone query
                       ▼
┌─────────────────────────────────────────────────────┐
│  Semantic Retrieval  (ChromaDB + all-MiniLM-L6-v2)  │
│  Top-K most relevant document chunks                 │
└──────────────────────┬──────────────────────────────┘
                       │ Retrieved context
                       ▼
┌─────────────────────────────────────────────────────┐
│  Answer Generation  (Groq LLM)                      │
│  Grounded answer from retrieved documents only      │
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

> **Note:** The embedding model (`all-MiniLM-L6-v2`) will be downloaded automatically from HuggingFace on first run (~90 MB).

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com).

---

## 🚀 Usage

### Step 1 — Add your documents

Place your `.txt` files inside the `Docs/` directory:

```
Docs/
├── company_policy.txt
├── product_manual.txt
└── faq.txt
```

### Step 2 — Run the ingestion pipeline

This loads, chunks, embeds, and stores your documents in ChromaDB:

```bash
python 1_ingestion_pipeline.py
```

To force a rebuild of the vector store:

```bash
python 1_ingestion_pipeline.py --rebuild
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
| **Sidebar — Model** | Choose from 4 Groq models (LLaMA 3.1 8B, LLaMA 3.3 70B, Mixtral 8x7B, Gemma2 9B) |
| **Sidebar — Top-K** | Control how many document chunks are retrieved per query (1–8) |
| **Sidebar — History Turns** | Limit how many past messages are sent to the LLM for context (2–12) |
| **Sidebar — Stats** | Live count of questions asked and history messages |
| **Sidebar — Status** | Live check for API key presence and vector DB availability |
| **Chat Area** | Full conversation history with animated message bubbles |
| **Source Expander** | Expandable panel showing which document chunks were used per answer |

---

## 🤖 Supported Groq Models

| Model | Context | Best For |
|---|---|---|
| `llama-3.1-8b-instant` | 131K | Fast responses, default |
| `llama-3.3-70b-versatile` | 128K | High-quality, complex reasoning |
| `mixtral-8x7b-32768` | 32K | Balanced speed and quality |
| `gemma2-9b-it` | 8K | Lightweight, instruction-tuned |

---

## 🔧 Configuration

Key constants in `streamlit_app.py` and pipeline scripts:

| Constant | Default | Description |
|---|---|---|
| `PERSIST_DIRECTORY` | `db/chroma_db` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `DEFAULT_CHAT_MODEL` | `llama-3.1-8b-instant` | Default Groq model |
| `TOP_K` | `3` | Documents retrieved per query |
| `MAX_HISTORY_TURNS` | `6` | Max past turns sent to LLM |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` | RAG orchestration & prompt management |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-groq` | Groq LLM integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `langchain-text-splitters` | Document chunking |
| `sentence-transformers` | Local embedding model |
| `python-dotenv` | `.env` file loading |

---

## 🛠️ Development Scripts

The numbered scripts represent a step-by-step build progression:

| Script | Purpose |
|---|---|
| `1_ingestion_pipeline.py` | Full ingestion: load → chunk → embed → persist |
| `2_retrieval_pipeline.py` | Test retrieval: query the vector store |
| `3_answer_generation.py` | Single-turn answer generation with context |
| `4_terminal_chatbot.py` | Multi-turn CLI chatbot with history |
| `5_history_aware_generation.py` | Demonstrates standalone question rewriting |

---

## 🔒 Security Notes

- **Never commit your `.env` file** — it is already listed in `.gitignore`
- The `GROQ_API_KEY` is loaded at runtime via `python-dotenv`
- The vector store in `db/` is local and not pushed to the repository

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with** ❤️ **using LangChain · Groq · ChromaDB · Streamlit · HuggingFace**

</div>
