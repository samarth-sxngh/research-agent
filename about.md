# NotebookLM — Understand Anything 🧠

A self-hosted, AI-powered research assistant inspired by Google's NotebookLM. Upload documents, websites, audio, and YouTube videos, then chat with your sources or generate podcast-style audio summaries using local and cloud AI models.

---

## Features

| Feature | Description |
|---|---|
| 📄 **Document Q&A** | Upload PDFs, text, and Markdown files and ask questions with citation-backed answers |
| 🌐 **Web Scraping** | Ingest any public URL as a source via Firecrawl |
| 🎥 **YouTube Transcription** | Extract and index audio from any YouTube video via AssemblyAI |
| 🎤 **Audio Upload** | Transcribe MP3/WAV/M4A/OGG files and query their contents |
| 🎙️ **Podcast Studio** | Generate AI-hosted two-speaker podcast scripts + audio (via Kokoro TTS) |
| 🧠 **Persistent Memory** | Conversation memory powered by Zep Cloud |
| 🔍 **Vector Search** | Semantic search over all your sources with Milvus Lite |

---

## Architecture

```
app.py  (Streamlit UI)
│
├── src/document_processing/   — PDF, text, Markdown ingestion & chunking
├── src/embeddings/            — FastEmbed embedding generation
├── src/vector_database/       — Milvus Lite vector store (CRUD + search)
├── src/generation/            — RAG pipeline using OpenAI / Ollama
├── src/memory/                — Zep Cloud session memory
├── src/audio_processing/      — AssemblyAI audio & YouTube transcription
├── src/web_scraping/          — Firecrawl web scraper
└── src/podcast/               — Ollama script generator + Kokoro TTS
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | Set in `.python-version` |
| **Ollama** | Required for podcast script generation (`llama3.2:1b` by default) |
| **OpenAI API Key** | For RAG answer generation (optional if using Ollama only) |
| **AssemblyAI API Key** | For audio/YouTube transcription (optional) |
| **Firecrawl API Key** | For web scraping (optional) |
| **Zep Cloud API Key** | For persistent conversation memory (optional) |

---

## Setup & Running

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd notebook-lm
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

Using `pip` with the provided requirements file:

```bash
pip install -r requirements.txt
```

Or using `uv` (faster, uses the lock file):

```bash
pip install uv
uv sync
```

### 4. Configure environment variables

Copy the example `.env` and fill in your API keys:

```bash
cp .env .env.local   # optional: keep secrets outside version control
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...          # Required for RAG generation
ASSEMBLYAI_API_KEY=...         # Optional: audio & YouTube
FIRECRAWL_API_KEY=...          # Optional: web scraping
ZEP_API_KEY=...                # Optional: conversation memory

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b       # Podcast script generation
```

### 5. Start Ollama (for podcast generation)

```bash
ollama serve                   # start the Ollama server
ollama pull llama3.2:1b        # download the default model
```

### 6. Run the application

```bash
streamlit run app.py
```

The app will be available at **http://localhost:8501** by default.

---

## Usage

1. Open the app in your browser at `http://localhost:8501`.
2. Go to the **"Add Sources"** tab and upload files, paste URLs, add a YouTube link, or paste plain text.
3. Switch to the **"Chat"** tab and ask questions — answers include inline citations.
4. Use the **"Studio"** tab to generate a podcast from any of your sources.

---

## Project Structure

```
notebook-lm/
├── app.py                  # Streamlit application entry point
├── main.py                 # CLI entry point (stub)
├── pyproject.toml          # Project metadata & uv dependency spec
├── requirements.txt        # pip-compatible dependency list
├── uv.lock                 # Reproducible lock file (uv)
├── .env                    # Environment variables (not committed)
├── .python-version         # Pinned Python version (3.12)
├── data/                   # Local data / ChromaDB storage
├── notebooks/              # Jupyter exploration notebooks
├── tests/                  # Test suite
└── src/
    ├── audio_processing/   # Audio & YouTube transcription
    ├── document_processing/# Document ingestion & chunking
    ├── embeddings/         # Embedding generation
    ├── generation/         # RAG pipeline
    ├── memory/             # Zep memory layer
    ├── podcast/            # Podcast script & TTS
    ├── vector_database/    # Milvus vector DB
    └── web_scraping/       # Web scraper
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Pipeline initialization failed` | Check all API keys in `.env` are set correctly |
| `Kokoro TTS not available` | Install `kokoro` manually: `pip install kokoro>=0.9.4` |
| `No content extracted from URL` | Verify `FIRECRAWL_API_KEY` is valid |
| `YouTube processing unavailable` | Set `ASSEMBLYAI_API_KEY` in `.env` |
| `Ollama connection refused` | Run `ollama serve` before starting the app |
| Milvus `.db` files accumulating | These are per-session databases; add `milvus_lite_*.db` to `.gitignore` |

---

## License

This project is open source. See [LICENSE](LICENSE) for details.
