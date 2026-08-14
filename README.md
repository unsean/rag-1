# RAG — PDF Q&A with OpenRouter + Qdrant

A Retrieval-Augmented Generation app that ingests PDFs, embeds text chunks with NVIDIA embeddings via OpenRouter, stores vectors in Qdrant, and answers questions using an LLM.

## Tech Stack

- **FastAPI** + **Inngest** — API server & event-driven workflows
- **OpenRouter** — embeddings (`nvidia/nemotron-3-embed-1b:free`) & chat (`openai/gpt-oss-20b:free`)
- **Qdrant** — vector database (via Docker)
- **Streamlit** — web UI for upload & query
- **LlamaIndex** — PDF parsing & text chunking

## Prerequisites

1. Python 3.11+ with [uv](https://docs.astral.sh/uv/)
2. Node.js (for Inngest dev server)
3. Docker (for Qdrant)

## Setup

```bash
# Clone
git clone https://github.com/unsean/rag-1.git
cd rag-1

# Install Python deps
uv sync

# Install Node deps
npm install

# Create .env
echo API_KEY=your_openrouter_api_key > .env
```

## Running

```bash
# 1. Start Qdrant (Docker)
docker run -d --name qdrantRAGDB -p 6333:6333 qdrant/qdrant

# 2. Start Inngest dev server
npm run dev:inngest

# 3. Start FastAPI server
npm run dev:api

# 4. Start Streamlit UI
streamlit run streamlit_app.py
```

- FastAPI: http://localhost:8000
- Inngest UI: http://localhost:8288
- Streamlit: http://localhost:8501

## Usage

1. Upload a PDF via the Streamlit UI
2. Wait for ingestion to complete (check Inngest UI)
3. Ask a question in the query box

## Project Structure

```
main.py           # FastAPI app + Inngest functions (ingest & query)
data_loader.py    # PDF loading, chunking, embedding
vector_db.py      # Qdrant client wrapper
custom_types.py   # Pydantic models for serialization
streamlit_app.py  # Web UI
```
