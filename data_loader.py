import os
import logging
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
)

EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBED_DIM = 2048

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
reader = PDFReader()

def load_chunk_pdf(path:str):
    docs = reader.load_data(path)
    logging.info(f"PDF loaded: {len(docs)} pages, texts non-empty: {sum(1 for d in docs if getattr(d, 'text', None) and d.text.strip())}")
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    chunks = [c.strip() for c in chunks if c and c.strip()]
    logging.info(f"Total chunks after split+filter: {len(chunks)}")
    return chunks

def embed_texts(texts:list[str]) -> list[list[float]]:
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("No non-empty texts to embed")
    logging.info(f"Embedding {len(texts)} texts, first 100 chars: {texts[0][:100] if texts else 'N/A'}")
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logging.error(f"Embedding API error: {e}")
        raise

