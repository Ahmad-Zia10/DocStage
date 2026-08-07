"""
  - Indexing phase  -> DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, the embedding model
  - Query phase     -> TOP_K, the LLM
"""

import os
from dotenv import load_dotenv

#  Load environment variables from .env file 
load_dotenv()

## Paths

DOCS_DIR = os.getenv("DOCS_DIR", "./fastapi_docs")

# Where the vector database persists its files on disk. 
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# A name for this collection inside Chroma.
COLLECTION_NAME = "fastapi_docs"

# Chunking

# Size is measured in CHARACTERS here (not tokens) — ~1000 chars is big enough to hold
# a complete idea, small enough to stay about one topic. 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150  # shared text so ideas sitting at boundaries aren't cut out.


# Retrieval 

TOP_K = 4  # how many nearest chunks similarity search returns per question


# Embedding model  (turns text -> meaning-vectors, used in BOTH phases)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

EMBEDDING_MODELS = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "voyage": "voyage-3.5",           # Anthropic's recommended provider
    "openai": "text-embedding-3-small",
}


# LLM 
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_TEMPERATURE = 0.0  # 0 = as deterministic/factual as possible


def get_embeddings():
    """Build the embedding object for the provider chosen above.

    IMPORTANT: whatever this returns must be used for BOTH indexing the docs
    and embedding the question at query time. Same model both times, always —
    otherwise the question and the chunks live in different number-spaces and
    similarity search is meaningless.
    """
    provider = EMBEDDING_PROVIDER.lower()
    model_name = EMBEDDING_MODELS[provider]

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)

    # if provider == "voyage":
    #     from langchain_voyageai import VoyageAIEmbeddings
    #     return VoyageAIEmbeddings(model=model_name)

    # if provider == "openai":
    #     from langchain_openai import OpenAIEmbeddings
    #     return OpenAIEmbeddings(model=model_name)

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
