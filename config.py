"""
  - Indexing phase  -> DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, the embedding model
  - Query phase     -> TOP_K, the LLM
"""
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

## Paths

DOCS_DIR = os.getenv("DOCS_DIR", "./fastapi_docs")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

COLLECTION_NAME = "fastapi_docs" 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150  

TOP_K = 4  # how many nearest chunks similarity search returns per question

# Hybrid search: how much to trust [dense/meaning, keyword/BM25]. Must sum to 1.0.
HYBRID_WEIGHTS = [0.5, 0.5]


# Embedding model  (turns text -> meaning-vectors, used in BOTH phases)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

EMBEDDING_MODELS = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "voyage": "voyage-3.5",           # Anthropic's recommended provider
    "openai": "text-embedding-3-small",
}


# LLM 
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.0  # 0 = as deterministic/factual as possible

#create an embeddings object to be used in both ingest.py and ask.py
def get_embeddings():
    provider = EMBEDDING_PROVIDER.lower()
    model_name = EMBEDDING_MODELS[provider]

    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)

    # if provider == "voyage":
    #     from langchain_voyageai import VoyageAIEmbeddings
    #     return VoyageAIEmbeddings(model=model_name)

    # if provider == "openai":
    #     from langchain_openai import OpenAIEmbeddings
    #     return OpenAIEmbeddings(model=model_name)

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
