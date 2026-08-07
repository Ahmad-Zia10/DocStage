"""
ingest.py  —  THE INDEXING PHASE

After this finishes, the vector database on disk holds one meaning-vector per
chunk, each tagged with the file it came from (so we can cite it later).

"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config


def load_markdown_docs(docs_dir: str) -> list[Document]:
    """Read every .md file under docs_dir into a LangChain Document.

    We attach the file's relative path as metadata['source']. That single line
    is what makes CITATIONS possible at the end — every chunk remembers where
    it came from.
    """
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Docs folder not found: {root.resolve()}\n"
            "See the README for how to fetch the FastAPI docs, or set DOCS_DIR."
        )

    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        source = str(path.relative_to(root))
        docs.append(Document(page_content=text, metadata={"source": source}))

    if not docs:
        raise ValueError(f"No .md files found under {root.resolve()}")

    print(f"Loaded {len(docs)} markdown files from {root}")
    return docs


def chunk_docs(docs: list[Document]) -> list[Document]:
    """Cut each document into passage-sized chunks (the ChunkViz step).

    RecursiveCharacterTextSplitter tries to split on natural boundaries first
    (paragraphs, then lines, then sentences) instead of blindly every N chars,
    so it rarely guillotines an idea mid-thought. Metadata (our 'source') is
    copied onto every resulting chunk automatically.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed every chunk and store the vectors in Chroma (persisted to disk).

    Chroma.from_documents does two things per chunk under the hood:
      1. calls the embedding model to turn the chunk's text into a vector
      2. stores that vector + the chunk text + its metadata
    """
    embeddings = config.get_embeddings()
    print(f"Embedding with provider='{config.EMBEDDING_PROVIDER}' ... "
          "(first run downloads the local model; give it a minute)")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )
    print(f"Stored {len(chunks)} vectors in Chroma at '{config.CHROMA_DIR}'")
    return vector_store


def main() -> None:
    docs = load_markdown_docs(config.DOCS_DIR)
    chunks = chunk_docs(docs)
    build_vector_store(chunks)
    print("\n Indexing complete. Now try:  python ask.py")


if __name__ == "__main__":
    main()
