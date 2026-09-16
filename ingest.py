"""
ingest.py  —  THE INDEXING PHASE

After this finishes, the vector database on disk holds one meaning-vector per
chunk, each tagged with the file it came from (so we can cite it later).

"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader


import config


def load_markdown_docs(docs_dir: str) -> list[Document]:
    loader = DirectoryLoader(
        docs_dir, glob="**/*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()   # each Document already carries metadata['source'] = file path
    if not docs:
        raise ValueError(f"No .md files found under '{docs_dir}'")
    print(f"Loaded {len(docs)} markdown docs from '{docs_dir}'")
    return docs


def chunk_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
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
