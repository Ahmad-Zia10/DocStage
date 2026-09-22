from functools import lru_cache

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
# EnsembleRetriever lives in langchain-classic as of LangChain 1.x (it used to
# be importable from `langchain.retrievers`).
from langchain_classic.retrievers import EnsembleRetriever

from ingest import load_markdown_docs, chunk_docs
import config


# The instruction we wrap around the retrieved passages. Two rules do the heavy
# lifting: (1) answer ONLY from the provided context — this is what stops the
# model from falling back on its fuzzy memory and hallucinating; (2) if the
# context doesn't contain the answer, say so instead of guessing.
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant for the FastAPI documentation.\n"
     "Answer the question using ONLY the documentation excerpts provided.\n"
     "If the excerpts do not contain the answer, say you don't know based on "
     "the docs — do not invent anything.\n"
     "Be concise and, where useful, include a short code snippet.\n"
     "End your answer with a 'Sources:' line listing the source files you used."),
    ("human",
     "Documentation excerpts:\n\n{context}\n\n"
     "Question: {question}"),       
])


@lru_cache(maxsize=1)
def _get_retriever():
    """Hybrid retriever: dense (meaning) + BM25 (keyword), fused.

    - Dense half: the existing Chroma vector search — catches meaning matches.
    - Keyword half: BM25 built in memory from the SAME chunks — catches literal
      term matches the embedding misses.
    - EnsembleRetriever fuses both ranked lists (weighted reciprocal-rank
      fusion), so a doc ranked high by EITHER search rises to the top.

    Cached so we don't rebuild on every question. Both halves must use the same
    embedding model / chunking as ingest.py (via config) — that's mandatory.
    """
    # 1. Dense retriever — the meaning half (unchanged from before).
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
        embedding_function=config.get_embeddings(),
    )
    dense = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})

    # 2. Keyword retriever — the literal half. BM25 indexes in memory, so we
    #    regenerate the same chunks ingest.py produced (no re-embedding).
    chunks = chunk_docs(load_markdown_docs(config.DOCS_DIR))
    keyword = BM25Retriever.from_documents(chunks)
    keyword.k = config.TOP_K

    # 3. Fuse them. Weights say how much to trust each half; start balanced.
    return EnsembleRetriever(
        retrievers=[dense, keyword],
        weights=config.HYBRID_WEIGHTS,   # [dense, keyword], e.g. [0.5, 0.5]
    )


@lru_cache(maxsize=1)
def _get_chain():
    """Build the tiny LangChain 'chain': prompt -> LLM -> plain text.

    The `|` operator is LangChain's LCEL syntax — read it left to right as
    'take the prompt, feed it to the model, then parse the model's reply to a
    string'. LangSmith (if enabled via env vars) traces every link here.
    """
    llm = ChatGroq(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    return PROMPT | llm | StrOutputParser()


def _format_context(docs: list[Document]) -> str:
    """Lay the retrieved chunks out as text, each tagged with its source so the
    model can cite it."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Excerpt {i} | source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def retrieve(question: str) -> list[Document]:
    """Just the retrieval step, exposed on its own so the evaluator can measure
    it without paying for an LLM call."""
    return _get_retriever().invoke(question)


def answer(question: str) -> dict:
    """Full RAG: retrieve real passages, then answer strictly from them.

    Returns a dict with the answer text and the list of source files, so the
    caller can show citations.
    """
    docs = retrieve(question)
    context = _format_context(docs)
    response = _get_chain().invoke({"context": context, "question": question})

    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)

    return {"answer": response, "sources": sources, "docs": docs}
