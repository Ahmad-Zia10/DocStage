"""
rag.py  —  THE QUERY PHASE (runs live, every time someone asks)
==============================================================
This is the bottom half of the pipeline:

    question -> EMBED -> SIMILARITY SEARCH -> stuff passages into a prompt
             -> Claude writes a grounded, cited answer

The public function you'll call from everywhere else is `answer(question)`.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

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
    """Open the persisted vector store and expose it as a retriever.

    Cached so we don't re-open Chroma on every question. The retriever uses the
    SAME embedding model as ingest.py (via config) — that's mandatory.
    """
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
        embedding_function=config.get_embeddings(),
    )
    return vector_store.as_retriever(search_kwargs={"k": config.TOP_K})


@lru_cache(maxsize=1)
def _get_chain():
    """Build the tiny LangChain 'chain': prompt -> LLM -> plain text.

    The `|` operator is LangChain's LCEL syntax — read it left to right as
    'take the prompt, feed it to the model, then parse the model's reply to a
    string'. LangSmith (if enabled via env vars) traces every link here.
    """
    llm = ChatAnthropic(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
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
