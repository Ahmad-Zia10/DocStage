import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from ingest import load_markdown_docs, chunk_docs
import config


# Two rules do the heavy lifting: answer ONLY from the excerpts, and say so
# when they don't contain the answer instead of guessing.
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

STOPWORDS = {"how", "do", "i", "a", "an", "the", "to", "in", "of", "on", "for",
             "is", "my", "and", "or", "it", "what", "can", "with", "from",
             "this", "that", "be", "are", "does"}


# ---------------------------------------------------------------- retrieval

def tokenize(text):
    """Lowercase, keep words, drop filler words."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


@lru_cache(maxsize=1)
def get_dense():
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
        embedding_function=config.get_embeddings(),
    )
    return store.as_retriever(search_kwargs={"k": config.CANDIDATE_K})


@lru_cache(maxsize=1)
def get_bm25():
    # Built from the SAME chunks as Chroma, so RRF can match them up
    chunks = chunk_docs(load_markdown_docs(config.DOCS_DIR))
    if config.BM25_CLEAN_TOKENS:
        bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenize)
    else:
        bm25 = BM25Retriever.from_documents(chunks)   # LangChain default: split on spaces
    bm25.k = config.CANDIDATE_K
    return bm25


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(config.RERANK_MODEL)


def rrf(result_lists, c=60):
    """Reciprocal Rank Fusion: score = sum of 1 / (c + rank) across lists."""
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked]


def candidates(question: str) -> list[Document]:
    """Stage 1: the full candidate list (dense, bm25, or RRF-fused), before any cutting."""
    if config.RETRIEVAL_MODE == "dense":
        return get_dense().invoke(question)
    if config.RETRIEVAL_MODE == "bm25":
        return get_bm25().invoke(question)
    return rrf([get_dense().invoke(question), get_bm25().invoke(question)])


def rerank(question: str, docs: list[Document]) -> list[Document]:
    """Stage 2: the cross-encoder reads question + chunk together and scores each pair."""
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked]


def retrieve(question: str) -> list[Document]:
    docs = candidates(question)
    if config.RERANK:
        docs = rerank(question, docs[: config.RERANK_POOL])
    # Cut to exactly TOP_K so every mode is measured on the same k
    return docs[: config.TOP_K]


# --------------------------------------------------------------- generation

@lru_cache(maxsize=1)
def _get_chain():
    """prompt -> LLM -> plain text."""
    llm = ChatGroq(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    return PROMPT | llm | StrOutputParser()


def _format_context(docs: list[Document]) -> str:
    """Tag each chunk with its source so the model can cite it."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Excerpt {i} | source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str) -> dict:
    """Retrieve real passages, then answer strictly from them."""
    docs = retrieve(question)
    context = _format_context(docs)
    response = _get_chain().invoke({"context": context, "question": question})

    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)

    return {"answer": response, "sources": sources, "docs": docs}