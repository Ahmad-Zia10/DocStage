# FastAPI Docs RAG — Beta

A question-answering assistant over the FastAPI documentation. You ask in plain
English; it finds the real, relevant passages from the docs, and Claude writes a
grounded answer **with citations** — instead of answering from fuzzy memory.

This is the **beta**: the straight-line RAG pipeline, instrumented and measured.
It is deliberately simple so it ships fast and gives you real baseline metrics.

## The pipeline (what the code does)

```
INDEXING (once, ahead of time)         QUERY (live, per question)
────────────────────────────           ──────────────────────────
docs → chunk → embed → store           question → embed → similarity search
      (ingest.py)                                 → stuff passages into prompt
                                                  → Claude answers + cites
                                                    (rag.py / ask.py)
```

| File               | Role                                                        |
|--------------------|-------------------------------------------------------------|
| `config.py`        | Every knob: paths, chunk size/overlap, top-k, model choices |
| `ingest.py`        | Indexing phase — load, **chunk**, **embed**, **store**      |
| `rag.py`           | Query phase — **retrieve** + **generate** (the LangChain chain) |
| `ask.py`           | Minimal CLI to ask questions                                |
| `evaluate.py`      | Metrics: retrieval hit-rate, latency, optional faithfulness |
| `eval/questions.json` | The starter evaluation set                               |

## Default tech choices (all swappable in `config.py`)

- **Embeddings:** a small model that runs **locally and free** (no API key).
- **Vector DB:** **Chroma**, stored on disk in `./chroma_db`.
- **LLM:** **Claude Haiku** (fast + cheap) for the answer step — the only piece
  that needs an API key.
- **Tracing:** **LangSmith** (optional) — set the env vars and it auto-records
  every step.

> Note: Anthropic does not provide its own embedding model and recommends
> [Voyage AI](https://docs.claude.com/en/docs/build-with-claude/embeddings).
> This project defaults to free local embeddings so you can run it at no cost;
> flip `EMBEDDING_PROVIDER` to `voyage` or `openai` for higher quality later.

## Setup

```bash
# 1. Create a virtual environment and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your key
cp .env.example .env
# open .env and paste your ANTHROPIC_API_KEY

# 3. Get the FastAPI docs (the corpus we answer from)
git clone --depth 1 https://github.com/fastapi/fastapi.git
# point the app at the English docs folder:
export DOCS_DIR=./fastapi/docs/en/docs      # (or set DOCS_DIR in .env)
```

## Run

```bash
# Index the docs (do this once; re-run if you change chunk settings)
python ingest.py

# Ask questions
python ask.py "How do I protect an endpoint so only logged-in users get in?"
python ask.py            # interactive mode

# Measure it (this is what goes on your portfolio)
python evaluate.py             # retrieval hit-rate + latency (free)
python evaluate.py --answers   # also print generated answers
python evaluate.py --judge     # also LLM-judged faithfulness (uses the API)
```

## Your baseline metrics

`evaluate.py` reports:

- **Retrieval hit-rate@k** — how often similarity search pulled the right doc
  into the top-k. Tests the retrieval half on its own.
- **Avg retrieval latency** — speed of the search step.
- **Faithfulness** (optional) — whether answers stay grounded in the retrieved
  text rather than hallucinating.

Record these now. They are the "before" numbers you'll beat in alpha.

## Where this goes next (alpha)

Each upgrade is justified by moving a metric above:

- **Hybrid search + reranking** → higher retrieval hit-rate
- **Knowledge graph (GraphRAG)** → answers multi-hop "what depends on what"
  questions that flat retrieval misses
- **LangGraph agent** → decide / loop / escalate instead of always answering
- **MCP tools** → take actions (e.g. open a GitHub issue), not just answer

## Tuning knobs to experiment with

All in `config.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBEDDING_PROVIDER`,
`LLM_MODEL`. Change one, re-run `ingest.py` (if it affects indexing) and
`evaluate.py`, and watch the numbers move. That loop *is* the job.
