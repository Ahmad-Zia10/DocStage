# Roadmap

The north star: turn a working **beta** RAG assistant into a **production**-ready,
agentic system — one deliberate, measured step at a time. Every upgrade must be
justified by a metric it moves, not by "it seemed cool."

**How we work**
- One build step at a time. Pick the next unchecked item; finish it before the next.
- Every step ends in an atomic commit, with the metric change noted in the body.
- Deep-dive questions are welcome *about the current step*; unrelated curiosities
  go in `NOTES.md` (the learning backlog) so they never block shipping.
- A step is only "done" when it meets its **Done when** line below.

---

## Phase 1 — Beta ✅ (complete)

Goal: a straight-line RAG pipeline that answers FastAPI questions from the real
docs, with citations, fully instrumented and measured.

- [x] Ingestion: load → chunk → embed → store (Chroma)
- [x] Query: embed → similarity search → grounded, cited generation (Claude)
- [x] LangChain wiring + optional LangSmith tracing
- [x] Eval harness: retrieval hit-rate, latency, optional faithfulness

**Done when:** runs end to end and prints baseline metrics. → **Baseline recorded:**
`hit-rate@4 = ____`, `avg latency = ____ ms`, `faithfulness = ____` _(fill in)._

---

## Phase 2 — Alpha (in order)

Each item names the metric it should move. Don't start the next until the current
one beats the previous number.

### 2.1 — Hybrid search (keyword + dense)
- **Goal:** combine BM25 keyword search with vector search so exact terms
  (function names, error codes) and meaning-matches both get found.
- **Moves:** retrieval hit-rate / recall@k.
- **Done when:** hybrid retrieval is in and hit-rate beats the beta baseline.

### 2.2 — Reranking
- **Goal:** re-score the retrieved candidates with a reranker so the *most*
  relevant chunks land at the top of the top-k.
- **Moves:** precision of the top results; answer quality.
- **Done when:** reranker is in and hit-rate@k (or judged answer quality) improves
  over 2.1.

### 2.3 — Knowledge graph (GraphRAG)
- **Goal:** store entities and relationships (feature → depends-on → utility) to
  answer multi-hop "what relates to / what breaks if I change X" questions.
- **Moves:** accuracy on a new multi-hop eval subset that flat retrieval fails.
- **Done when:** the graph answers a relationship question the beta could not,
  measured on the multi-hop subset.

### 2.4 — LangGraph agent
- **Goal:** replace the fixed chain with a graph that can decide, loop, and branch
  — route intent, gate on confidence, re-query on weak retrieval, escalate when
  it can't answer.
- **Moves:** fewer wrong/confident answers; handles ambiguous and out-of-scope.
- **Done when:** the system chooses answer / clarify / escalate correctly, with
  routing accuracy measured.

### 2.5 — MCP tools (take actions)
- **Goal:** let the agent act, not just answer — e.g. search or open a GitHub
  issue via an MCP server.
- **Moves:** tool-call accuracy (right tool, right arguments).
- **Done when:** the agent completes a real tool action, measured for correctness.

---

## Phase 3 — Production (hardening)

Goal: safe, observable, deployable, and useful on real traffic.

- [ ] Multi-tenant data isolation + auth
- [ ] Guardrails: PII handling, prompt-injection defense, out-of-scope refusal
- [ ] Human-in-the-loop handoff for low-confidence answers
- [ ] Feedback loop: thumbs-down → new eval cases
- [ ] Caching + cost/latency controls
- [ ] Deploy: containerize, CI, one cloud target
- [ ] Second channel: web widget and/or Slack adapter

**Done when:** deployed and monitored, cost/latency within target, and deflection
measured on real or synthetic traffic.

---

## Portfolio metrics to capture along the way

Keep a simple table of before/after numbers as you go — this is the story that
proves competence:

| Milestone            | hit-rate@k | latency (p50) | faithfulness | notes |
|----------------------|-----------|---------------|--------------|-------|
| Beta baseline        |           |               |              |       |
| + Hybrid search      |           |               |              |       |
| + Reranking          |           |               |              |       |
| + Knowledge graph    |           |               |              | multi-hop subset |