"""Top reranker score per eval question, next to whether retrieval got it right.
Used to pick the agent's confidence threshold."""

from evaluate import load_questions, first_rank
from rag import retrieve

rows = []
for item in load_questions():
    docs = retrieve(item["question"])
    top = docs[0].metadata["rerank_score"]
    rank = first_rank(docs, item["expected_source_contains"])
    rows.append((top, rank, item["question"]))

for top, rank, question in sorted(rows, key=lambda r: r[0], reverse=True):
    result = f"rank {rank}" if rank else "MISS"
    print(f"{top:7.2f}  {result:>7}  {question}")