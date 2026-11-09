"""evaluate.py — retrieval metrics, candidate-pool ceiling, latency, optional judge."""

import argparse
import json
import statistics
import time
from pathlib import Path

from langchain_groq import ChatGroq

import config
from rag import retrieve, candidates, answer

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def first_rank(docs, expected):
    """Rank of the first chunk whose source matches ANY accepted label."""
    if isinstance(expected, str):
        expected = [expected]
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "").lower()
        if any(e.lower() in src for e in expected):
            return i
    return None


def judge_groundedness(question, generated, docs):
    """Graded LLM-as-judge: 1-5 score + the single unsupported claim (or 'none')."""
    context = "\n\n".join(d.page_content for d in docs)
    judge = ChatGroq(model=config.LLM_MODEL, temperature=0)
    raw = judge.invoke(
        "You grade whether an ANSWER is grounded in the SOURCE TEXT.\n"
        "Reply in EXACTLY this format:\n"
        "SCORE: <integer 1-5, where 5 = every claim supported, 1 = mostly unsupported>\n"
        "UNSUPPORTED: <the single most important claim in the ANSWER not supported "
        "by the SOURCE TEXT, or 'none'>\n\n"
        f"SOURCE TEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:\n{generated}"
    ).content

    score, unsupported = 3, "(unparsed)"
    for line in raw.splitlines():
        if line.upper().startswith("SCORE:"):
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                score = int(digits[0])
        elif line.upper().startswith("UNSUPPORTED:"):
            unsupported = line.split(":", 1)[1].strip()
    return score, unsupported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    questions = load_questions()

    print("Warming up...")
    retrieve(questions[0]["question"])   # loads embeddings, BM25 and reranker before timing

    print(f"\nEvaluating {len(questions)} questions (k={config.TOP_K}, "
          f"mode={config.RETRIEVAL_MODE}, rerank={config.RERANK}, "
          f"pool={config.RERANK_POOL})\n")

    hits_k = hits_1 = pool_hits = 0
    rr_sum = 0.0
    latencies, scores = [], []

    for item in questions:
        q, expected = item["question"], item["expected_source_contains"]

        start = time.perf_counter()
        docs = retrieve(q)
        latencies.append(time.perf_counter() - start)

        rank = first_rank(docs, expected)
        pool_rank = first_rank(candidates(q)[: config.RERANK_POOL], expected)

        hits_k += rank is not None
        hits_1 += rank == 1
        rr_sum += 1 / rank if rank else 0
        pool_hits += pool_rank is not None

        mark = f"rank {rank}" if rank else "MISS"
        note = "" if pool_rank else "   [not in pool]"
        print(f"[{mark:>7}] ({latencies[-1] * 1000:.0f} ms) {q}{note}")

        if args.answers or args.judge:
            result = answer(q)
            if args.answers:
                print(f"          answer: {result['answer'][:200]}...")
            if args.judge:
                score, unsupported = judge_groundedness(q, result["answer"], result["docs"])
                scores.append(score)
                print(f"          groundedness: {score}/5  |  unsupported: {unsupported}")

    n = len(questions)
    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  hit-rate@{config.TOP_K}:  {hits_k}/{n} = {hits_k / n:.0%}")
    print(f"  hit-rate@1:  {hits_1}/{n} = {hits_1 / n:.0%}")
    print(f"  MRR:         {rr_sum / n:.3f}")
    print(f"  hit@{config.RERANK_POOL} (pool): {pool_hits}/{n} = {pool_hits / n:.0%}")
    print(f"  latency p50: {1000 * statistics.median(latencies):.0f} ms   "
          f"avg: {1000 * sum(latencies) / n:.0f} ms")
    if scores:
        grounded = sum(1 for s in scores if s >= 4)
        print(f"  groundedness avg: {sum(scores) / len(scores):.1f}/5")
        print(f"  grounded (>=4):   {grounded}/{len(scores)} = {grounded / len(scores):.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()