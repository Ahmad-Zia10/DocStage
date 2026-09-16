"""evaluate.py — strengthened eval harness (graded judge, hit@1, MRR, clean latency)."""

import argparse
import json
import time
from pathlib import Path

import config
from rag import retrieve, answer

QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def retrieval_metrics(question: str, expected: str):
    """Retrieve, then find the RANK of the first correct doc (for hit@1 + MRR)."""
    start = time.perf_counter()
    docs = retrieve(question)
    latency = time.perf_counter() - start

    sources = [d.metadata.get("source", "") for d in docs]
    rank = None
    for i, s in enumerate(sources, start=1):
        if expected.lower() in s.lower():
            rank = i
            break
    return rank, latency, sources


def judge_groundedness(question: str, generated: str, docs):
    """Graded LLM-as-judge: 1-5 score + the single unsupported claim (or 'none')."""
    context = "\n\n".join(d.page_content for d in docs)
    judge = config.get_llm()  # uses your configured provider (Groq)
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

    # Warm up so the one-time model load isn't counted in latency.
    print("Warming up...")
    retrieve(questions[0]["question"])

    hits_k = hits_1 = 0
    reciprocal_sum = 0.0
    latencies = []
    scores = []

    print(f"\nEvaluating {len(questions)} questions "
          f"(k={config.TOP_K}, embeddings={config.EMBEDDING_PROVIDER})\n")

    for item in questions:
        q, expected = item["question"], item["expected_source_contains"]
        rank, latency, sources = retrieval_metrics(q, expected)
        latencies.append(latency)

        hits_k += int(rank is not None)
        hits_1 += int(rank == 1)
        reciprocal_sum += (1.0 / rank) if rank else 0.0

        mark = f"rank {rank}" if rank else "MISS"
        print(f"[{mark:>7}] ({latency*1000:.0f} ms) {q}")

        if args.answers or args.judge:
            result = answer(q)
            if args.answers:
                print(f"          answer: {result['answer'][:200]}...")
            if args.judge:
                score, unsupported = judge_groundedness(q, result["answer"], result["docs"])
                scores.append(score)
                print(f"          groundedness: {score}/5  |  unsupported: {unsupported}")
        print()

    n = len(questions)
    print("=" * 60)
    print("RESULTS")
    print(f"  hit-rate@{config.TOP_K}: {hits_k}/{n} = {hits_k / n:.0%}")
    print(f"  hit-rate@1:  {hits_1}/{n} = {hits_1 / n:.0%}")
    print(f"  MRR:         {reciprocal_sum / n:.3f}")
    print(f"  avg latency: {1000 * sum(latencies) / n:.0f} ms  (post-warmup)")
    if scores:
        grounded = sum(1 for s in scores if s >= 4)
        print(f"  groundedness avg: {sum(scores) / len(scores):.1f}/5")
        print(f"  grounded (>=4):   {grounded}/{len(scores)} = {grounded / len(scores):.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()