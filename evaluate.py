"""
evaluate.py  —  turn "I built a chatbot" into "I measured it"
=============================================================
This is the file that makes your portfolio credible. It runs a fixed set of
questions through the system and reports numbers you can quote.

Two metrics, matching what you learned:

  1. Retrieval hit-rate@k  (free, no LLM calls)
     For each question we know which doc *should* answer it. Did similarity
     search actually pull that doc into the top-k? This isolates whether the
     RETRIEVAL half is working, independent of the LLM.

  2. Faithfulness  (optional, uses the LLM as a judge — costs a little)
     Is the generated answer actually supported by the retrieved excerpts, or
     did the model wander off and make things up? Enable with --judge.

Run:
    python evaluate.py            # retrieval hit-rate + latency only (free)
    python evaluate.py --answers  # also print each generated answer
    python evaluate.py --judge    # also LLM-judge faithfulness (uses API)
"""

import argparse
import json
import time
from pathlib import Path

import config
from rag import retrieve, answer

QUESTIONS_PATH = Path(__file__).parent / "eval" / "questions.json"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def retrieval_hit(question: str, expected_substring: str) -> tuple[bool, float, list[str]]:
    """Retrieve, then check whether any retrieved chunk's source contains the
    expected substring. Also time how long retrieval took."""
    start = time.perf_counter()
    docs = retrieve(question)
    latency = time.perf_counter() - start

    sources = [d.metadata.get("source", "") for d in docs]
    hit = any(expected_substring.lower() in s.lower() for s in sources)
    return hit, latency, sources


def judge_faithfulness(question: str, generated: str, docs) -> bool:
    """Ask the LLM whether the answer is supported by the retrieved excerpts.

    A deliberately simple LLM-as-judge: it returns YES/NO. Real eval suites use
    more careful rubrics, but this teaches the core idea — grade the output
    against the evidence it was supposed to use.
    """
    from langchain_anthropic import ChatAnthropic

    context = "\n\n".join(d.page_content for d in docs)
    judge = ChatAnthropic(model=config.LLM_MODEL, temperature=0)
    verdict = judge.invoke(
        "You are grading whether an answer is fully supported by the given "
        "source text. Reply with exactly one word: YES or NO.\n\n"
        f"SOURCE TEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:\n{generated}\n\n"
        "Is every claim in the ANSWER supported by the SOURCE TEXT?"
    ).content.strip().upper()
    return verdict.startswith("YES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true",
                        help="also generate and print each answer")
    parser.add_argument("--judge", action="store_true",
                        help="also LLM-judge faithfulness (uses the API)")
    args = parser.parse_args()

    questions = load_questions()
    hits = 0
    latencies = []
    faithful = 0
    judged = 0

    print(f"\nEvaluating {len(questions)} questions "
          f"(k={config.TOP_K}, embeddings={config.EMBEDDING_PROVIDER})\n")

    for item in questions:
        q = item["question"]
        expected = item["expected_source_contains"]

        hit, latency, sources = retrieval_hit(q, expected)
        hits += int(hit)
        latencies.append(latency)

        mark = "HIT " if hit else "MISS"
        print(f"[{mark}] {q}")
        print(f"       expected source ~ '{expected}'  |  retrieved: {sources}")

        if args.answers or args.judge:
            result = answer(q)
            if args.answers:
                print(f"       answer: {result['answer'][:300]}...")
            if args.judge:
                ok = judge_faithfulness(q, result["answer"], result["docs"])
                judged += 1
                faithful += int(ok)
                print(f"       faithful: {'YES' if ok else 'NO'}")
        print()

    n = len(questions)
    avg_latency_ms = 1000 * sum(latencies) / n

    print("=" * 60)
    print("RESULTS")
    print(f"  Retrieval hit-rate@{config.TOP_K}: {hits}/{n} = {hits / n:.0%}")
    print(f"  Avg retrieval latency:     {avg_latency_ms:.0f} ms")
    if judged:
        print(f"  Faithfulness (LLM judge):  {faithful}/{judged} = {faithful / judged:.0%}")
    print("=" * 60)
    print("\nThese are your baseline numbers. Every alpha upgrade (hybrid "
          "search, reranking, the knowledge graph) should move them UP.\n")


if __name__ == "__main__":
    main()
