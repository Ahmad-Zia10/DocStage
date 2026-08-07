"""
ask.py  —  a thin interface over the RAG chain
==============================================
Deliberately minimal. In beta the interface doesn't matter; the pipeline does.
(In production this is where a web widget or Slack adapter would plug in.)

Usage:
    python ask.py "How do I protect an endpoint so only logged-in users can access it?"
    python ask.py           # then type questions interactively; blank line quits
"""

import sys

from rag import answer


def show(question: str) -> None:
    result = answer(question)
    print("\n" + "=" * 70)
    print(result["answer"])
    print("-" * 70)
    print("Sources used:")
    for src in result["sources"]:
        print(f"  - {src}")
    print("=" * 70 + "\n")


def main() -> None:
    if len(sys.argv) > 1:
        # Question passed on the command line.
        show(" ".join(sys.argv[1:]))
        return

    # Interactive mode.
    print("Ask a FastAPI question (blank line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        show(question)


if __name__ == "__main__":
    main()
