"""Ask the LangGraph agent. Prints each step as it runs, then the answer."""

import sys
from pathlib import Path

from agent.graph import agent


def main():
    question = " ".join(sys.argv[1:]) or input("Question: ")

    state = {}
    for step in agent.stream({"question": question}, stream_mode="updates"):
        for node, update in step.items():
            print(f"-> {node}")
            state.update(update)

    print("\n" + state["answer"])
    print("\nSources:")
    for name in dict.fromkeys(Path(d.metadata["source"]).name for d in state["docs"]):
        print("  -", name)


if __name__ == "__main__":
    main()