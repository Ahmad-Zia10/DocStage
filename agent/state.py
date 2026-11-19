from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str            # the user's original question
    query: str               # what we search with (rewritten on retries)
    attempts: int            # how many searches so far
    docs: list[Document]     # chunks from the latest search
    relevant: bool           # grader verdict on those chunks
    answer: str              # final answer