from typing import TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    question: str            # the user's question
    docs: list[Document]     # chunks found by retrieval
    answer: str              # the final answer