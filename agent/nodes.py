from rag import retrieve, generate


def retrieve_docs(state):
    """Find the top chunks for the question."""
    return {"docs": retrieve(state["question"])}


def generate_answer(state):
    """Write the answer from the retrieved chunks."""
    return {"answer": generate(state["question"], state["docs"])}