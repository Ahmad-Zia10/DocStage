from langgraph.graph import StateGraph, START, END

from agent.state import State
from agent.nodes import retrieve_docs, generate_answer


def build_graph():
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


agent = build_graph()