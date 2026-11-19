from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (retrieve_docs, grade_docs, rewrite_query,
                         generate_answer, escalate, route_after_grade)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("grade", grade_docs)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate_answer)
    graph.add_node("escalate", escalate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()


agent = build_graph()