from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (retrieve_docs, grade_docs, rewrite_query,
                         generate_answer, escalate, route_after_grade)
from agent.nodes import (classify_question, refuse, route_after_classify,
                         retrieve_docs, grade_docs, rewrite_query,
                         generate_answer, escalate, route_after_grade)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("grade", grade_docs)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate_answer)
    graph.add_node("escalate", escalate)
    graph.add_node("classify", classify_question)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("refuse", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)


    return graph.compile()


agent = build_graph()