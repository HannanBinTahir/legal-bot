from langgraph.graph import StateGraph, END
from .models import AgentState
from .agents import classify_query, handle_general_query, parse_user_input, legal_search_agent, analyze_and_summarize, generate_project_roadmap, route_query_type
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

workflow = StateGraph(AgentState)


workflow = StateGraph(AgentState)
workflow.add_node("classify_query", classify_query)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("parse_user_input", parse_user_input)
workflow.add_node("legal_search_agent", legal_search_agent)
workflow.add_node("analyze_and_summarize", analyze_and_summarize)
workflow.add_node("generate_roadmap", generate_project_roadmap)
workflow.set_entry_point("classify_query")
workflow.add_conditional_edges("classify_query", route_query_type, {
    "legal_query": "parse_user_input",
    "general_query": "handle_general_query",
})
workflow.add_edge("parse_user_input", "legal_search_agent")
workflow.add_edge("legal_search_agent", "analyze_and_summarize")
workflow.add_edge("analyze_and_summarize", "generate_roadmap")
workflow.add_edge("generate_roadmap", END)
workflow.add_edge("handle_general_query", END)


# Add nodes and edges as in main file
app = workflow.compile(checkpointer=memory)