from typing import List
import streamlit as st
import time
from pprint import pformat
from langchain_core.prompts import ChatPromptTemplate
from .models import ProjectLocation, TavilyResult, AgentState, QueryClassifier
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_tavily import TavilySearch
import os

llm = ChatGroq(model="llama-3.1-8b-instant")
tavily_search_tool = TavilySearch(max_results=5, search_depth="advanced")

def classify_query(state: AgentState) -> dict:
    user_input = state["user_input"]
    if st.session_state.get('verbose_output'):
        st.info("Executing Node: classify_query - Determining query type...")
        # st.markdown(f"**User Query:** `{user_input}`")
        time.sleep(0.05)

    prompt_classifier = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an AI assistant designed to classify user queries. "
         "Determine if the user's input is a 'legal_query' related to construction, permits, zoning, or owner-builder rights, "
         "or a 'general_query' (e.g., greetings, questions about your capabilities, company information, 'hello', 'hi'). "
         "Respond only with 'legal_query' or 'general_query'."),
        ("human", "{query}")
    ])

    classifier_chain = prompt_classifier | llm.with_structured_output(QueryClassifier)
    
    query_classification = "legal_query" # Default to general
    try:
        classification_result: QueryClassifier = classifier_chain.invoke({"query": user_input})
        query_classification = classification_result.query_type
    except Exception as e:
        print(f"Warning: Could not classify query using LLM. Defaulting to 'general_query'. Error: {e}") # Log to console

    if st.session_state.get('verbose_output'): # Removed direct st.markdown
        st.markdown(f"**Query Classified as:** `{query_classification}`")
        time.sleep(0.05)
    
    return {"query_type": query_classification}

def handle_general_query(state: AgentState) -> dict:
    user_input = state["user_input"]
    if st.session_state.get('verbose_output'): # Removed direct st.info/markdown
        st.info("Executing Node: handle_general_query - Responding to general query...")
        st.markdown(f"**User Query:** `{user_input}`")
        time.sleep(0.05)

    prompt_general_response = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a friendly AI assistant. Respond politely to general greetings or questions about yourself/company capabilities. "
         "This is company description. Please follow this"
         "Owner Builder Concepts (OBC) provides educational and informational content designed to help property owners better understand construction project planning, permitting, and owner-builder rights. OBC and its chatbot do not offer legal, engineering, or contracting advice."
        "All information presented through our chatbot, website, Substack posts, and roadmap materials is provided as-is and is based on publicly available sources, general building practices, and user-supplied data. Construction laws and building codes vary by city and state. Always consult with your local building authority, a licensed professional, or legal advisor before making decisions related to your project."
        "OBC is not responsible for any actions taken based on the information provided through this platform. Use of this information is at your own risk."
         "Do not attempt to provide legal advice or construction-related information here."),
        ("human", "{query}")
    ])

    general_response_chain = prompt_general_response | llm
    
    response_content = "Hello there! I'm here to help you with legal information and project roadmaps related to construction. How can I assist you today?"
    try:
        llm_response = general_response_chain.invoke({"query": user_input})
        response_content = llm_response.content
    except Exception as e:
        if st.session_state.get('verbose_output'): # Removed direct st.error
            st.error(f"Error generating general response: {e}")
        print(f"Error generating general response: {e}") # Log to console

    if st.session_state.get('verbose_output'): # Removed direct st.markdown/write
        st.markdown(f"**General Response:**")
        st.write(response_content)
        time.sleep(0.05)
    
    return {"project_roadmap": response_content}


def parse_user_input(state: AgentState) -> dict:
    user_input = state["user_input"]
    
    if st.session_state.get('verbose_output'):
        st.info("Executing Node: parse_user_input - Extracting project details...")
        st.markdown(f"**User Query:** `{user_input}`")
        time.sleep(0.05)
    
    prompt_parser = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts project details from user queries. Identify the project type, city, and state."),
        ("human", "{query}")
    ])
    
    # Using .with_structured_output from ChatGroq (real LLM)
    chain = prompt_parser | llm.with_structured_output(ProjectLocation) 
    
    extracted_project_type = "unknown"
    extracted_city = "unknown"
    extracted_geo_state = "unknown"

    try:
        parsed_info = chain.invoke({"query": user_input})
        # ChatGroq's with_structured_output will typically return a Pydantic object directly.
        if isinstance(parsed_info, ProjectLocation):
            extracted_project_type = parsed_info.project_type
            extracted_city = parsed_info.city
            extracted_geo_state = parsed_info.geo_state
        else: # Fallback if it returns something unexpected (e.g., dict or AIMessage content)
            if hasattr(parsed_info, 'content'): # For AIMessage content
                try:
                    import json
                    parsed_info_dict = json.loads(parsed_info.content)
                    parsed_info = ProjectLocation(**parsed_info_dict)
                    extracted_project_type = parsed_info.project_type
                    extracted_city = parsed_info.city
                    extracted_geo_state = parsed_info.geo_state
                except (json.JSONDecodeError, AttributeError):
                    if st.session_state.get('verbose_output'):
                        st.error(f"Error decoding structured output from LLM for ProjectLocation: {type(parsed_info)} - {str(parsed_info)}")
                    parsed_info = ProjectLocation(project_type="unknown", city="unknown", geo_state="unknown") # Fallback
            elif isinstance(parsed_info, dict): # For direct dictionary return
                extracted_project_type = parsed_info.get("project_type", "unknown")
                extracted_city = parsed_info.get("city", "unknown")
                extracted_geo_state = parsed_info.get("geo_state", "unknown")

        if st.session_state.get('verbose_output'):
            st.markdown(f"**Extracted Project Details:**")
            st.json({
                "project_type": extracted_project_type,
                "city": extracted_city,
                "geo_state": extracted_geo_state
            })
            time.sleep(0.05) # Small delay for rendering

    except Exception as e:
        if st.session_state.get('verbose_output'):
            st.warning(f"Warning: Could not extract structured info from LLM. Error: {e}")
        pass
        
    return {
        "project_type": extracted_project_type,
        "city": extracted_city,
        "geo_state": extracted_geo_state,
    }


def legal_search_agent(state: AgentState) -> AgentState:
    new_state = state.copy()

    project_type = new_state["project_type"]
    city = new_state["city"]
    geo_state = new_state["geo_state"]

    if project_type == "unknown" or city == "unknown" or geo_state == "unknown":
        if st.session_state.get('verbose_output'):
            st.warning("Legal Search Agent: Skipping search due to unknown project details from user input.")
        new_state["legal_info_found"] = False
        new_state["tavily_search_results"] = []
        return new_state

    queries = [
        f"owner-builder rights {city}, {geo_state}",
        f"{project_type} permit requirements {city}, {geo_state}",
        f"zoning laws {city}, {geo_state} {project_type} construction",
        f"local construction ordinances {city}, {geo_state}",
        f"building codes {project_type} {city}, {geo_state}"
    ]

    all_valid_search_results: List[TavilyResult] = []
    legal_info_found = False

    if st.session_state.get('verbose_output'):
        st.markdown("---") # Separator for clarity
        st.info("Legal Search Agent: Starting web searches for legal information...")
        time.sleep(0.05)

    for query in queries:
        try:
            if st.session_state.get('verbose_output'):
                st.markdown(f"Searching Tavily for: **`{query}`**")
                time.sleep(0.05)
            tavily_response_dict = tavily_search_tool.invoke({"query": query}) # This is the Mock Tool call

            if "results" in tavily_response_dict and isinstance(tavily_response_dict["results"], list):
                for result in tavily_response_dict["results"]:
                    if all(k in result for k in ["title", "content", "url"]):
                        all_valid_search_results.append(
                            TavilyResult(
                                title=result["title"],
                                content=result["content"],
                                url=result["url"]
                            )
                        )
                        legal_info_found = True
                    else:
                        if st.session_state.get('verbose_output'):
                            st.warning(f"Skipping malformed Tavily result (missing 'title', 'content', or 'url'): {result}")
            else:
                if st.session_state.get('verbose_output'):
                    st.warning(f"Tavily response for '{query}' did not contain a valid 'results' list or was empty. Response: {tavily_response_dict}")

        except Exception as e:
            if st.session_state.get('verbose_output'):
                st.error(f"Error during Tavily search for '{query}': {e}")
        time.sleep(0.05) # Small delay after each search query

    new_state["tavily_search_results"] = all_valid_search_results
    new_state["legal_info_found"] = legal_info_found

    if st.session_state.get('verbose_output'):
        st.success(f"Legal Search Agent: Completed search. Found {len(all_valid_search_results)} valid results. Legal information found: {legal_info_found}")
        if all_valid_search_results:
            st.markdown("**Tavily Search Results (Preview):**")
            for i, result in enumerate(all_valid_search_results): # Show top 3 results
                st.markdown(f"**Result {i+1}:** [{result.get('title', 'N/A')}]({result.get('url', '#')})")
                st.write(f"Content: {result.get('content', '')}") # Truncate content, show more
                time.sleep(0.01) # Very small delay per result
            time.sleep(0.05) # Small delay for rendering

    return new_state


def analyze_and_summarize(state: AgentState) -> dict:
    legal_info_found = state["legal_info_found"]
    project_type = state["project_type"]
    city = state["city"]
    geo_state = state["geo_state"]
    tavily_search_results = state["tavily_search_results"]

    final_legal_summary = "No legal summary could be generated."
    route_decision = "end"

    if legal_info_found and tavily_search_results:
        formatted_search_results = []
        for i, result in enumerate(tavily_search_results):
            formatted_search_results.append(
                f"--- Result {i+1} ---\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {result.get('content', 'N/A')}\n"
                f"-------------------\n"
            )
        search_results_for_llm = "\n".join(formatted_search_results)

        prompt_summarizer = ChatPromptTemplate.from_messages([
            ("system",
             f"You are a legal expert. Synthesize the following search results to provide a clear, concise summary of owner-builder rights, permit requirements, zoning laws, and local construction ordinances for a {project_type} project in {city}, {geo_state}. Focus on actionable information for an owner-builder. "
             "If a specific piece of information isn't found across all provided results, state that it's not available in the provided results. Cite sources by URL for each piece of information extracted from a specific result.Do include the urls provided by the user"),
            ("human", f"Here are the search results:\n\n{search_results_for_llm}")
        ])
        
        
        # Invoke real LLM for summarization
        summary_chain = prompt_summarizer | llm
        final_legal_summary = summary_chain.invoke({"search_results": search_results_for_llm}).content
        route_decision = "roadmap"

    if st.session_state.get('verbose_output'):
        st.markdown("---") # Separator for clarity
        st.info(f"Analyze and Summarize: Legal summary generated. Routing to '{route_decision}'.")
        st.markdown(f"**Generated Legal Summary:**")
        st.write(final_legal_summary) # Display the full summary
        time.sleep(0.05) # Small delay for rendering

    return {
        "legal_summary": final_legal_summary,
        "route_decision": route_decision
    }


def generate_project_roadmap(state: AgentState) -> dict:
    legal_summary = state["legal_summary"]
    project_type = state["project_type"]
    city = state["city"]
    geo_state = state["geo_state"]

    roadmap = "A project roadmap could not be generated due to missing legal information."

    if legal_summary and "No legal summary" not in legal_summary:

        system_prompt_content = (
            f"You are a project manager expert. Based on the following legal information for a {project_type} project in {city}, {geo_state}, "
            f"outline a step-by-step project roadmap from Phase 1: Legal Understanding through Phase 7: Final Inspections. "
            f"Detail the key actions in each phase. Incorporate the legal requirements where relevant.\n\n"
            f"Legal Summary:\n{legal_summary}\n\n"
        )
        
        system_prompt_content += (
            f"When detailing actions in a phase where information from a search result is used, "
            f"always cite the URL of the relevant search result directly within that phase's description. "
            f"For example: 'Action: Obtain permit (Source: https://example.com/permit-info)'. "
            f"Do not present any timelines."
            f"Always mention this descliminet Disclaimer to Embed:"
            f"Owner Builder Concepts (OBC) provides educational and informational content designed to help property owners better understand construction project planning, permitting, and owner-builder rights. OBC and its chatbot do not offer legal, engineering, or contracting advice."
            f"All information presented through our chatbot, website, Substack posts, and roadmap materials is provided as-is and is based on publicly available sources, general building practices, and user-supplied data. Construction laws and building codes vary by city and state. Always consult with your local building authority, a licensed professional, or legal advisor before making decisions related to your project."
            f"OBC is not responsible for any actions taken based on the information provided through this platform. Use of this information is at your own risk."
        )

        prompt_roadmap = ChatPromptTemplate.from_messages([
            ("system", system_prompt_content),
            ("human", "Generate the project roadmap.")
        ])

        roadmap_chain = prompt_roadmap | llm
        roadmap = roadmap_chain.invoke({}).content
        
    if st.session_state.get('verbose_output'):
        st.markdown("---")
        st.success(f"Generate Project Roadmap: Roadmap created for {project_type} in {city}, {geo_state}.")
        st.markdown(f"**Generated Project Roadmap (Preview):**")
        st.write(roadmap)
        time.sleep(0.05)
        
    return {"project_roadmap": roadmap}

def route_query_type(state: AgentState) -> str:
    """Routes based on the 'query_type' field in the state."""
    if state["query_type"] == "legal_query":
        return "legal_query"
    elif state["query_type"] == "general_query":
        return "general_query"
    return "general_query" # Fallback