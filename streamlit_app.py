import streamlit as st
import os, uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.globals import set_verbose
from langchain_community.tools import DuckDuckGoSearchRun


# 1️⃣ Page setup
st.set_page_config(page_title="🏗️ OBC Project Assistant", layout="centered")
load_dotenv()

# Ensure API keys are loaded and available
# It's good practice to check if they are actually loaded
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()
if not tavily_api_key:
    st.error("TAVILY_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Set environment variables for LangChain/LangGraph
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Enable verbose logging for LangChain
set_verbose(True)

# 2️⃣ Clear session state only
def clear_everything():
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! Ask me about legal requirements."}
    ]
    st.session_state.thread_id = str(uuid.uuid4())
    st.cache_resource.clear() # Clear cache when resetting chat to re-initialize agent

# 3️⃣ Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! Ask me about legal requirements."}
    ]

# 4️⃣ Sidebar controls
st.sidebar.title("⚙️ Options")
st.sidebar.button("🧹 Clear Chat", on_click=clear_everything)
show_thinking = st.sidebar.checkbox("🤖 Show Thinking Process", value=True) # Set to True for easier debugging initially

# 5️⃣ Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6️⃣ Build agent (once per session)
@st.cache_resource
def get_agent():
    llm = ChatGroq(model="llama3-70b-8192", temperature=0) # Set temperature to 0 for more deterministic behavior

    tavily_tool = TavilySearch(max_results=5, search_depth="advanced", include_domains=[".gov"])
    duckduckgosearch_tool = DuckDuckGoSearchRun()
    
    prompt = (
        """
        You are the **OBC Project Assistant** — a legally informed, professional chatbot advising homeowners on construction, zoning, and permitting.

        **CRITICAL INSTRUCTION: Your primary function is to provide legally accurate information by performing live web searches. You MUST use the 'tavily_search_results' tool for ALL legal or regulatory questions that require current or specific factual details about construction, zoning, or permitting, including but not limited to setbacks, height limits, permit requirements, and local ordinances.**

        **1. Clarify Up Front**
        • Ask follow-up questions to confirm exact project details: city & state, dimensions, structure type (attached/detached, open or roofed), and precise location on the parcel. If the user's initial query lacks specific location details, you MUST first clarify the city and state before attempting any legal lookup.

        **2. Always Perform Live Legal Lookup (via tavily tool)**
        • Always call tools
        • For any legal or regulatory questions, or when specific local details (like city/state codes) are requested or implied, **your first action after clarifying location MUST be to run precise web searches via the 'tavily_search_results' tool.**
        • Formulate search queries that are specific and likely to yield official government or authoritative sources (e.g., "Los Angeles zoning setback requirements for sheds", "permit requirements for deck in Austin TX").
        • Prioritize authoritative domains such as **.gov**, **.org**, **.edu** for accurate legal information.
        • Summarize findings in your own words (avoid long quotes), and **always include direct URLs** to these sources in the '🔍 Live Lookup' section. If a search yields no relevant official sources, clearly state that and explain what you found instead.

        **3. Use Structured Headings**
        - **🏡 Inquiry** – Restate the user’s scenario (location, dimensions, etc.).
        - **🔍 Live Lookup** – Say what you searched, summarize key legal points, and include links (emphasize .gov / .org / .edu sources). This section is mandatory for any legal/regulatory question.
        - **📏 Key Zoning/Code Factors** – Provide setbacks, height limits, structure type distinctions.
        - **📄 Permit Requirements** – Clarify permit triggers, exemptions.
        - **🛠 Owner‑Builder Notes** – Outline responsibilities/rights (e.g. California owner-builder laws).
        - **🗂 Next Steps** – Provide actionable recommendations:
          • Contact local building department (include phone & website)
          • Ask about your parcel’s specific setback requirements
          • Create a basic site plan
          • Adjust structure size or type based on code feedback

        **4. Include Disclaimers**
        • Use a brief disclaimer: *“I am not a lawyer; please verify with authorities or consult legal counsel.”*

        **5. Style & Clarity**
        • Keep it concise, professional, and reader-friendly.
        • Use bullet points and clear headings.
        • Ground every assertion in live legal sources, citing URLs.
        """
    )
    memory = MemorySaver()
    # Set debug=True for more verbose LangGraph internal logging
    return create_react_agent(model=llm, tools=[tavily_tool, duckduckgosearch_tool], prompt=prompt, checkpointer=memory, debug=True)

agent = get_agent()

# 7️⃣ Handle new input
if user_input := st.chat_input("Your project question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # LangGraph expects a list of HumanMessage objects, and we should preserve the full history
    # The `history` variable should contain all messages for the agent to maintain context.
    # When sending to the agent, we need to convert Streamlit messages to LangChain message types.
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # For assistant messages, we need to correctly handle tool_calls if they were present
            # This part is more complex if you're not explicitly storing tool_calls in session_state.
            # For now, we'll treat them as generic AIMessage content.
            langchain_messages.append(AIMessage(content=msg["content"]))
        # Add a placeholder for tool messages if they were to be stored
        # elif msg["role"] == "tool":
        #     langchain_messages.append(ToolMessage(content=msg["content"], tool_call_id=msg["tool_call_id"]))


    history = {"messages": langchain_messages}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    full_response = ""
    step_logs = []  # 🧠 Collect thinking steps for this interaction only

    # Stream the agent's response and capture thinking steps
    # We will collect all steps and the final response before rendering to display thinking first.
    for chunk in agent.stream(history, config=config, stream_mode="updates"):
        if "agent" in chunk:
            if "messages" in chunk["agent"]:
                for m in chunk["agent"]["messages"]:
                    if isinstance(m, AIMessage):
                        if m.tool_calls:
                            step_logs.append(f"**Tool Call Detected:** `{m.tool_calls}`")
                        full_response += m.content # Append content from AI messages to build the full response
                    elif isinstance(m, ToolMessage):
                        step_logs.append(f"**Tool Output:** `{m.content}` (for call ID: {m.tool_call_id})")
            
            # Capture tool_inputs/tool_outputs directly if present in the chunk for granular debugging
            if "tool_inputs" in chunk["agent"]:
                step_logs.append(f"**Tool Input (Agent):** {chunk['agent']['tool_inputs']}")
            if "tool_outputs" in chunk["agent"]:
                step_logs.append(f"**Tool Output (Agent):** {chunk['agent']['tool_outputs']}")
        
        # Other potential keys in "updates" mode:
        for key, value in chunk.items():
            if key not in ["agent", "messages", "__end__", "__start__"]:
                step_logs.append(f"**{key.capitalize()} Update:** `{value}`")

    # After the streaming loop is complete, render the thinking process and then the final response
    if show_thinking and step_logs:
        with st.chat_message("assistant"): # New message block for thinking
            with st.expander("🧠 Agent Thinking Process", expanded=True):
                for log in step_logs:
                    st.markdown(log)

    with st.chat_message("assistant"): # New message block for final response
        st.markdown(full_response)

    # Save assistant response to session (only the final content)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
