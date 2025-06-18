import streamlit as st
import os, uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain.globals import set_verbose

# 1️⃣ Page setup
st.set_page_config(page_title="🏗️ OBC Project Assistant", layout="centered")
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

set_verbose(True)

# 2️⃣ Clear session state only
def clear_everything():
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! Ask me about legal requirements."}
    ]
    st.session_state.thread_id = str(uuid.uuid4())

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
show_thinking = st.sidebar.checkbox("🤖 Show Thinking Process", value=False)

# 5️⃣ Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6️⃣ Build agent (once per session)
@st.cache_resource
def get_agent():
    llm = ChatGroq(model="llama3-8b-8192")
    tavily = TavilySearch(max_results=5, search_depth="advanced")
    prompt = (
        "You are the OBC Project Assistant: a legal-aware chatbot. "
        "Ask clarifying questions, generate precise search queries, fetch current legal sources, "
        "summarize with URLs, next steps, and a disclaimer. "
        "Based on user question you have to provide answer based on legal data."
        "You don't have to share the search query. You have to bring live results for it"
        "You always have to provide Urls when seaching web"
    )
    memory = MemorySaver()
    return create_react_agent(model=llm, tools=[tavily], prompt=prompt, checkpointer=memory, debug=False)

agent = get_agent()

# 7️⃣ Handle new input
if user_input := st.chat_input("Your project question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = {
        "messages": [HumanMessage(content=m["content"]) for m in st.session_state.messages if m["role"] == "user"]
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    assistant_msg = st.chat_message("assistant")
    assistant_text = assistant_msg.empty()
    full_response = ""
    step_logs = []  # 🧠 Collect thinking steps for this interaction only

    for chunk in agent.stream(history, config=config, stream_mode="updates"):
        if "agent" in chunk and "messages" in chunk["agent"]:
            for m in chunk["agent"]["messages"]:
                full_response += m.content
                assistant_text.markdown(full_response)

# Save assistant response to session
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 👇 Per-message thinking expander
    if show_thinking and step_logs:
        with st.chat_message("assistant"):
            with st.expander("🧠 Agent Thinking Process", expanded=True):
                for log in step_logs:
                    st.markdown(log)
