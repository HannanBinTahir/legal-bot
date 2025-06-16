import streamlit as st
import uuid
from src.workflow import app
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage

# Config
st.set_page_config(page_title="OBC Project Assistant", layout="wide")
st.title("🏡 OBC Project Assistant")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_list" not in st.session_state:
    st.session_state.history_list = []

def clear_chat():
    st.session_state.messages = []

st.sidebar.button("Clear Chat", on_click=clear_chat)

# Display chat history sidebar
# st.sidebar.markdown("### Chat History")
# for i, item in enumerate(st.session_state.history_list):
#     snippet = item["query"][:50] + ("..." if len(item["query"]) > 50 else "")
#     if st.sidebar.button(f"{i+1}: {snippet}"):
#         st.session_state.messages = [
#             {"role": "user", "content": item["query"]},
#             {"role": "assistant", "content": item["answer"]},
#         ]
#         st.experimental_rerun()

# Render current conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if prompt := st.chat_input("Your Project Query"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build history prompt
    prev = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
    formatted = "\n".join(f"- {msg}" for msg in prev)
    combined = f"""Previous Messages:\n{formatted}\n\nLatest Message:\n{prompt}"""

    cb = StreamlitCallbackHandler(st.container())
    initial_state = {
        "user_input": combined,
        "messages": [HumanMessage(content=combined)],
        "project_type": "unknown",
        "city": "unknown",
        "geo_state": "unknown",
        "legal_info_found": False,
        "legal_summary": "",
        "suggested_websites": [],
        "project_roadmap": "",
        "route_decision": ""
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}, "callbacks": [cb]}

    try:
        for _ in app.stream(initial_state, config=config, stream_mode="debug"):
            pass
        final = app.get_state(config).values
        roadmap = final.get("project_roadmap", "No roadmap generated.")
    except Exception as e:
        st.error(f"Execution error: {e}")
        roadmap = "Roadmap generation failed."

    st.chat_message("assistant").markdown(roadmap)
    st.session_state.messages.append({"role": "assistant", "content": roadmap})
    # Save to history
    st.session_state.history_list.append({"query": prompt, "answer": roadmap})

# Sidebar verbose toggle
st.sidebar.checkbox("Show Detailed Agent Trace", key="verbose_output")
