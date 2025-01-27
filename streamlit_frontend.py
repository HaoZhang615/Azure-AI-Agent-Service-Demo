import os
import streamlit as st
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool, BingGroundingTool, ToolSet
from azure.identity import DefaultAzureCredential
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# function to clear chat history
def clear_chat_history():
    if st.session_state.agent_id:
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(), conn_str=os.environ["PROJECT_CONNECTION_STRING"]
        )
        project_client.agents.delete_agent(st.session_state.agent_id)
        st.session_state.agent_id = None
    if st.session_state.thread_id:
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(), conn_str=os.environ["PROJECT_CONNECTION_STRING"]
        )
        project_client.agents.delete_thread(st.session_state.thread_id)
        st.session_state.thread_id = None
    st.session_state.messages = []

st.title("Azure AI Agent Service Chat")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_id" not in st.session_state:
    st.session_state.agent_id = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.subheader("Agent Settings")
    instructions = st.text_area("Instructions", "You are a helpful agent")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_prompt_tokens = st.slider("Max Prompt Tokens", min_value=500, max_value=100000, value=20000, step=100)
    max_completion_tokens = st.slider("Max Completion Tokens", min_value=1000, max_value=100000, value=1000, step=100)
    tool_choices = ["BingGrounding", "CodeInterpreter"]
    selected_tools = st.multiselect("Choose Tools", tool_choices, default=["BingGrounding"])
    if st.button("Restart Conversation :arrows_counterclockwise:"):
        clear_chat_history()

if user_input := st.chat_input("Ask the agent something:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(), conn_str=os.environ["PROJECT_CONNECTION_STRING"]
    )

    with project_client:
        if st.session_state.thread_id is None:
            thread = project_client.agents.create_thread()
            st.session_state.thread_id = thread.id
        else:
            thread = project_client.agents.get_thread(st.session_state.thread_id)

        bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
        bing = BingGroundingTool(connection_id=bing_connection.id)
        code_interpreter = CodeInterpreterTool()
        toolset = ToolSet()
        if "BingGrounding" in selected_tools:
            toolset.add(bing)
        if "CodeInterpreter" in selected_tools:
            toolset.add(code_interpreter)

        if st.session_state.agent_id is None:
            agent = project_client.agents.create_agent(
                model="gpt-4o",
                name="my-agent",
                instructions=instructions,
                temperature=temperature,
                toolset=toolset,
            )
            st.session_state.agent_id = agent.id
        else:
            agent = project_client.agents.get_agent(st.session_state.agent_id)

        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=user_input,
        )
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id, 
            assistant_id=agent.id,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
        )
        if run.status == "failed":
            ai_reply = f"Run failed: {run.last_error}"
        else:
            messages = project_client.agents.list_messages(thread_id=thread.id)
            last_msg = messages.get_last_text_message_by_role("assistant")
            ai_reply = last_msg.text.value if last_msg else "No response from agent"
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        with st.chat_message("assistant"):
            st.markdown(ai_reply)
