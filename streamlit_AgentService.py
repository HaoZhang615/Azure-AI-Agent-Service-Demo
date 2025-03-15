import os
import streamlit as st
import re
import shutil
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.projects.models import CodeInterpreterTool, BingGroundingTool, ToolSet
from azure.identity import DefaultAzureCredential
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
import json
import uuid

# Load environment variables from .env file
load_dotenv()
conn_str = os.environ["PROJECT_CONNECTION_STRING"]
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), 
    conn_str=conn_str
)
# instantiate Azure OpenAI client
AOAI_client = ChatCompletionsClient(
    endpoint=os.environ["AOAI_ENDPOINT"],  # Of the form https://<your-resouce-name>.openai.azure.com/openai/deployments/<your-deployment-name>
    credential=AzureKeyCredential(os.environ["AOAI_KEY"]),
    api_version="2024-10-21",  # Azure OpenAI api-version. See https://aka.ms/azsdk/azure-ai-inference/azure-openai-api-versions
)
# function to clear chat history
def clear_chat_history():
    """Clears all chat history and resets session states."""
    if st.session_state.agent_id:
        project_client.agents.delete_agent(st.session_state.agent_id)
        st.session_state.agent_id = None
    if st.session_state.thread_id:
        project_client.agents.delete_thread(st.session_state.thread_id)
        st.session_state.thread_id = None
    st.session_state.messages = []
    st.rerun()
# Prepare markdown for the image (assumes one image)
def st_markdown(markdown_string):
    """
    Custom markdown function to handle images and escape {}.
    """
    
    parts = re.split(
        r"!\[(.*?)\]\((.*?)\)", 
        markdown_string
    )
    for i, part in enumerate(parts):
        if i % 3 == 0:
            st.markdown(part)
        elif i % 3 == 2:
            st.image(part)  # Add caption if you want -> , caption=title)
st.set_page_config(page_title="Azure AI Foundry powered Agent Service Chatbot", page_icon=":cloud:", layout="wide")
st.title("Azure AI Agent Service Chat")

agent_conversations_path = Path("agent_conversations").resolve()
agent_conversations_path.mkdir(exist_ok=True)

def load_session(session_id):
    file_path = agent_conversations_path / f"{session_id}.json"
    if file_path.exists():
        data = json.loads(file_path.read_text())
        st.session_state.agent_id = data.get("agent_id", None)
        st.session_state.thread_id = data.get("thread_id", None)
        st.session_state.messages = data.get("messages", [])
    else:
        st.session_state.agent_id = None
        st.session_state.thread_id = None
        st.session_state.messages = []

def save_session(session_id):
    file_path = agent_conversations_path / f"{session_id}.json"
    data = {
        "agent_id": st.session_state.agent_id,
        "thread_id": st.session_state.thread_id,
        "messages": st.session_state.messages
    }
    file_path.write_text(json.dumps(data), encoding="utf-8")

# use Azure OpenAI gpt-4o-mini to summarize the conversation 
# into a short sentence of no more than 6 words
def summarize_conversation(messages):
    prompt = """Summarize the conversation so far into a 6-word or less title. 
    Do not use any quotation marks or punctuation. 
    Do not include any other commentary or description."""
    formatted_messages = [{"role": "user", "content": m["content"]} for m in messages if m["role"] == "user"]
    formatted_messages.append({"role": "user", "content": prompt})
    response = AOAI_client.complete(
        messages=formatted_messages, 
        max_tokens=64, 
        temperature=0)
    return response.choices[0].message.content.strip()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_id" not in st.session_state:
    st.session_state.agent_id = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(st_markdown(message["content"]))  # call st_markdown() for images

with st.sidebar:
    st.subheader("Agent Settings")
    instructions = st.text_area("Instructions", "You are a helpful agent")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_prompt_tokens = st.slider("Max Prompt Tokens", min_value=500, max_value=100000, value=20000, step=100)
    max_completion_tokens = st.slider("Max Completion Tokens", min_value=1000, max_value=100000, value=1000, step=100)
    tool_choices = ["BingGrounding", "CodeInterpreter"]
    selected_tools = st.multiselect("Choose Tools", tool_choices, default=["BingGrounding", "CodeInterpreter"])
    if st.button("Restart Conversation :arrows_counterclockwise:"):
        clear_chat_history()

    st.write("Available Sessions:")
    sorted_files = sorted(agent_conversations_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True) # sort by last modified time
    for conv_file in sorted_files:
        sid = conv_file.stem
        session_data = json.loads(conv_file.read_text())
        short_summary = summarize_conversation(session_data.get("messages", []))
        cols = st.columns([3,1])
        if cols[0].button(short_summary, key=f"switch_{sid}"):
            load_session(sid)
        if cols[1].button("delete", key=f"delete_{sid}"):
            conv_file.unlink(missing_ok=True)
            session_folder = agent_conversations_path / sid
            if session_folder.exists() and session_folder.is_dir():
                shutil.rmtree(session_folder)
            st.rerun()
    if st.button("New Thread"):
        new_id = str(uuid.uuid4())
        st.session_state.agent_id = None
        st.session_state.thread_id = None
        st.session_state.session_id = new_id
        st.session_state.messages = []
        save_session(new_id)
        st.rerun()

if user_query := st.chat_input("Ask the agent something:"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    progress_indicator = st.progress(0)
    st.session_state.progress = 0
    
    with project_client:
        if st.session_state.thread_id is None:
            thread = project_client.agents.create_thread()
            st.session_state.thread_id = thread.id
            st.session_state.progress += 25
            progress_indicator.progress(st.session_state.progress, "creating new thread...")
        else:
            thread = project_client.agents.get_thread(st.session_state.thread_id)
            st.session_state.progress += 25
            progress_indicator.progress(st.session_state.progress, "thinking...")

        bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
        bing = BingGroundingTool(connection_id=bing_connection.id)
        code_interpreter = CodeInterpreterTool()
        toolset = ToolSet()
        if "BingGrounding" in selected_tools:
            toolset.add(bing)
        if "CodeInterpreter" in selected_tools:
            toolset.add(code_interpreter)
        st.session_state.progress += 25
        progress_indicator.progress(st.session_state.progress, "adding tools...")

        if st.session_state.agent_id is None:
            agent = project_client.agents.create_agent(
                model="gpt-4o",
                name="my-agent",
                instructions=instructions,
                temperature=temperature,
                toolset=toolset,
            )
            st.session_state.agent_id = agent.id
            st.session_state.progress += 25
            progress_indicator.progress(st.session_state.progress, "initializing new agent...")
        else:
            agent = project_client.agents.get_agent(st.session_state.agent_id)

        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=user_query,
        )
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id, 
            assistant_id=agent.id,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
        )
        if run.status == "failed":
            agent_response = f"Run failed: {run.last_error}"
        else:
            messages = project_client.agents.list_messages(thread_id=thread.id)
            last_msg = messages.get_last_text_message_by_role("assistant")
            if last_msg:
                agent_response = last_msg.text.value
                if "BingGrounding" in selected_tools and last_msg.text.annotations:
                    st.session_state.progress += 25
                    progress_indicator.progress(st.session_state.progress, "Grounding using Bing...")
                    citations = []
                    for annotation in last_msg.text.annotations:
                        citation_url = annotation.get("url_citation", {}).get("url")
                        if citation_url:
                            citations.append(f"{annotation.text}: {citation_url}")
                    if citations:
                        agent_response += "\n\n### Citations:\n" + "\n".join(f"- {c}" for c in citations)
                images_found = False
                python_code = None
                image_md = ""
                if "CodeInterpreter" in selected_tools:
                    last_message_content = messages["data"][0].get("content", [None])
                    images_found = any("image_contents" in msg for msg in last_message_content)
                    if messages.image_contents:
                        for image_content in messages.image_contents:
                            images_found = True
                            file_id = image_content.image_file.file_id
                            images_dir = agent_conversations_path / st.session_state.session_id / "images"
                            images_dir.mkdir(parents=True, exist_ok=True)
                            file_name = images_dir / f"code_interpreter_result_{uuid.uuid4().hex[:8]}.png"
                            project_client.agents.save_file(
                                file_id=file_id,
                                file_name=file_name.name,  # use only the filename
                                target_dir=str(file_name.parent.resolve())  # specify target folder
                            )
                            image_md = f"![image]({file_name.as_posix()})"
                        st.session_state.progress += 25
                        progress_indicator.progress(st.session_state.progress, "Executing Code Interpreter...")
                    run_details = project_client.agents.list_run_steps(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                    for steps in run_details.data:
                        if getattr(steps.step_details, 'type', None) == "tool_calls":
                            for calls in steps.step_details.tool_calls:
                                input_value = getattr(calls.code_interpreter, 'input', None) if hasattr(calls, 'code_interpreter') else None
                                if input_value:
                                    python_code = input_value
                # Combine responses in one message.
                combined_response = agent_response
                if images_found:
                    combined_response += "\n\n" + image_md
                if python_code:
                    combined_response += "\n\n**Used Python Code Snippet:**\n```python\n" + python_code + "\n```"
                    #print(combined_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": combined_response
                })
                
                with st.chat_message("assistant"):
                    st.markdown(st_markdown(combined_response))
                    print(st_markdown(combined_response))
            else:
                agent_response = "No response from agent"
    save_session(st.session_state.session_id)
    st.rerun()
    progress_indicator.empty()