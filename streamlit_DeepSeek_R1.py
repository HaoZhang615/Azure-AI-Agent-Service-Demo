import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.search.documents import SearchClient
import streamlit as st
import json
import uuid
from pathlib import Path

load_dotenv()

# instantiate the DeepSeek R1 client
DS_client = ChatCompletionsClient(
    endpoint=os.environ["DeepSeek_R1_Endpoint"],
    credential=AzureKeyCredential(os.environ["DeepSeek_R1_Key"]),
    model="DeepSeek-R1"
)

# instantiate Azure OpenAI client
AOAI_client = ChatCompletionsClient(
    endpoint=os.environ["AOAI_ENDPOINT"],  # Of the form https://<your-resouce-name>.openai.azure.com/openai/deployments/<your-deployment-name>
    credential=AzureKeyCredential(os.environ["AOAI_KEY"]),
    api_version="2024-10-21",  # Azure OpenAI api-version. See https://aka.ms/azsdk/azure-ai-inference/azure-openai-api-versions
)


# Set the page configuration
st.set_page_config(page_title="Azure AI Foundry powered DeepSeek R1 Chatbot", page_icon=":whale2:", layout="wide")
st.title("Azure AI Foundry powered DeepSeek R1 Chatbot")

conversations_path = Path("conversations")
conversations_path.mkdir(exist_ok=True)

def load_messages(session_id):
    file_path = conversations_path / f"{session_id}.json"
    if file_path.exists():
        return json.loads(file_path.read_text())
    return []

def save_messages(session_id, messages):
    file_path = conversations_path / f"{session_id}.json"
    # Remove the <think>...</think> block from the assistant's response
    for message in messages:
        if message["role"] == "assistant":
            start = message["content"].find("<think>")
            end = message["content"].find("</think>") + len("</think>")
            if start != -1 and end != -1:
                message["content"] = message["content"][:start] + message["content"][end:]
    file_path.write_text(json.dumps(messages), encoding="utf-8")

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
        temperature=0.9)
    return response.choices[0].message.content.strip()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)

with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
    max_tokens = st.slider("Max. Tokens", 10, 4096, 2048, 64)
    system_prompt = st.text_area("System Prompt", "You are a helpful assistant.")
    
    def clear_chat():
        st.session_state.messages = []
    if st.button("Restart Conversation :arrows_counterclockwise:"):
        clear_chat()

    st.write("Available Sessions:")
    for conv_file in conversations_path.glob("*.json"):
        sid = conv_file.stem
        conv_data = json.loads(conv_file.read_text())
        short_summary = summarize_conversation(conv_data)
        cols = st.columns([3,1])
        if cols[0].button(short_summary or sid, key=f"switch_{sid}"):
            st.session_state.session_id = sid
            st.session_state.messages = load_messages(sid)
        if cols[1].button("Close", key=f"close_{sid}"):
            conv_file.unlink(missing_ok=True)
    if st.button("New Thread"):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.session_state.messages = []
        save_messages(new_id, [])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user messages
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Prepend the system message to the list of messages
        messages = [SystemMessage(content=system_prompt)]
        for m in st.session_state.messages:
            if m["role"] == "user":
                messages.append(UserMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AssistantMessage(content=m["content"]))
        stream = DS_client.complete(
            stream=True,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_buffer = ""
        def stream_generator():
            global response_buffer
            for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    response_buffer += content  # Accumulate response
                    yield content 
        st.write_stream(stream_generator)  # Display response in Streamlit
        st.session_state.messages.append({"role": "assistant", "content": response_buffer})
        save_messages(st.session_state.session_id, st.session_state.messages)
