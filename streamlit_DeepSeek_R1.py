import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
import streamlit as st

load_dotenv()

# For Serverless API or Managed Compute endpoints
client = ChatCompletionsClient(
    endpoint=os.environ["DeepSeek_R1_Endpoint"],
    credential=AzureKeyCredential(os.environ["DeepSeek_R1_Key"])
)

# Set the page configuration
st.set_page_config(page_title="Azure AI Foundry powered DeepSeek R1 Chatbot", page_icon=":whale2:", layout="wide")
st.title("Azure AI Foundry powered DeepSeek R1 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
    max_tokens = st.slider("Max. Tokens", 10, 4096, 2048, 64)
    system_prompt = st.text_area("System Prompt", "You are a helpful assistant.")
    
    def clear_chat():
        st.session_state.messages = []
    if st.button("Restart Conversation :arrows_counterclockwise:"):
        clear_chat()

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
        stream = client.complete(
            stream=True,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model="DeepSeek-R1"
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
