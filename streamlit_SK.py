import os
import asyncio
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import logging
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Load environment variables
load_dotenv()

# Define the LightsPlugin class from the quick_start example
class LightsPlugin:
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False},
        {"id": 2, "name": "Porch light", "is_on": False},
        {"id": 3, "name": "Chandelier", "is_on": True},
    ]

    @kernel_function(
        name="get_lights",
        description="Gets a list of lights and their current state",
    )
    def get_state(
        self,
    ) -> str:
        """Gets a list of lights and their current state."""
        return self.lights

    @kernel_function(
        name="change_state",
        description="Changes the state of the light",
    )
    def change_state(
        self,
        id: int,
        is_on: bool,
    ) -> str:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return light
        return None

# Set up conversation management functions
conversations_path = Path("SK_conversations")
conversations_path.mkdir(exist_ok=True)

def load_messages(session_id):
    file_path = conversations_path / f"{session_id}.json"
    if file_path.exists():
        return json.loads(file_path.read_text())
    return []

def save_messages(session_id, messages):
    file_path = conversations_path / f"{session_id}.json"
    file_path.write_text(json.dumps(messages), encoding="utf-8")

# Use Azure OpenAI to summarize the conversation - make this synchronous
def summarize_conversation(kernel, messages):
    if not messages:
        return "New conversation"
        
    try:
        # Create a simple synchronous version that doesn't require async/await
        chat_history = ChatHistory()
        # Just use the first few messages to create a summary
        for m in messages[:3]:
            if m["role"] == "user":
                chat_history.add_user_message(m["content"])
            elif m["role"] == "assistant":
                chat_history.add_assistant_message(m["content"])
        
        chat_completion = kernel.get_service(type=ChatCompletionClientBase)
        prompt = "Summarize our conversation so far into a 6-word or less title. No quotes or punctuation."
        chat_history.add_user_message(prompt)
        
        execution_settings = AzureChatPromptExecutionSettings(
            temperature=0.0,
            max_tokens=64
        )
        
        # Run the async function using asyncio.run() to get synchronous behavior
        result = asyncio.run(chat_completion.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings
        ))
        
        return str(result).strip()
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Chat session"

# Async version of the summarize function for later use
async def async_summarize_conversation(kernel, messages):
    chat_history = ChatHistory()
    for m in messages[:3]:
        if m["role"] == "user":
            chat_history.add_user_message(m["content"])
        elif m["role"] == "assistant":
            chat_history.add_assistant_message(m["content"])
    
    chat_completion = kernel.get_service(type=ChatCompletionClientBase)
    prompt = "Summarize our conversation so far into a 6-word or less title. No quotes or punctuation."
    chat_history.add_user_message(prompt)
    
    execution_settings = AzureChatPromptExecutionSettings(
        temperature=0.0,
        max_tokens=64
    )
    
    result = await chat_completion.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings
    )
    
    return str(result).strip()

# Initialize Semantic Kernel
@st.cache_resource
def initialize_kernel():
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ))
    kernel.add_plugin(LightsPlugin(), plugin_name="Lights")
    return kernel

# Set Streamlit page configuration
st.set_page_config(page_title="Semantic Kernel Smart Home Assistant", page_icon="🏠", layout="wide")
st.title("Smart Home Assistant with Semantic Kernel")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)
if "kernel" not in st.session_state:
    st.session_state.kernel = initialize_kernel()

# Sidebar configuration
with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    system_prompt = st.text_area("System Prompt", 
                                "You are a helpful home automation assistant. You can control lights and provide information about their status.")
    
    if st.button("Restart Conversation 🔄"):
        st.session_state.messages = []
        save_messages(st.session_state.session_id, [])

    st.write("Available Sessions:")
    sorted_files = sorted(conversations_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    for conv_file in sorted_files:
        sid = conv_file.stem
        conv_data = json.loads(conv_file.read_text())
        
        # Get summary synchronously
        if len(conv_data) > 0:
            display_name = summarize_conversation(st.session_state.kernel, conv_data) or sid[:8]
            cols = st.columns([3,1])
            
            if cols[0].button(display_name, key=f"switch_{sid}"):
                st.session_state.session_id = sid
                st.session_state.messages = load_messages(sid)
                st.rerun()

            if cols[1].button("❌", key=f"delete_{sid}"):
                conv_file.unlink(missing_ok=True)
                st.rerun()
                
    if st.button("New Thread"):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.session_state.messages = []
        save_messages(new_id, [])
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process user input
async def process_message(kernel, user_input, messages):
    # Create a ChatHistory from the messages
    chat_history = ChatHistory()
    
    # Add system message
    chat_history.add_system_message(system_prompt)
    
    # Add previous messages to history
    for m in messages:
        if m["role"] == "user":
            chat_history.add_user_message(m["content"])
        elif m["role"] == "assistant":
            chat_history.add_assistant_message(m["content"])
    
    # Add current user input
    chat_history.add_user_message(user_input)
    
    # Setup execution settings
    execution_settings = AzureChatPromptExecutionSettings(
        temperature=temperature,
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )
    
    # Get the chat completion service
    chat_completion = kernel.get_service(type=ChatCompletionClientBase)
    
    # Get response
    result = await chat_completion.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )
    
    return str(result)

# Handle new user messages
if user_input := st.chat_input("Ask about your smart home..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Process the message using Semantic Kernel
        with st.spinner("Thinking..."):
            response = asyncio.run(
                process_message(
                    st.session_state.kernel,
                    user_input,
                    st.session_state.messages
                )
            )
        
        message_placeholder.markdown(response)
    
    # Add assistant response to messages
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save the conversation
    save_messages(st.session_state.session_id, st.session_state.messages)