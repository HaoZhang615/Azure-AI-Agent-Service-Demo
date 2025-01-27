from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Azure OpenAI Chatbot")

# Initialize session state attributes  
if "messages" not in st.session_state:  
    st.session_state.messages = []  

# Sidebar Configuration
with st.sidebar:
    # input box for user to enter their Azure OpenAI endpoint
    if 'Azure_OpenAI_Endpoint' in st.secrets:
        st.success('Azure OpenAI endpoint already provided!', icon='✅')
        azure_endpoint = st.secrets['Azure_OpenAI_Endpoint']
    else:
        azure_endpoint = st.text_input('Enter Azure OpenAI endpoint: e.g. https://xxxx.openai.azure.com/', type='password')

    # input box for user to enter their Azure OpenAI key
    if 'Azure_OpenAI_Key' in st.secrets:
        st.success('Azure OpenAI API Key already provided!', icon='✅')
        api_key = st.secrets['Azure_OpenAI_Key']
    else:
        api_key = st.text_input('Enter Azure OpenAI API key:', type='password')

    # input box for user to enter their Azure OpenAI model deployment name
    if 'Azure_OpenAI_Model_Deployment_Name' in st.secrets:
        st.success('Azure OpenAI model deployment name already provided!', icon='✅')
        model_name = st.secrets['Azure_OpenAI_Model_Deployment_Name']
    else:
        model_name = st.text_input('Enter Azure OpenAI model deployment name:')

    # Temperature and token slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    Max_Token = st.sidebar.slider(
        "Max. Tokens",
        min_value=10,
        max_value=4096,
        value=256,
        step=64
    )
    # default system message
    default_system_message = "You are an helpful AI assistant."
    # make customizable system prompt
    system_prompt = st.sidebar.text_area("System Prompt", default_system_message, height=200)

    def clear_chat_history():
        st.session_state.messages = []
        st.session_state.chat_history = []
    if st.button("Restart Conversation :arrows_counterclockwise:"):
        clear_chat_history()

# Azure Open AI Configuration
api_version = "2024-02-01"
client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    azure_endpoint = azure_endpoint,
)
  
# Display chat history  
for message in st.session_state.messages:  
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])  

# Handle new message  
if prompt := st.chat_input("type here..."):  
    st.session_state.messages.append({"role": "user", "content": prompt})  
    with st.chat_message("user"):  
        st.markdown(prompt)  
    with st.chat_message("assistant"):  
        # Define the system message
        system_message = {"role": "system", "content": system_prompt}
        
        # Prepend the system message to the list of messages
        messages = [system_message] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        stream = client.chat.completions.create(  
            model=model_name,  
            messages=messages,  
            stream=True,  
            temperature=temperature,  
            max_tokens=Max_Token  
        )  
        response = st.write_stream(stream)  
        st.session_state.messages.append({"role": "assistant", "content": response})
