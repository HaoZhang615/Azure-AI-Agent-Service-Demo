import os
import asyncio
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from opentelemetry import trace
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents.azure_ai import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.contents import AuthorRole, ChatMessageContent

# Load environment variables
load_dotenv()

# Initialize tracing
tracer = trace.get_tracer(__name__)

# Define the termination strategy class - simpler approach from the step3 example
class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        if not history or len(history) == 0:
            return False
        return "approved" in history[-1].content.lower()

# Set up conversation management functions
conversations_path = Path("AgentGroupChat_conversations")
conversations_path.mkdir(exist_ok=True)

def load_messages(session_id):
    file_path = conversations_path / f"{session_id}.json"
    if file_path.exists():
        return json.loads(file_path.read_text())
    return []

def save_messages(session_id, messages):
    file_path = conversations_path / f"{session_id}.json"
    file_path.write_text(json.dumps(messages), encoding="utf-8")

# Create a simple function to summarize conversations
def summarize_conversation(messages):
    if not messages:
        return "New conversation"
    
    try:
        # Use the first user message as a summary
        for msg in messages:
            if msg["role"] == "user":
                summary = msg["content"][:30]
                if len(msg["content"]) > 30:
                    summary += "..."
                return summary
        return "Chat session"
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Chat session"

# Function to get an appropriate emoji avatar based on agent name
def get_agent_avatar(agent_name):
    # Default to a robot emoji if no match is found
    default_avatar = "🤖"
    
    # Map common agent names to emojis
    avatar_map = {
        "ArtDirector": "🎨",
        "CopyWriter": "✍️",
        "Reviewer": "👓",
        "Editor": "📝",
        "Designer": "🖌️",
        "Marketer": "📊",
        "Writer": "📚",
        "Assistant": "💬",
    }
    
    # Try to match the full name first
    if agent_name in avatar_map:
        return avatar_map[agent_name]
        
    # Try to match partial names
    for key in avatar_map:
        if key.lower() in agent_name.lower():
            return avatar_map[key]
            
    return default_avatar

# Set Streamlit page configuration
st.set_page_config(page_title="Azure AI Agent Group Chat", page_icon="🤖", layout="wide")
st.title("Azure AI Agent Group Chat")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)

# Sidebar configuration
with st.sidebar:
    st.header("Agent Configuration")
    
    reviewer_name = st.text_input("Reviewer Agent Name", "ArtDirector")
    reviewer_instructions = st.text_area("Reviewer Instructions", 
                                """You are an art director who has strong opinions about copywriting born of a love for David Ogilvy.
The goal is to determine if the given copy is ready to print.
If so, state that it is approved by saying the word "APPROVED". Do not use the word "APPROVED" unless it is really really good.
If not good enough, provide insight on how to refine suggested copy without example.""")
    
    copywriter_name = st.text_input("Copywriter Agent Name", "CopyWriter")
    copywriter_instructions = st.text_area("Copywriter Instructions", 
                                """You are a copywriter with 8 years of experience and are known for brevity and a dry humor.
The goal is to refine and decide on the single best copy as an expert in the field.
Only provide a single proposal per response.
You're laser focused on the goal at hand.
Don't waste time with chit chat.
Consider suggestions when refining an idea.""")
    
    max_iterations = st.slider("Maximum Iterations", 2, 20, 15)
    
    if st.button("Restart Conversation 🔄"):
        st.session_state.messages = []
        save_messages(st.session_state.session_id, [])
        st.rerun()

    # Session management
    st.write("Available Sessions:")
    sorted_files = sorted(conversations_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    for conv_file in sorted_files:
        sid = conv_file.stem
        conv_data = json.loads(conv_file.read_text())
        
        # Get summary for display
        display_name = summarize_conversation(conv_data) or sid[:8]
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

# Function to run the Azure AI Agent Group Chat - modified version
@tracer.start_as_current_span(name="ai_agent_group_chat_streamlit")
async def run_agent_group_chat(user_input):
    try:
        ai_agent_settings = AzureAIAgentSettings.create()
        
        # Initialize agents and client
        async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
            # Create the reviewer agent definition and instance
            reviewer_agent_definition = await client.agents.create_agent(
                model=ai_agent_settings.model_deployment_name,
                name=reviewer_name,
                instructions=reviewer_instructions,
            )
            agent_reviewer = AzureAIAgent(
                client=client,
                definition=reviewer_agent_definition,
            )
            
            # Create the copywriter agent definition and instance
            copy_writer_agent_definition = await client.agents.create_agent(
                model=ai_agent_settings.model_deployment_name,
                name=copywriter_name,
                instructions=copywriter_instructions,
            )
            agent_writer = AzureAIAgent(
                client=client,
                definition=copy_writer_agent_definition,
            )
            
            # Format user input for clarity
            formatted_input = f"Create a slogan for: {user_input}"
            
            # Place the agents in a group chat with the termination strategy
            chat = AgentGroupChat(
                agents=[agent_writer, agent_reviewer],
                termination_strategy=ApprovalTerminationStrategy(
                    agents=[agent_reviewer], 
                    maximum_iterations=max_iterations
                ),
            )
            
            try:
                # Add user message to the chat
                await chat.add_chat_message(message=formatted_input)
                
                # Process messages and display them one after the other
                responses = []
                async for content in chat.invoke():
                    if content.role == AuthorRole.ASSISTANT:
                        agent = content.name or "Assistant"
                        response_text = content.content
                        
                        # Display the agent's message immediately
                        with st.chat_message("assistant", avatar=get_agent_avatar(agent)):
                            st.markdown(f"**{agent}**: {response_text}")
                        
                        responses.append({
                            "agent": agent,
                            "content": response_text
                        })
                
                # Format the final response for saving to history
                formatted_responses = [f"{get_agent_avatar(r['agent'])} **{r['agent']}**: {r['content']}" for r in responses]
                final_response = "\n\n".join(formatted_responses)
                
                # Clean up resources
                await chat.reset()
                await client.agents.delete_agent(agent_reviewer.id)
                await client.agents.delete_agent(agent_writer.id)
                
                return final_response
                
            except Exception as e:
                st.error(f"Error in agent chat: {str(e)}")
                
                # Clean up resources even on error
                try:
                    await chat.reset()
                    await client.agents.delete_agent(agent_reviewer.id)
                    await client.agents.delete_agent(agent_writer.id)
                except Exception:
                    pass
                    
                return f"Error: {str(e)}"
    except Exception as e:
        return f"Configuration error: {str(e)}"

# Handle new user messages
if user_input := st.chat_input("Enter your request for the agents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process with Azure AI Agent Group Chat
    with st.spinner("Agents are collaborating on your request..."):
        response = asyncio.run(run_agent_group_chat(user_input))
        
    # Save the complete response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save the conversation
    save_messages(st.session_state.session_id, st.session_state.messages)
    
    # Force a rerun to make sure the latest messages are shown
    st.rerun()
