import os
import asyncio
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from opentelemetry import trace
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.azure_ai import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

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

# Function to create a Kernel instance with Azure OpenAI service
def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_GPT4o_MINI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ))
    return kernel

# Set Streamlit page configuration
st.set_page_config(page_title="Azure AI Agent Group Chat", page_icon="🤖", layout="wide")
st.title("Azure AI Agent Group Chat")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)
if "show_inner_monologue" not in st.session_state:
    st.session_state.show_inner_monologue = True

# Sidebar configuration
with st.sidebar:
    st.header("Agent Configuration")
    
    # Toggle for showing/hiding inner monologue
    st.session_state.show_inner_monologue = st.checkbox("Show Agent Inner Monologue", 
                                                     value=st.session_state.show_inner_monologue)
    
    reviewer_name = st.text_input("Reviewer Agent Name", "Reviewer")
    # Set the default reviewer instructions
    reviewer_instructions = st.text_area("Reviewer Instructions", 
                                """You are an art director who has strong opinions about copywriting born of a love for David Ogilvy.
The goal is to determine if the given copy is ready to print.
If so, state that it is approved by saying the word "APPROVED". Do not use the word "APPROVED" unless it is really really good.
If not good enough, provide insight on how to refine suggested copy without example.""")
    
    copywriter_name = st.text_input("Copywriter Agent Name", "CopyWriter")
    # Set the default copywriter instructions
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
    # Check if the message has inner monologue data
    if "role" in msg and msg["role"] == "assistant" and "inner_monologue" in msg and st.session_state.show_inner_monologue:
        # Display the inner monologue in an expander
        with st.expander("💭 Agents' Inner Monologue", expanded=True):
            st.markdown(msg["inner_monologue"])
        
        # Display just the final response (if available)
        if "final_response" in msg:
            with st.chat_message("assistant"):
                st.markdown(msg["final_response"])
    else:
        # Display normal messages
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Function to run the Azure AI Agent Group Chat - modified version
@tracer.start_as_current_span(name="ai_agent_group_chat_streamlit")
async def run_agent_group_chat(user_input):
    try:
        # Create a single kernel instance for all agents.
        kernel = create_kernel()
        az_ai_agent_settings = AzureAIAgentSettings.create()
        # Initialize agents and client for Azure AI Agent
        async with (
            DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(credential=creds) as client,
        ):
            # Create reviewer agent using ChatCompletionAgents approach.
            agent_reviewer = ChatCompletionAgent(
                kernel=kernel,
                name=reviewer_name,
                instructions=reviewer_instructions,
            )
            # Create the copywriter agent using Azure AI Agent Service approach.
            copy_writer_agent_definition = await client.agents.create_agent(
                model=az_ai_agent_settings.model_deployment_name,
                name=copywriter_name,
                instructions=copywriter_instructions,
            )
            agent_writer = AzureAIAgent(
                client=client,
                definition=copy_writer_agent_definition,
            )
            
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
                await chat.add_chat_message(message=user_input)
                
                # Create a container for live updates during processing
                live_update_container = st.empty()
                
                # Process messages and display them one after the other
                responses = []
                final_response = None
                final_agent = None
                is_approved = False
                
                async for content in chat.invoke():
                    if content.role == AuthorRole.ASSISTANT:
                        agent = content.name or "Assistant"
                        response_text = content.content
                        
                        # Add to responses list
                        responses.append({
                            "agent": agent,
                            "content": response_text
                        })
                        
                        # Check if this is the approval message
                        if agent == reviewer_name and "approved" in response_text.lower():
                            is_approved = True
                            final_agent = agent
                            final_response = response_text
                        
                        # If not approved yet, record the latest response from the copywriter
                        if not is_approved and agent == copywriter_name:
                            final_agent = agent
                            final_response = response_text
                        
                        # Display only the current message in the live update container
                        with live_update_container.container():
                            st.write(f"💭 {get_agent_avatar(agent)} **{agent}** is thinking...")
                            st.markdown(f"{response_text}")
                
                # Clear the live update container once processing is complete
                live_update_container.empty()
                
                # Prepare the inner monologue for saving
                formatted_responses = [f"{get_agent_avatar(r['agent'])} **{r['agent']}**: {r['content']}" for r in responses]
                inner_monologue_text = "\n\n".join(formatted_responses)
                
                # Prepare the final response
                if final_response:
                    if is_approved:
                        # For approved content, extract the copywriter's latest proposal
                        copywriter_responses = [r for r in responses if r["agent"] == copywriter_name]
                        if copywriter_responses:
                            final_proposal = copywriter_responses[-1]["content"]
                            final_result = f"**Final Approved Copy:**\n\n{final_proposal}\n\n{get_agent_avatar(reviewer_name)} *Approved by {reviewer_name}*"
                        else:
                            final_result = f"{get_agent_avatar(final_agent)} **{final_agent}**: {final_response}"
                    else:
                        # For non-approved content, show the latest message
                        final_result = f"{get_agent_avatar(final_agent)} **{final_agent}**: {final_response}"
                else:
                    final_result = "No response generated."
                
                await chat.reset()
                
                # Return both the inner monologue and final result
                return {
                    "inner_monologue": inner_monologue_text,
                    "final_response": final_result
                }
                
            except Exception as e:
                st.error(f"Error in agent chat: {str(e)}")
                try:
                    await chat.reset()
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
        
    # Save the response (could be a string error message or a dict with inner_monologue and final_response)
    if isinstance(response, dict):
        # Save structured response with inner monologue and final response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["final_response"],  # for backward compatibility
            "inner_monologue": response["inner_monologue"],
            "final_response": response["final_response"]
        })
    else:
        # Save error message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save the conversation
    save_messages(st.session_state.session_id, st.session_state.messages)
    
    # Force a rerun to make sure the latest messages are shown
    st.rerun()
