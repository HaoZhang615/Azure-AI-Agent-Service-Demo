import os
import asyncio
from typing import Annotated
from pathlib import Path, PurePath
from dotenv import load_dotenv

# Get the absolute path to the .env file in the semantic_kernal folder
current_dir = Path(__file__).parent
env_path = PurePath(current_dir).joinpath(".env")

print(f"Loading environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path)

# Add logging to verify environment variables are loaded
print(f"Loaded environment variables from: {env_path}")
print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function

# Set up Azure Search client
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
credential = DefaultAzureCredential() if key is None or key == "" else AzureKeyCredential(key)
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=credential
)

# Define a plugin for internal knowledge base queries
class InternalKnowledgeBasePlugin:
    """A plugin for querying the internal knowledge base."""

    @kernel_function(description="Query the internal knowledge base for information.")
    async def query_kb(
        self, query: Annotated[str, "The query to look up in the internal knowledge base."]
    ) -> Annotated[str, "Returns information from the internal knowledge base."]:
        vector_query = VectorizableTextQuery(
            text=query, 
            k_nearest_neighbors=3, 
            fields="text_vector", 
            exhaustive=True
        )
        search_results = search_client.search(
            search_text=query,  
            vector_queries=[vector_query],
            select=["title", "chunk_id", "chunk"],
            top=3
        )

        # Chunk id has format {parent_id}_pages_{page_number}
        sources_formatted = "\n".join([
            f'# Source "{document["title"]}" - Page {document["chunk_id"].split("_")[-1]}\n{document["chunk"]}' 
            for document in search_results
        ])
        
        return sources_formatted


# Sample user queries for demonstration
USER_INPUTS = [
    "Hello, tell me some of the key summary from our financial report.",
    # "What was some of the highlights from the sustainability perspective?",
    # "Thank you for your help"
]

async def main():
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "kb_agent"
    kernel = Kernel()
    kernel.add_plugin(InternalKnowledgeBasePlugin(), plugin_name="kb")
    
    print (f"deployment name: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print (f"endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print (f"api version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-06-01')}")
    
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        )
    )

    # 2. Configure the function choice behavior to auto invoke kernel functions
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 3. Create the agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Internal_KB_Assistant",  # Changed from "Internal KB Assistant"
        instructions="""You are an internal knowledge base assistant that responds to inquiries about the company's internal knowledge.
        Use the knowledge base plugin to look up information when needed.
        Make sure to act politely and professionally.""",
        arguments=KernelArguments(settings=settings),
    )

    # 4. Create a chat history to hold the conversation
    chat_history = ChatHistory()

    for user_input in USER_INPUTS:
        # 5. Add the user input to the chat history
        chat_history.add_user_message(user_input)
        print(f"# User: {user_input}")
        # 6. Invoke the agent for a response
        async for content in agent.invoke(chat_history):
            print(f"# {content.name}: ", end="")
            if (
                not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items)
                and content.content.strip()
            ):
                # We only want to print the content if it's not a function call or result
                print(f"{content.content}", end="", flush=True)
        print("")


if __name__ == "__main__":
    asyncio.run(main())