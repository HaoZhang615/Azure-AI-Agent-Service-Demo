import os
import asyncio
import aiohttp
from typing import Annotated, Dict, List, Optional
from pathlib import Path, PurePath
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential
import logging

# Get the absolute path to the .env file in the semantic_kernal folder
current_dir = Path(__file__).parent
env_path = PurePath(current_dir).joinpath(".env")

print(f"Loading environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path)

# Add logging to verify environment variables are loaded
print(f"Loaded environment variables from: {env_path}")
print(f"BING_SEARCH_API_KEY: {os.getenv('BING_SEARCH_API_KEY')}")

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function

# Bing Search Configuration
bing_api_key = os.getenv("BING_SEARCH_API_KEY")
has_bing_api_key = bing_api_key is not None and bing_api_key != ''
bing_api_endpoint = os.getenv("BING_SEARCH_API_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

# CosmosDB Configuration
credential = DefaultAzureCredential()
cosmos_endpoint = os.getenv("COSMOSDB_ENDPOINT")
cosmos_client = CosmosClient(cosmos_endpoint, credential)
database_name = os.getenv("COSMOSDB_DATABASE")
database = cosmos_client.create_database_if_not_exists(id=database_name)
product_container_name = "Product"
producturl_container = database.get_container_client(os.getenv("COSMOSDB_ProductUrl_CONTAINER"))

def get_target_company():
    # use the first item in the Product container and get the value of the field "company"
    database = cosmos_client.get_database_client(database_name)
    container = database.get_container_client(product_container_name)
    try:
        # Query the container for the first item
        items = list(container.read_all_items())
        if items:
            return items[0]["company"]
        else:
            return None
    except exceptions.CosmosResourceNotFoundError as e:
        logging.error(f"CosmosHttpResponseError: {e}")
        return None

class WebSearchPlugin:
    """A plugin for searching the web using Bing."""
    
    @kernel_function(description="Search Bing for relevant web information.")
    async def search_web(
        self,
        query: Annotated[str, "The search query in concise terms that works efficiently in Bing Search."],
        up_to_date: Annotated[bool, "Whether up-to-date information is needed."] = False
    ) -> Annotated[str, "Search results from the web."]:
        if not has_bing_api_key:
            return "Web search is currently unavailable because no Bing Search API key was provided."
        
        try:
            # Retrieve the single ProductUrl document from CosmosDB
            items = list(producturl_container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True
            ))
            if items:
                restricted_urls = items[0].get('urls', [])
                if restricted_urls:
                    concatenated_url_string = " OR ".join(
                        f"site:{url.replace('https://','').replace('http://','')}" for url in restricted_urls
                    )
                    query = f"{query} {concatenated_url_string}"
        except Exception as ex:
            logging.error(f"Error querying ProductUrl container: {ex}")

        headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
        params = {"q": query, "count": 3}
        url = bing_api_endpoint
        if up_to_date:
            params.update({"sortby": "Date"})
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    search_results = await response.json()
                    results = []
                    if search_results and "webPages" in search_results and "value" in search_results["webPages"]:
                        for v in search_results["webPages"]["value"]:
                            result = {
                                "source_title": v["name"],
                                "content": v["snippet"],
                                "source_url": v["url"]
                            }
                            results.append(result)
                    formatted_result = "\n".join([
                        f'{i}. content: {item["content"]}, source_title: {item["source_title"]}, source_url: {item["source_url"]}' 
                        for i, item in enumerate(results, 1)
                    ])
                    return formatted_result
        except Exception as ex:
            return f"Error during web search: {str(ex)}"

# Sample user queries for demonstration
USER_INPUTS = [
    "what are some of the latest news about Unilever?",
]

async def main():
    target_company = get_target_company() or "our company"
    
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "web_search_agent"
    kernel = Kernel()
    kernel.add_plugin(WebSearchPlugin(), plugin_name="web")
    
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
        name="Web_Search_Agent",
        instructions=f"""
        You are a web search agent that queries Bing search engine to retrieve up-to-date information from the Internet.
        If user is asking about any related products or services from {target_company}'s competitors or from the same industry, you promote {target_company}'s equivalent products by directing to search the same category across {target_company}'s portfolio.
        
        Your tasks are:
        - Use the 'search_web' function to look up the web to find the solution to the user's issue directly, or search web to help as part of the whole solution.
        - Act politely and professionally. If the source is a famous and credible website, you can mention it to the user e.g. 'according to <source>'.
        
        Interaction goes over voice, so it's *super* important that answers are as short as possible.
        """,
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

    # After predetermined inputs, switch to interactive mode
    print("\nSwitching to interactive mode. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        chat_history.add_user_message(user_input)
        
        async for content in agent.invoke(chat_history):
            print(f"{content.name}: ", end="")
            if (
                not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items)
                and content.content.strip()
            ):
                print(f"{content.content}", end="", flush=True)
        print("")

if __name__ == "__main__":
    asyncio.run(main())