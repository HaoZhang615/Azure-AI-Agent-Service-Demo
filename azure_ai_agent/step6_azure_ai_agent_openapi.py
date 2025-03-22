# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os
from typing import Dict
from dotenv import load_dotenv
import aiohttp  # Add import for async HTTP requests

from azure.ai.projects.models import OpenApiAnonymousAuthDetails, OpenApiTool
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents.azure_ai import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.contents import AuthorRole

"""
The following sample demonstrates how to create a simple, Azure AI agent that
uses OpenAPI tools to answer user questions.
"""

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
ip_geolocation_key = os.environ.get("IP_GEOLOCATION_KEY")

# Simulate a conversation with the agent
USER_INPUTS = [
    # "What's my current location?",
    "What is the current weather at my location?",
]

# Function to get the current public IP address
async def get_public_ip():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.ipify.org?format=json") as response:
            if response.status == 200:
                data = await response.json()
                return data.get("ip")
            return None


async def main() -> None:
    ai_agent_settings = AzureAIAgentSettings.create()
    
    # Get the current public IP address
    current_ip = await get_public_ip()
    # print(f"# Current public IP: {current_ip}")

    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 1. Read in the OpenAPI spec files
        openapi_spec_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "azure_ai_agent",
            "resources",
        )
        with open(os.path.join(openapi_spec_file_path, "weather.json")) as weather_file:
            weather_openapi_spec = json.loads(weather_file.read())
        
        # Load and modify the IP geolocation spec to include the API key and current IP
        with open(os.path.join(openapi_spec_file_path, "ip_geo_location.json")) as ip_geo_location_file:
            ip_geo_location_openapi_spec = json.loads(ip_geo_location_file.read())
            
            # Modify the parameters to include default values
            for param in ip_geo_location_openapi_spec["paths"]["/ipgeo"]["get"]["parameters"]:
                if param["name"] == "apiKey":
                    # Set API key as default in the schema
                    param["schema"]["default"] = ip_geolocation_key
                
                if param["name"] == "ip":
                    # Set current IP as default and indicate it in description
                    param["schema"]["default"] = current_ip
                    param["description"] = f"The IP address to look up. Use this exact IP: {current_ip}"

        # 2. Create OpenAPI tools
        # Weather API with anonymous auth
        weather_auth = OpenApiAnonymousAuthDetails()
        openapi_weather = OpenApiTool(
            name="get_weather",
            spec=weather_openapi_spec,
            description="Retrieve weather information for a specific city or location name",
            auth=weather_auth,
        )
        
        # IP Geolocation API with anonymous auth (API key and IP are now embedded in the spec)
        ip_geo_auth = OpenApiAnonymousAuthDetails()
        openapi_ip_geo_location = OpenApiTool(
            name="get_ip_geo_location",
            spec=ip_geo_location_openapi_spec,
            description=f"Determines geolocation based on an IP address. Use this to get the user's current location based on IP: {current_ip}",
            auth=ip_geo_auth,
        )

        # Create a system message that instructs the agent how to use these tools
        system_message = f"""
        You are a helpful assistant that can provide location and weather information.
        
        IMPORTANT: When asked about the user's current location or local weather:
        1. First use the get_ip_geo_location tool to determine the user's location from their IP address
        2. The user's IP address is {current_ip} - always use this exact value
        3. Then use the get_weather tool with the location information if weather is requested
        
        Do not try to guess the user's location. ALWAYS use the get_ip_geo_location tool to get accurate information.
        """

        # 3. Create an agent on the Azure AI agent service with the OpenAPI tools
        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            instructions=system_message,
            tools=openapi_weather.definitions + openapi_ip_geo_location.definitions,
        )

        # 4. Create a Semantic Kernel agent for the Azure AI agent
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
        )

        # 5. Create a new thread on the Azure AI agent service
        thread = await client.agents.create_thread()

        try:
            for user_input in USER_INPUTS:
                # 6. Add the user input as a chat message
                await agent.add_chat_message(thread_id=thread.id, message=user_input)
                print(f"# User: '{user_input}'")
                # 7. Invoke the agent for the specified thread for response
                async for content in agent.invoke(thread_id=thread.id):
                    if content.role != AuthorRole.TOOL:
                        print(f"# Agent: {content.content}")
        finally:
            # 8. Cleanup: Delete the thread and agent
            await client.agents.delete_thread(thread.id)
            await client.agents.delete_agent(agent.id)


if __name__ == "__main__":
    asyncio.run(main())
