import os
import asyncio
import requests
from typing import Annotated
from pathlib import Path, PurePath
from dotenv import load_dotenv

# Get the absolute path to the .env file in the semantic_kernal folder
current_dir = Path(__file__).parent
env_path = PurePath(current_dir).joinpath(".env")

print(f"Loading environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path)

print(f"Loaded environment variables from: {env_path}")
print(f"SEND_EMAIL_LOGIC_APP_URL: {os.getenv('SEND_EMAIL_LOGIC_APP_URL')}")

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function

SEND_EMAIL_LOGIC_APP_URL = os.getenv("SEND_EMAIL_LOGIC_APP_URL")

# Define a plugin for email assistant functionality
class EmailAssistantPlugin:
    """A plugin for sending emails and providing conversation recap."""

    @kernel_function(description="Send an email to the specified user.")
    async def send_email(
        self, 
        to: Annotated[str, "The recipient's email address."],
        subject: Annotated[str, "The subject of the email."],
        body: Annotated[str, "The body of the email."]
    ) -> Annotated[str, "Result of the email sending operation."]:
        try:
            params = {"to": to, "subject": subject, "body": body}
            res = requests.post(SEND_EMAIL_LOGIC_APP_URL, json=params)
            res.raise_for_status()
            return "Email sent successfully."
        except Exception as e:
            return f"Failed to send email: {e}"

# Sample user queries for demonstration
USER_INPUTS = [
    """I need to send an email to hao.zhang@microsoft.com about our meeting tomorrow as a reminder. 
    Use the subject 'urgent meeting reminder', body 'please come tomorrow on time at 9 am'.
    you don't need to ask me for confirmation, just send it.""",
]
async def main():
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "email_assistant"
    kernel = Kernel()
    kernel.add_plugin(EmailAssistantPlugin(), plugin_name="email")
    
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
        name="Executive_Assistant",
        instructions="""You are an executive assistant that helps with administrative tasks.
        Interaction goes over voice, so it's *super* important that answers are as short as possible. Use professional language.
        
        Your tasks are - on the request by the user:
        - Provide a summary of the conversation in a structured format.
        - Send an email to the specified user using the "send_email" function.
        
        NOTES:
        - Every time you are about to send an email, make sure to confirm all the details with the user.""",
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