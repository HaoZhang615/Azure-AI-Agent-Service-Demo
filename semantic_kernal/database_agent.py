import os
import asyncio
from pathlib import Path, PurePath
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.identity import DefaultAzureCredential
from typing import Dict, List, Optional, Union, Annotated
import logging
import uuid
from datetime import datetime, timezone, timedelta

# Get the absolute path to the .env file in the semantic_kernal folder
current_dir = Path(__file__).parent
env_path = PurePath(current_dir).joinpath(".env")

print(f"Loading environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path)

# Add logging to verify environment variables are loaded
print(f"Loaded environment variables from: {env_path}")
print(f"COSMOSDB_ENDPOINT: {os.getenv('COSMOSDB_ENDPOINT')}")

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function

# CosmosDB Configuration
credential = DefaultAzureCredential()
cosmos_endpoint = os.getenv("COSMOSDB_ENDPOINT")
cosmos_client = CosmosClient(cosmos_endpoint, credential)
database_name = os.getenv("COSMOSDB_DATABASE")
database = cosmos_client.create_database_if_not_exists(id=database_name)
customer_container_name = "Customer"
purchase_container_name = "Purchases"
product_container_name = "Product"

class DatabasePlugin:
    """A plugin for interacting with the database."""
    
    def __init__(self, customer_id: str):
        self.customer_id = customer_id

    def validate_customer_exists(self, container) -> bool:
        """Validates if a customer exists in the database."""
        query = "SELECT VALUE COUNT(1) FROM c WHERE c.customer_id = @customer_id"
        parameters = [{"name": "@customer_id", "value": self.customer_id}]
        result = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return result[0] > 0 if result else False

    @kernel_function(description="Create a new purchase record in the Purchases container.")
    async def create_purchases_record(
        self, 
        purchase_record: Annotated[Dict, "The purchase record containing product_id and quantity."]
    ) -> Annotated[str, "Result of the purchase record creation."]:
        """Creates a new purchase record in the Purchases container."""
        parameters = {"purchase_record": purchase_record}
        
        # Check if product_id is missing in purchase_record
        if "product_id" not in purchase_record:
            # First, check if top-level product_id is provided
            if "product_id" in parameters:
                purchase_record["product_id"] = parameters["product_id"]
            # Otherwise, if a product_name is provided in purchase_record, derive product_id from it
            elif "product_name" in purchase_record:
                product_name = purchase_record["product_name"]
                product_container = database.get_container_client(product_container_name)
                query = "SELECT TOP 1 * FROM c WHERE CONTAINS(c.name, @name)"
                query_params = [{"name": "@name", "value": product_name}]
                results = list(product_container.query_items(query=query, parameters=query_params, enable_cross_partition_query=True))
                if results:
                    purchase_record["product_id"] = results[0]["product_id"]
                    # Optionally remove product_name to avoid redundancy
                    del purchase_record["product_name"]
                else:
                    return f"Product with name '{product_name}' not found. Please check the product name."

        container = database.get_container_client(purchase_container_name)
        product_container = database.get_container_client(product_container_name)
        
        # Validate customer exists
        customer_container = database.get_container_client(customer_container_name)
        if not self.validate_customer_exists(customer_container):
            return f"Customer with ID {self.customer_id} not found"
        
        # Get product details
        if "product_id" in purchase_record:
            product_query = "SELECT * FROM c WHERE c.product_id = @product_id"
            product_params = [{"name": "@product_id", "value": purchase_record["product_id"]}]
            product_results = list(product_container.query_items(
                query=product_query,
                parameters=product_params,
                enable_cross_partition_query=True
            ))
            if product_results:
                # limit to first result and extract product details
                product_details = {
                    "name": product_results[0]["name"],
                    "category": product_results[0]["category"],
                    "type": product_results[0]["type"],
                    "brand": product_results[0]["brand"],
                    "company": product_results[0]["company"],
                    "unit_price": product_results[0]["unit_price"],
                    "weight": product_results[0]["weight"],
                    "color": product_results[0].get("color", ""),
                    "material": product_results[0].get("material", "")
                }
            else:
                return f"Product with ID {purchase_record['product_id']} not found"
        else:
            return "Missing required field: product_id"

        # Create final purchase record with required schema
        final_record = {
            "customer_id": self.customer_id,
            "product_id": purchase_record["product_id"],
            "quantity": purchase_record.get("quantity", 1),  # Default to 1 if not specified
            "purchasing_date": datetime.now(timezone.utc).isoformat(),
            # Default to current date + 5 days
            "delivered_date": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
            "order_number": str(uuid.uuid4().hex),
            "product_details": product_details,
            "total_price": product_details.get("unit_price", 0) * purchase_record.get("quantity", 1),
            "id": str(uuid.uuid4())
        }
        
        try:
            container.create_item(body=final_record)
            return "Purchase record created successfully."
        except exceptions.CosmosHttpResponseError as e:
            logging.error(f"Failed to create purchase record: {e}")
            return f"Failed to create purchase record: {str(e)}"

    @kernel_function(description="Update customer information in the Customer container.")
    async def update_customer_record(
        self,
        first_name: Annotated[str, "Customer's first name"] = None,
        last_name: Annotated[str, "Customer's last name"] = None,
        email: Annotated[str, "Customer's email address"] = None,
        address: Annotated[Dict, "Customer's address"] = None,
        phone_number: Annotated[str, "Customer's phone number"] = None
    ) -> Annotated[Dict, "Result of the customer record update."]:
        """Updates an existing customer record in the Customer container."""
        container = database.get_container_client(customer_container_name)

        # Query to find the customer document using customer_id
        query = f"SELECT * FROM c WHERE c.customer_id = '{self.customer_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return {"status": "error", "message": "Customer record not found"}
        
        customer_doc = items[0]
        
        # Create update data dictionary with only provided values
        update_data = {}
        if first_name:
            update_data['first_name'] = first_name
        if last_name:
            update_data['last_name'] = last_name
        if email:
            update_data['email'] = email
        if address:
            update_data['address'] = address
        if phone_number:
            update_data['phone_number'] = phone_number
            
        # Update the document with allowed fields only
        customer_doc.update(update_data)
        
        # Replace the item without explicitly passing the partition key
        container.replace_item(
            item=customer_doc,
            body=customer_doc
        )
        
        return {"status": "success", "message": "Customer record updated successfully"}

    @kernel_function(description="Retrieve the current customer's information.")
    async def get_customer_record(self) -> Annotated[Union[Dict, str], "The customer record or error message."]:
        """Retrieves the customer record from the Customer container."""
        container = database.get_container_client(customer_container_name)
        try:
            query = """SELECT 
                c.customer_id,
                c.first_name,
                c.last_name,
                c.email,
                c.address,
                c.phone_number
            FROM c 
            WHERE c.customer_id = @customer_id"""
            
            parameters = [{"name": "@customer_id", "value": self.customer_id}]
            items = list(container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            if not items:
                return f"No customer found with ID: {self.customer_id}"
            return items[0]
        except exceptions.CosmosHttpResponseError as e:
            logging.error(f"Failed to get customer record: {e}")
            return f"Failed to get customer record: {str(e)}"

    @kernel_function(description="Retrieve all products or a specific product from the catalog.")
    async def get_product_record(
        self,
        product_id: Annotated[str, "Optional: The specific product ID to look up"] = None
    ) -> Annotated[Union[List[Dict], Dict, str], "Product record(s) or error message."]:
        """Retrieves product records from the Product container."""
        container = database.get_container_client(product_container_name)
        try:
            if product_id:
                query = """SELECT 
                    c.product_id,
                    c.name,
                    c.category,
                    c.type,
                    c.brand,
                    c.company,
                    c.unit_price,
                    c.weight
                FROM c 
                WHERE c.product_id = @product_id"""
                query_parameters = [{"name": "@product_id", "value": product_id}]
                items = list(container.query_items(query=query, parameters=query_parameters, enable_cross_partition_query=True))
                return items[0] if items else f"No product found with ID: {product_id}"
            else:
                items = list(container.read_all_items())
                return items if items else "No products found."
        except exceptions.CosmosHttpResponseError as e:
            logging.error(f"Failed to get product record(s): {e}")
            return f"Failed to get product record(s): {str(e)}"

    @kernel_function(description="Retrieve all purchases for the current customer.")
    async def get_purchases_record(self) -> Annotated[Union[List[Dict], str], "Purchase records or error message."]:
        """Retrieves all purchase records for the current customer with product details."""
        purchases_container = database.get_container_client(purchase_container_name)
        product_container = database.get_container_client(product_container_name)
        
        # Validate customer exists
        customer_container = database.get_container_client(customer_container_name)
        if not self.validate_customer_exists(customer_container):
            return f"Customer with ID {self.customer_id} not found"
        
        try:
            # First get all purchases for the customer
            query = """SELECT 
                c.customer_id,
                c.product_id,
                c.quantity,
                c.purchasing_date,
                c.delivered_date,
                c.order_number,
                c.total_price
            FROM c 
            WHERE c.customer_id = @customer_id"""
            
            parameters = [{"name": "@customer_id", "value": self.customer_id}]
            purchases = list(purchases_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if not purchases:
                return f"No purchases found for customer: {self.customer_id}"

            # Enhance purchase records with product details
            enhanced_purchases = []
            for purchase in purchases:
                # Get product details
                product_query = """SELECT 
                    c.name,
                    c.category,
                    c.type,
                    c.brand,
                    c.company,
                    c.unit_price,
                    c.weight
                FROM c 
                WHERE c.product_id = @product_id"""
                product_params = [{"name": "@product_id", "value": purchase["product_id"]}]
                
                product = list(product_container.query_items(
                    query=product_query,
                    parameters=product_params,
                    enable_cross_partition_query=True
                ))

                if product:
                    # Create clean purchase record without technical fields
                    clean_purchase = {
                        "quantity": purchase["quantity"],
                        "purchase_date": purchase["purchasing_date"],
                        "delivery_date": purchase["delivered_date"],
                        "total_price": purchase["total_price"],
                        "product": {
                            "name": product[0]["name"],
                            "category": product[0]["category"],
                            "type": product[0]["type"],
                            "brand": product[0]["brand"],
                            "company": product[0]["company"],
                            "price": product[0]["unit_price"],
                            "weight": product[0]["weight"]
                        }
                    }
                    enhanced_purchases.append(clean_purchase)
                else:
                    # Include purchase with minimal technical details if product not found
                    clean_purchase = {
                        "quantity": purchase["quantity"],
                        "purchase_date": purchase["purchasing_date"],
                        "delivery_date": purchase["delivered_date"],
                        "total_price": purchase["total_price"],
                        "product": {"error": "Product details not found"}
                    }
                    enhanced_purchases.append(clean_purchase)

            return enhanced_purchases

        except exceptions.CosmosHttpResponseError as e:
            logging.error(f"Failed to get purchase records: {e}")
            return f"Failed to get purchase records: {str(e)}"

# Sample user queries for demonstration
USER_INPUTS = [
    "Show me my profile information, my customer ID is b63d3f5bbaa7395caf90cdfddb2bc54f"
]

async def main():
    customer_id = "customer123"  # Example customer ID - in production, this would be dynamically set
    
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "database_agent"
    kernel = Kernel()
    kernel.add_plugin(DatabasePlugin(customer_id), plugin_name="db")
    
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
        name="Database_Agent",
        instructions="""
You are a database assistant that manages records in CosmosDB with specific operations for each container type. Use the provided functions to perform database operations.
Interaction goes over voice, so it's *super* important that answers are as short as possible. Use professional language.

Available operations:
- create_purchases_record: creates a new purchase record in the Purchases container.
- update_customer_record: updates the existing customer record in the Customer container.
- get_customer_record: retrieves the customer's information from the Customer container.
- get_product_record: retrieves all available products or a specific product from the Product container.
- get_purchases_record: retrieves all purchases for the current customer from the Purchases container.

NOTES:
- Before updating or creating any records, make sure to confirm the details with the user.
- All operations automatically use the current customer's ID.
- Purchases are always associated with both the current customer and a product_id.
- Before creating or updating a record, use the 'get' functions to retrieve the required schema of the respective container.

IMPORTANT: Never invent new tool or function names. Always use only the provided functions when interacting with the database.
        """,
        arguments=KernelArguments(settings=settings),
    )

    # 4. Create a chat history to hold the conversation
    chat_history = ChatHistory()

    # Interactive conversation loop
    print(f"Starting conversation with Database Agent (Customer ID: {customer_id})")
    
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
