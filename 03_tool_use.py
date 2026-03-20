"""
AutoGen - Lesson 3: Tool / Function Calling
An agent that uses custom tools to get work done.
Similar to LangChain's @tool decorator - but AutoGen style.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# Configure Azure OpenAI
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model="gpt-4o",
)


# ---- Define Tools (just regular Python functions) ----

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    Evaluate a math expression and return the result.
    Args:
        expression: A math expression like '2 + 3 * 4'
    """
    try:
        result = eval(expression)  # Safe for simple math in learning context
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def get_customer_info(customer_id: str) -> str:
    """
    Look up customer information by ID.
    Args:
        customer_id: The customer ID to look up (e.g., 'C001')
    """
    # Simulated CRM database
    customers = {
        "C001": {"name": "Acme Corp", "plan": "Enterprise", "revenue": "$250,000", "status": "Active"},
        "C002": {"name": "TechStart Inc", "plan": "Starter", "revenue": "$15,000", "status": "Active"},
        "C003": {"name": "Global Retail", "plan": "Professional", "revenue": "$85,000", "status": "Churned"},
    }
    customer = customers.get(customer_id)
    if customer:
        return f"Customer: {customer['name']}, Plan: {customer['plan']}, Revenue: {customer['revenue']}, Status: {customer['status']}"
    return f"Customer {customer_id} not found."


# ---- Create Agent with Tools ----

agent = AssistantAgent(
    name="crm_assistant",
    model_client=model_client,
    tools=[get_current_time, calculate, get_customer_info],
    system_message="""You are a CRM assistant that helps with customer queries.
    You have access to tools for:
    - Getting the current time
    - Doing calculations
    - Looking up customer information
    Use the appropriate tool when needed. Be concise in your answers.""",
)


async def main():
    print("=" * 60)
    print("LESSON 3: Agent with Tools")
    print("=" * 60)

    # Test 1: Tool that needs no arguments
    print("\n--- Question 1: What time is it? ---")
    response = await agent.on_messages(
        [TextMessage(content="What is the current date and time?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)

    # Reset agent for next question
    await agent.on_reset(CancellationToken())

    # Test 2: Tool with arguments (CRM lookup)
    print("\n--- Question 2: Customer lookup ---")
    response = await agent.on_messages(
        [TextMessage(content="Look up customer C001 and C003. Compare their status.", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)

    # Reset agent for next question
    await agent.on_reset(CancellationToken())

    # Test 3: Tool with calculation
    print("\n--- Question 3: Calculation ---")
    response = await agent.on_messages(
        [TextMessage(content="If customer C001 revenue is $250,000 and we give 15% discount, what is the new revenue?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)


if __name__ == "__main__":
    asyncio.run(main())
