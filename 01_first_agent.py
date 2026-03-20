"""
AutoGen - Lesson 1: Your First Agent
A simple assistant agent that answers a question.
"""

import asyncio
import os
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

# Create an assistant agent
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful AI assistant. Keep your answers short and clear.",
)


async def main():
    # Ask the agent a question
    response = await agent.on_messages(
        [TextMessage(content="What is AutoGen in 2-3 sentences?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)


if __name__ == "__main__":
    asyncio.run(main())
