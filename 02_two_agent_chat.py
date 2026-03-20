"""
AutoGen - Lesson 2: Two-Agent Conversation
Two agents collaborate - one writes code, the other reviews it.
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
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

# Agent 1: Coder - writes Python code
coder = AssistantAgent(
    name="coder",
    model_client=model_client,
    system_message="""You are a Python developer.
    When given a task, write clean Python code to solve it.
    Keep code simple and well-commented.""",
)

# Agent 2: Reviewer - reviews the code
reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="""You are a senior code reviewer.
    Review the code provided by the coder.
    Check for: bugs, readability, best practices.
    Give specific feedback.
    If the code is good, say 'APPROVE' at the end.""",
)

# Termination condition: stop when reviewer says "APPROVE"
termination = TextMentionTermination("APPROVE")

# Create a team with round-robin turns: coder → reviewer → coder → reviewer...
team = RoundRobinGroupChat(
    participants=[coder, reviewer],
    termination_condition=termination,
    max_turns=6,  # Safety limit to avoid infinite loops
)


async def main():
    print("=" * 60)
    print("TWO-AGENT CHAT: Coder + Reviewer")
    print("=" * 60)

    # Give the team a task
    result = await team.run(
        task="Write a Python function that checks if a string is a palindrome. Include examples."
    )

    # Print the full conversation
    for message in result.messages:
        print(f"\n{'-' * 40}")
        print(f">> {message.source.upper()}")
        print(f"{'-' * 40}")
        print(message.content.encode("ascii", "replace").decode())


if __name__ == "__main__":
    asyncio.run(main())
