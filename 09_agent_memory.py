"""
AutoGen - Lesson 9: Agent with Memory
Agents that remember information across conversations.

Types of Memory:
1. Short-term: Within a single conversation (built-in, automatic)
2. Long-term: Across conversations (needs explicit setup)

This lesson covers both.
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


# =====================================================
# PART 1: Short-Term Memory (Within a conversation)
# =====================================================

async def demo_short_term_memory():
    print("=" * 60)
    print("PART 1: Short-Term Memory (within conversation)")
    print("=" * 60)

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful CRM assistant. Be concise.",
    )

    # Turn 1: Tell the agent something
    print("\n>> USER: My name is Suman and I work on Oracle CX")
    response = await agent.on_messages(
        [TextMessage(content="My name is Suman and I work on Oracle CX", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")

    # Turn 2: Agent should remember (NO reset between turns)
    print("\n>> USER: What is my name and what do I work on?")
    response = await agent.on_messages(
        [TextMessage(content="What is my name and what do I work on?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")

    # Turn 3: Reset and ask again - agent FORGETS
    await agent.on_reset(CancellationToken())
    print("\n>> [RESET AGENT MEMORY]")

    print("\n>> USER: What is my name?")
    response = await agent.on_messages(
        [TextMessage(content="What is my name?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")


# =====================================================
# PART 2: Long-Term Memory (Across conversations)
# Using a simple file-based approach
# =====================================================

import json

MEMORY_FILE = "agent_memory.json"


def load_memory() -> dict:
    """Load memory from file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"facts": [], "preferences": []}


def save_memory(memory: dict):
    """Save memory to file."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def add_to_memory(category: str, fact: str) -> str:
    """
    Save a fact to long-term memory.
    Args:
        category: Category of the fact ('facts' or 'preferences')
        fact: The information to remember
    """
    memory = load_memory()
    if fact not in memory.get(category, []):
        memory.setdefault(category, []).append(fact)
        save_memory(memory)
        return f"Saved to memory: {fact}"
    return f"Already in memory: {fact}"


def recall_memory() -> str:
    """Retrieve all stored memories."""
    memory = load_memory()
    if not memory["facts"] and not memory["preferences"]:
        return "No memories stored yet."
    result = "Stored memories:\n"
    if memory["facts"]:
        result += "\nFacts:\n" + "\n".join(f"  - {f}" for f in memory["facts"])
    if memory["preferences"]:
        result += "\nPreferences:\n" + "\n".join(f"  - {p}" for p in memory["preferences"])
    return result


async def demo_long_term_memory():
    print("\n\n" + "=" * 60)
    print("PART 2: Long-Term Memory (across conversations)")
    print("=" * 60)

    # Clean up any previous memory file
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    agent = AssistantAgent(
        name="memory_agent",
        model_client=model_client,
        tools=[add_to_memory, recall_memory],
        system_message="""You are a CRM assistant with long-term memory.
        When the user tells you something important about themselves or a customer,
        use add_to_memory to save it.
        When asked to recall, use recall_memory to retrieve stored information.
        Be concise.""",
    )

    # Conversation 1: Save some facts
    print("\n--- Conversation 1: Teaching the agent ---")
    print("\n>> USER: Remember that Acme Corp is our biggest customer worth $500K")
    response = await agent.on_messages(
        [TextMessage(
            content="Remember that Acme Corp is our biggest customer worth $500K",
            source="user",
        )],
        cancellation_token=CancellationToken(),
    )
    print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")

    print("\n>> USER: Also remember that my preference is to always offer 10% max discount")
    response = await agent.on_messages(
        [TextMessage(
            content="Also remember that my preference is to always offer 10% max discount",
            source="user",
        )],
        cancellation_token=CancellationToken(),
    )
    print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")

    # Simulate "new conversation" by creating a NEW agent instance
    print("\n--- Conversation 2: New session (agent recreated) ---")
    agent2 = AssistantAgent(
        name="memory_agent",
        model_client=model_client,
        tools=[add_to_memory, recall_memory],
        system_message="""You are a CRM assistant with long-term memory.
        When the user asks about past information, use recall_memory to check.
        Be concise.""",
    )

    print("\n>> USER: What do you remember about me and my customers?")
    response = await agent2.on_messages(
        [TextMessage(
            content="What do you remember about me and my customers?",
            source="user",
        )],
        cancellation_token=CancellationToken(),
    )
    print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")

    # Show the actual memory file
    print(f"\n--- Memory file contents ({MEMORY_FILE}) ---")
    print(json.dumps(load_memory(), indent=2))

    # Clean up
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)


async def main():
    await demo_short_term_memory()
    await demo_long_term_memory()


if __name__ == "__main__":
    asyncio.run(main())
