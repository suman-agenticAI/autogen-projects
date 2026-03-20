"""
AutoGen - Lesson 6: Selector Group Chat
The LLM decides which agent should speak next based on conversation context.
No fixed order - the most relevant agent is selected dynamically.
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
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

# ---- Specialized Agents ----

sales_agent = AssistantAgent(
    name="sales_agent",
    model_client=model_client,
    description="Expert in sales strategy, pricing, deals, and revenue growth.",
    system_message="""You are a Sales Expert.
    Only respond when the topic is about sales, pricing, deals, or revenue.
    Keep responses under 100 words.""",
)

technical_agent = AssistantAgent(
    name="technical_agent",
    model_client=model_client,
    description="Expert in technical issues, integrations, APIs, and product features.",
    system_message="""You are a Technical Expert.
    Only respond when the topic is about technical issues, integrations, or product features.
    Keep responses under 100 words.""",
)

legal_agent = AssistantAgent(
    name="legal_agent",
    model_client=model_client,
    description="Expert in contracts, compliance, terms of service, and legal matters.",
    system_message="""You are a Legal Expert.
    Only respond when the topic is about contracts, compliance, or legal matters.
    Keep responses under 100 words.""",
)

coordinator = AssistantAgent(
    name="coordinator",
    model_client=model_client,
    description="Coordinates the discussion, summarizes findings, and creates action plans.",
    system_message="""You are a Coordinator.
    After hearing from specialists, summarize the discussion into a clear action plan.
    When the plan is complete, say DONE at the end.
    Keep responses under 150 words.""",
)

# ---- Termination ----
termination = TextMentionTermination("DONE") | MaxMessageTermination(10)

# ---- Selector Group Chat ----
# The LLM reads the conversation and picks the best agent to speak next
team = SelectorGroupChat(
    participants=[sales_agent, technical_agent, legal_agent, coordinator],
    model_client=model_client,  # LLM used to SELECT the next speaker
    termination_condition=termination,
)


async def main():
    print("=" * 60)
    print("LESSON 6: Selector Group Chat")
    print("LLM dynamically picks who speaks next")
    print("=" * 60)

    # A complex task that touches multiple domains
    task = """
    Customer: MegaRetail Inc wants to renew their contract but has these concerns:
    1. They want a 20% discount on their $500K Enterprise plan
    2. Their API integration keeps failing with timeout errors
    3. They need a GDPR compliance clause added to the contract

    Address all their concerns and create an action plan.
    """

    result = await team.run(task=task)

    # Print the conversation
    for message in result.messages:
        print(f"\n{'-' * 50}")
        print(f">> {message.source.upper()}")
        print(f"{'-' * 50}")
        content = message.content
        if isinstance(content, str):
            print(content.encode("ascii", "replace").decode())


if __name__ == "__main__":
    asyncio.run(main())
