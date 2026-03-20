"""
AutoGen - Lesson 4: Multi-Agent Group Chat
Three specialized agents collaborate on a CRM task.
Think of it as a team meeting between specialists.
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
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

# ---- Agent 1: Sales Strategist ----
sales_strategist = AssistantAgent(
    name="sales_strategist",
    model_client=model_client,
    system_message="""You are a Sales Strategist.
    Your job is to analyze customer situations and propose sales strategies.
    Focus on: upselling, cross-selling, retention strategies.
    Keep responses under 150 words.
    Wait for input from the Data Analyst and Customer Success Manager before finalizing.
    Only say FINALIZED after all team members have contributed.""",
)

# ---- Agent 2: Data Analyst ----
data_analyst = AssistantAgent(
    name="data_analyst",
    model_client=model_client,
    system_message="""You are a Data Analyst.
    Your job is to provide data-driven insights and metrics.
    Focus on: revenue impact, success rates, benchmarks, ROI calculations.
    Keep responses under 150 words.
    Support your points with numbers and percentages.""",
)

# ---- Agent 3: Customer Success Manager ----
csm = AssistantAgent(
    name="customer_success_manager",
    model_client=model_client,
    system_message="""You are a Customer Success Manager.
    Your job is to advocate for the customer's perspective.
    Focus on: customer satisfaction, risk of churn, relationship health.
    Keep responses under 150 words.
    Flag any risks to the customer relationship.""",
)

# ---- Termination Conditions ----
# Stop when someone says "FINALIZED" OR after 9 messages max
termination = TextMentionTermination("FINALIZED") | MaxMessageTermination(9)

# ---- Create the Team ----
team = RoundRobinGroupChat(
    participants=[sales_strategist, data_analyst, csm],
    termination_condition=termination,
)


async def main():
    print("=" * 60)
    print("LESSON 4: Multi-Agent Group Chat")
    print("Sales Strategist + Data Analyst + Customer Success Manager")
    print("=" * 60)

    # Give the team a real CRM scenario
    task = """
    Customer: Acme Corp (Enterprise plan, $250K annual revenue)
    Situation: Their contract renews in 60 days. Usage has dropped 30%
    in the last quarter. They've been evaluating a competitor.

    Question: What should we do to retain this customer and grow the account?
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
