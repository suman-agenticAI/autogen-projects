"""
AutoGen - Lesson 5: Fan-Out / Fan-In Pattern
Multiple agents work in PARALLEL, then results are combined.

Pattern:
  Task --> [Agent1, Agent2, Agent3] (parallel) --> Aggregator --> Final Answer

Real-world CRM use case: Preparing a customer account review.
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

# The customer scenario (shared input for all agents)
CUSTOMER_SCENARIO = """
Customer: Acme Corp
Plan: Enterprise ($250K/year)
Contract Renewal: 60 days
Usage: Dropped 30% last quarter
Support Tickets: 15 open (5 critical)
NPS Score: 6/10 (was 8/10 last year)
Competitor: They've had 3 demos with CompetitorX
"""

# ---- Agent 1: Sales Analyst ----
sales_analyst = AssistantAgent(
    name="sales_analyst",
    model_client=model_client,
    system_message="""You are a Sales Analyst.
    Analyze the customer from a REVENUE perspective only.
    Cover: revenue risk, upsell opportunities, pricing strategy.
    Keep response under 100 words. Be specific with numbers.""",
)

# ---- Agent 2: Support Analyst ----
support_analyst = AssistantAgent(
    name="support_analyst",
    model_client=model_client,
    system_message="""You are a Support Analyst.
    Analyze the customer from a SUPPORT HEALTH perspective only.
    Cover: ticket severity, response times, unresolved issues.
    Keep response under 100 words. Be specific.""",
)

# ---- Agent 3: Competitive Intel Analyst ----
competitive_analyst = AssistantAgent(
    name="competitive_analyst",
    model_client=model_client,
    system_message="""You are a Competitive Intelligence Analyst.
    Analyze the customer from a COMPETITIVE THREAT perspective only.
    Cover: competitor strengths, our weaknesses, switching cost.
    Keep response under 100 words. Be specific.""",
)

# ---- Aggregator Agent (Fan-In) ----
aggregator = AssistantAgent(
    name="aggregator",
    model_client=model_client,
    system_message="""You are an Executive Summary Writer.
    You will receive analyses from 3 different analysts.
    Combine them into ONE clear executive summary with:
    1. Overall Risk Level (High/Medium/Low)
    2. Top 3 Action Items (prioritized)
    3. Expected Outcome
    Keep it under 150 words. Be direct and actionable.""",
)


async def run_agent(agent, task):
    """Helper to run a single agent and return its response."""
    response = await agent.on_messages(
        [TextMessage(content=task, source="user")],
        cancellation_token=CancellationToken(),
    )
    return response.chat_message.content


async def main():
    print("=" * 60)
    print("LESSON 5: Fan-Out / Fan-In Pattern")
    print("=" * 60)

    # ---- FAN-OUT: Run 3 agents in PARALLEL ----
    print("\n>> FAN-OUT: Running 3 analysts in parallel...\n")

    # asyncio.gather runs all 3 at the same time!
    sales_result, support_result, competitive_result = await asyncio.gather(
        run_agent(sales_analyst, CUSTOMER_SCENARIO),
        run_agent(support_analyst, CUSTOMER_SCENARIO),
        run_agent(competitive_analyst, CUSTOMER_SCENARIO),
    )

    # Print individual results
    print(f"{'-' * 50}")
    print(">> SALES ANALYST")
    print(f"{'-' * 50}")
    print(sales_result.encode("ascii", "replace").decode())

    print(f"\n{'-' * 50}")
    print(">> SUPPORT ANALYST")
    print(f"{'-' * 50}")
    print(support_result.encode("ascii", "replace").decode())

    print(f"\n{'-' * 50}")
    print(">> COMPETITIVE ANALYST")
    print(f"{'-' * 50}")
    print(competitive_result.encode("ascii", "replace").decode())

    # ---- FAN-IN: Combine all results ----
    print(f"\n{'=' * 50}")
    print(">> FAN-IN: Aggregating results...")
    print(f"{'=' * 50}")

    combined_input = f"""
    Here are the analyses from 3 specialists:

    --- SALES ANALYSIS ---
    {sales_result}

    --- SUPPORT ANALYSIS ---
    {support_result}

    --- COMPETITIVE ANALYSIS ---
    {competitive_result}

    Combine these into one executive summary.
    """

    final_summary = await run_agent(aggregator, combined_input)

    print(f"\n{'-' * 50}")
    print(">> EXECUTIVE SUMMARY")
    print(f"{'-' * 50}")
    print(final_summary.encode("ascii", "replace").decode())


if __name__ == "__main__":
    asyncio.run(main())
