"""
AutoGen - Lesson 12: Custom Agent
Build your own agent class with custom behavior.

Why custom agents?
- AssistantAgent always calls LLM - what if you don't need LLM for every step?
- You want to add business logic INSIDE the agent
- You need to connect to external systems (CRM, database, API)
- You want full control over how the agent behaves

Think of it as: AssistantAgent = pre-built car, CustomAgent = build your own car
"""

import asyncio
import os
from typing import Sequence
from dotenv import load_dotenv
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.messages import TextMessage, ChatMessage
from autogen_agentchat.base import Response
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Load environment variables
load_dotenv()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model="gpt-4o",
)


# =====================================================
# PART 1: Simple Custom Agent (No LLM needed)
# An agent that looks up data from a "CRM database"
# =====================================================

class CRMDatabaseAgent(BaseChatAgent):
    """Custom agent that queries a CRM database.
    No LLM needed - pure business logic."""

    def __init__(self):
        super().__init__(
            name="crm_database",
            description="Looks up customer data from the CRM database.",
        )
        # Simulated CRM database
        self._database = {
            "C001": {
                "name": "Acme Corp",
                "plan": "Enterprise",
                "revenue": 250000,
                "nps": 6,
                "tickets_open": 5,
                "usage_trend": "declining",
                "contract_end": "2026-05-15",
            },
            "C002": {
                "name": "TechStart Inc",
                "plan": "Starter",
                "revenue": 15000,
                "nps": 9,
                "tickets_open": 0,
                "usage_trend": "growing",
                "contract_end": "2026-12-01",
            },
            "C003": {
                "name": "Global Retail",
                "plan": "Professional",
                "revenue": 85000,
                "nps": 4,
                "tickets_open": 12,
                "usage_trend": "declining",
                "contract_end": "2026-04-01",
            },
        }

    @property
    def produced_message_types(self) -> list[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # Get the last message
        last_msg = messages[-1].content.lower()

        # Simple keyword matching (no LLM needed!)
        for cid, data in self._database.items():
            if cid.lower() in last_msg or data["name"].lower() in last_msg:
                result = (
                    f"Customer: {data['name']} ({cid})\n"
                    f"  Plan: {data['plan']}\n"
                    f"  Revenue: ${data['revenue']:,}\n"
                    f"  NPS: {data['nps']}/10\n"
                    f"  Open Tickets: {data['tickets_open']}\n"
                    f"  Usage Trend: {data['usage_trend']}\n"
                    f"  Contract Ends: {data['contract_end']}"
                )
                return Response(chat_message=TextMessage(content=result, source=self.name))

        # If asking for all customers
        if "all" in last_msg or "list" in last_msg:
            result = "All Customers:\n"
            for cid, data in self._database.items():
                result += f"  {cid}: {data['name']} - ${data['revenue']:,} - NPS {data['nps']}/10\n"
            return Response(chat_message=TextMessage(content=result, source=self.name))

        return Response(
            chat_message=TextMessage(content="Customer not found. Try: C001, C002, or C003", source=self.name)
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass  # No state to reset


async def demo_simple_custom():
    print("=" * 60)
    print("PART 1: Simple Custom Agent (No LLM)")
    print("=" * 60)

    crm = CRMDatabaseAgent()

    # Direct query - no LLM call, instant response
    queries = ["Look up C001", "Tell me about Global Retail", "Show all customers"]

    for query in queries:
        print(f"\n>> USER: {query}")
        response = await crm.on_messages(
            [TextMessage(content=query, source="user")],
            cancellation_token=CancellationToken(),
        )
        print(f">> CRM DATABASE:\n{response.chat_message.content}")


# =====================================================
# PART 2: Custom Agent with Business Logic
# An agent that scores customer health automatically
# =====================================================

class CustomerHealthScorer(BaseChatAgent):
    """Custom agent that calculates customer health score.
    Uses rules-based logic, not LLM."""

    def __init__(self):
        super().__init__(
            name="health_scorer",
            description="Calculates customer health score based on metrics.",
        )

    @property
    def produced_message_types(self) -> list[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        last_msg = messages[-1].content

        # Extract numbers from the message (simple parsing)
        score = 100  # Start with perfect score
        reasons = []

        # NPS scoring
        if "nps" in last_msg.lower():
            for word in last_msg.split():
                try:
                    nps = int(word.split("/")[0])
                    if nps <= 5:
                        score -= 30
                        reasons.append(f"Low NPS ({nps}/10): -30 points")
                    elif nps <= 7:
                        score -= 15
                        reasons.append(f"Moderate NPS ({nps}/10): -15 points")
                    break
                except (ValueError, IndexError):
                    continue

        # Ticket scoring
        if "ticket" in last_msg.lower():
            for word in last_msg.split():
                try:
                    tickets = int(word)
                    if tickets > 10:
                        score -= 25
                        reasons.append(f"High tickets ({tickets}): -25 points")
                    elif tickets > 5:
                        score -= 15
                        reasons.append(f"Moderate tickets ({tickets}): -15 points")
                    break
                except ValueError:
                    continue

        # Usage trend
        if "declining" in last_msg.lower():
            score -= 20
            reasons.append("Declining usage: -20 points")
        elif "growing" in last_msg.lower():
            score += 0
            reasons.append("Growing usage: no deduction")

        # Determine health level
        if score >= 80:
            health = "HEALTHY"
        elif score >= 50:
            health = "AT RISK"
        else:
            health = "CRITICAL"

        result = f"Health Score: {score}/100 ({health})\n"
        result += "Breakdown:\n" + "\n".join(f"  - {r}" for r in reasons)

        return Response(chat_message=TextMessage(content=result, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def demo_health_scorer():
    print("\n\n" + "=" * 60)
    print("PART 2: Custom Agent with Business Logic")
    print("=" * 60)

    scorer = CustomerHealthScorer()

    test_cases = [
        "Customer has NPS 4/10, 12 open tickets, declining usage",
        "Customer has NPS 9/10, 0 open tickets, growing usage",
        "Customer has NPS 6/10, 5 open tickets, declining usage",
    ]

    for case in test_cases:
        print(f"\n>> INPUT: {case}")
        response = await scorer.on_messages(
            [TextMessage(content=case, source="user")],
            cancellation_token=CancellationToken(),
        )
        print(f">> {response.chat_message.content}")


# =====================================================
# PART 3: Custom Agent + LLM Agent Working Together
# CRM lookup (no LLM) + AI analyst (LLM) in a team
# =====================================================

async def demo_hybrid_team():
    print("\n\n" + "=" * 60)
    print("PART 3: Custom + LLM Agents in a Team")
    print("=" * 60)

    # Custom agent: fast, no LLM, no cost
    crm = CRMDatabaseAgent()

    # LLM agent: smart analysis
    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="""You are a customer analyst.
        When you receive customer data from the CRM database,
        provide a brief risk assessment and one key recommendation.
        Keep response under 80 words.""",
    )

    team = RoundRobinGroupChat(
        participants=[crm, analyst],
        termination_condition=MaxMessageTermination(4),
    )

    result = await team.run(task="Look up customer C003 and assess their risk")

    for message in result.messages:
        print(f"\n{'-' * 50}")
        print(f">> {message.source.upper()}")
        print(f"{'-' * 50}")
        content = message.content
        if isinstance(content, str):
            print(content.encode("ascii", "replace").decode())


async def main():
    await demo_simple_custom()
    await demo_health_scorer()
    await demo_hybrid_team()


if __name__ == "__main__":
    asyncio.run(main())
