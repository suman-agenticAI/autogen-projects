"""
AutoGen - Lesson 10: Structured Output
Force agents to return data in a specific format (Pydantic model).

Why?
- Free text: "Revenue is around 500K and risk is high"
- Structured: {"revenue": 500000, "risk_level": "HIGH"}

Production systems need structured data, not free text.
"""

import asyncio
import os
from pydantic import BaseModel
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
# PART 1: Without Structured Output (the problem)
# =====================================================

async def demo_unstructured():
    print("=" * 60)
    print("PART 1: Without Structured Output (free text)")
    print("=" * 60)

    agent = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="You are a CRM analyst. Analyze the customer and provide risk assessment. Be concise.",
    )

    response = await agent.on_messages(
        [TextMessage(content="Analyze customer: Acme Corp, $250K revenue, usage dropped 30%, NPS 6/10", source="user")],
        cancellation_token=CancellationToken(),
    )

    print(f"\nAgent response (free text):")
    print(response.chat_message.content.encode("ascii", "replace").decode())
    print(f"\nType: {type(response.chat_message.content)}")
    print("Problem: How do you extract risk_level or revenue from this text?")


# =====================================================
# PART 2: With Structured Output (the solution)
# =====================================================

# Define the EXACT structure you want back
class CustomerAnalysis(BaseModel):
    """Structured output for customer analysis."""
    customer_name: str
    revenue: float
    risk_level: str  # HIGH, MEDIUM, LOW
    churn_probability: float  # 0.0 to 1.0
    top_concerns: list[str]
    recommended_action: str


async def demo_structured():
    print("\n\n" + "=" * 60)
    print("PART 2: With Structured Output (Pydantic model)")
    print("=" * 60)

    agent = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="You are a CRM analyst. Analyze the customer and provide risk assessment.",
        output_content_type=CustomerAnalysis,  # Forces structured output!
    )

    response = await agent.on_messages(
        [TextMessage(content="Analyze customer: Acme Corp, $250K revenue, usage dropped 30%, NPS 6/10", source="user")],
        cancellation_token=CancellationToken(),
    )

    # The response is already a structured Pydantic object
    result = response.chat_message.content

    print(f"\nAgent response (structured):")
    print(f"  Customer:     {result.customer_name}")
    print(f"  Revenue:      ${result.revenue:,.0f}")
    print(f"  Risk Level:   {result.risk_level}")
    print(f"  Churn Prob:   {result.churn_probability:.0%}")
    print(f"  Top Concerns: {result.top_concerns}")
    print(f"  Action:       {result.recommended_action}")
    print(f"\nType: {type(result)}")
    print("Benefit: You can use result.risk_level directly in your code!")


# =====================================================
# PART 3: Structured Output in a Pipeline
# =====================================================

class LeadScore(BaseModel):
    """Structured output for lead scoring."""
    company_name: str
    score: int  # 0-100
    grade: str  # A, B, C, D, F
    buying_signals: list[str]
    next_step: str


async def demo_pipeline():
    print("\n\n" + "=" * 60)
    print("PART 3: Using Structured Output in a Pipeline")
    print("=" * 60)

    scorer = AssistantAgent(
        name="lead_scorer",
        model_client=model_client,
        system_message="You are a lead scoring expert. Score leads from 0-100 and grade them A-F.",
        output_content_type=LeadScore,
    )

    # Score multiple leads
    leads = [
        "TechStart Inc: 10 employees, visited pricing page 5 times, downloaded whitepaper, requested demo",
        "BigBank Corp: 5000 employees, visited homepage once, no engagement",
        "FastGrow LLC: 200 employees, attended webinar, CEO follows us on LinkedIn, budget approved",
    ]

    results = []
    for lead in leads:
        response = await scorer.on_messages(
            [TextMessage(content=f"Score this lead: {lead}", source="user")],
            cancellation_token=CancellationToken(),
        )
        result = response.chat_message.content
        results.append(result)
        await scorer.on_reset(CancellationToken())

    # Now you can use structured data programmatically!
    print("\nLead Scoring Results:")
    print(f"{'Company':<20} {'Score':<8} {'Grade':<8} {'Next Step'}")
    print("-" * 70)
    for r in results:
        print(f"{r.company_name:<20} {r.score:<8} {r.grade:<8} {r.next_step}")

    # Filter high-priority leads (this is why structure matters!)
    hot_leads = [r for r in results if r.score >= 70]
    print(f"\nHot leads (score >= 70): {[r.company_name for r in hot_leads]}")


async def main():
    await demo_unstructured()
    await demo_structured()
    await demo_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
