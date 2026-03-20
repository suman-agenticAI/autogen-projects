"""
Council Citizen Request System — AutoGen Version.

Converted from LangGraph to AutoGen, demonstrating:
- Structured Output (Classifier)
- Fan-Out / Fan-In (Parallel specialist agents)
- Tool Use (MCP tools for CRM operations)
- Custom Agent (General queue - no LLM)
- Aggregator (LLM combines results)

LangGraph Flow:
  START -> classify -> dispatch (fan-out) -> [waste|roads|pets|general] -> aggregator -> send_email -> END

AutoGen Flow:
  classify_email() -> asyncio.gather() (fan-out) -> [specialists in parallel] -> aggregate_results() (fan-in) -> email

Key Difference:
- LangGraph: StateGraph with nodes, edges, Send() for fan-out
- AutoGen: Python async functions with asyncio.gather() for fan-out
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from classifier import classify_email
from specialist_agents import run_specialist
from aggregator import aggregate_results

load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI client (replaces ChatAnthropic from LangGraph version)
# ---------------------------------------------------------------------------
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model="gpt-4o",
)


# ---------------------------------------------------------------------------
# Main orchestrator — replaces LangGraph StateGraph
# ---------------------------------------------------------------------------
async def process_citizen_request(
    email_text: str,
    customer_id: str,
    customer_name: str,
    parent_sr_number: str,
) -> dict:
    """
    Process a citizen email through the full pipeline.

    LangGraph used: StateGraph + nodes + edges + Send()
    AutoGen uses: async functions + asyncio.gather()
    """

    # ---- Step 1: Classify (same as classify_node in LangGraph) ----
    print("\n[STEP 1] Classifying email...")
    intents = await classify_email(email_text, model_client)

    # ---- Step 2: Fan-Out to specialists (replaces dispatch + Send()) ----
    print("\n[STEP 2] Dispatching to specialist agents (parallel)...")

    async def run_one_specialist(intent):
        task_data = {
            "parent_sr_id": parent_sr_number,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "category": intent["category"],
            "issue_description": intent["issue_description"],
        }
        return await run_specialist(intent["category"], task_data, model_client)

    # Fan-out: all specialists run in PARALLEL (same as LangGraph Send())
    agent_results = await asyncio.gather(
        *[run_one_specialist(intent) for intent in intents]
    )
    agent_results = list(agent_results)

    # ---- Step 3: Aggregate (same as aggregator_node in LangGraph) ----
    print("\n[STEP 3] Aggregating results into final email...")
    email_body = await aggregate_results(customer_name, agent_results, model_client)

    # ---- Step 4: Send email (same as send_email_node in LangGraph) ----
    has_actionable_content = any(
        r.get("status") in ("missing_info", "not_council_jurisdiction")
        for r in agent_results
    )

    if has_actionable_content:
        print(f"\n[STEP 4] Email would be sent via Parent SR: {parent_sr_number}")
        # In production: call send_sr_communication via MCP
    else:
        print(f"\n[STEP 4] No missing info - skipping email (parent SR already acknowledged)")

    return {
        "email_text": email_text,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "parent_sr_number": parent_sr_number,
        "intents": intents,
        "agent_results": agent_results,
        "customer_email_body": email_body,
        "final_status": "completed",
    }


# ---------------------------------------------------------------------------
# Test — same test case as LangGraph version
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    test_input = {
        "email_text": """Dear Council,

I have two issues to report.

First, my general waste bin at 42 Banksia Drive was not collected last Monday.
The truck drove past without stopping.

Second, there is a large pothole on the corner of Banksia Drive and Elm Street
that has been getting worse for weeks. It's now a safety hazard for cyclists.

Please address both issues urgently.

Regards,
John Smith""",
        "customer_id": "300001234567",
        "customer_name": "John Smith",
        "parent_sr_number": "SR-2024-100",
    }

    print("=" * 60)
    print("COUNCIL AGENT (AutoGen Version) - Multi-Intent Test")
    print("=" * 60)

    result = asyncio.run(process_citizen_request(**test_input))

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"\nStatus: {result['final_status']}")
    print(f"Intents found: {len(result['intents'])}")
    print(f"Agent results: {len(result['agent_results'])}")

    for i, r in enumerate(result["agent_results"], 1):
        print(f"\n--- Agent Result {i} ---")
        print(f"  Category: {r.get('category')}")
        print(f"  Sub-category: {r.get('sub_category')}")
        print(f"  Status: {r.get('status')}")
        print(f"  SR Number: {r.get('child_sr_number')}")

    print(f"\n--- Customer Email ---")
    email = result["customer_email_body"]
    if isinstance(email, str):
        print(email.encode("ascii", "replace").decode())
    else:
        print(str(email))
