"""
AutoGen - Lesson 8: Human-in-the-Loop
Agent proposes actions, but a HUMAN must approve before proceeding.

Use cases:
- Approving emails before sending
- Reviewing code before deploying
- Confirming orders before processing
- Any high-stakes decision that needs human oversight
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
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


# ---- Custom Tool that requires approval ----

def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to a customer.
    Args:
        to: Email recipient address
        subject: Email subject line
        body: Email body content
    """
    # In real world, this would call an email API
    return f"Email sent successfully to {to} with subject: {subject}"


def apply_discount(customer_id: str, discount_percent: float) -> str:
    """
    Apply a discount to a customer's account.
    Args:
        customer_id: The customer ID
        discount_percent: Discount percentage to apply
    """
    # In real world, this would update CRM
    return f"Discount of {discount_percent}% applied to customer {customer_id}"


# ---- Agent that proposes actions ----
proposal_agent = AssistantAgent(
    name="proposal_agent",
    model_client=model_client,
    tools=[send_email, apply_discount],
    system_message="""You are a CRM assistant that helps with customer actions.
    When asked to do something:
    1. First PROPOSE what you plan to do (explain the email content or discount details)
    2. Wait for the human reviewer to approve
    3. Only execute the tools AFTER getting approval
    Keep responses concise.""",
)

# ---- Human Reviewer (UserProxyAgent alternative) ----
# In AutoGen 0.7+, we use a simple assistant that pauses for input
human_reviewer = AssistantAgent(
    name="human_reviewer",
    model_client=model_client,
    system_message="""You are simulating a human reviewer.
    When the proposal_agent proposes an action:
    - Review it carefully
    - If it looks good, say: "APPROVED - proceed with the action"
    - If something needs changes, suggest modifications
    After the action is executed, say DONE.""",
)

# ---- Termination ----
termination = TextMentionTermination("DONE") | MaxMessageTermination(8)

# ---- Team ----
team = RoundRobinGroupChat(
    participants=[proposal_agent, human_reviewer],
    termination_condition=termination,
)


async def main():
    print("=" * 60)
    print("LESSON 8: Human-in-the-Loop")
    print("Agent proposes -> Human reviews -> Agent executes")
    print("=" * 60)

    task = """
    Customer Acme Corp (ID: C001) churned 3 months ago due to poor support.
    We've improved our SLA from 24hrs to 4hrs.

    Please:
    1. Send a win-back email to contact@acmecorp.com
    2. Offer them a 15% discount to come back
    """

    # Stream output to console in real-time
    result = await Console(team.run_stream(task=task))

    print("\n" + "=" * 60)
    print("CONVERSATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
