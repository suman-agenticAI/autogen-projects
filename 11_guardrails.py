"""
AutoGen - Lesson 11: Guardrails & Safety
Prevent agents from doing harmful, off-topic, or unauthorized things.

Guardrail Types:
1. Input Guardrail  - Check user message BEFORE agent processes it
2. Output Guardrail - Check agent response BEFORE returning to user
3. Tool Guardrail   - Restrict what tools can do

Flow:
  User --> [Input Guard] --> Agent --> [Output Guard] --> Response
           "Block bad input"          "Filter bad output"
"""

import asyncio
import os
import re
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
# PART 1: Input Guardrail
# Check user input BEFORE sending to agent
# =====================================================

# Blocked topics for a CRM assistant
BLOCKED_TOPICS = ["hack", "exploit", "password", "credential", "delete all", "drop table"]

# Sensitive data patterns
SENSITIVE_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',   # SSN: 123-45-6789
    r'\b\d{16}\b',               # Credit card: 1234567890123456
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
]


def input_guardrail(user_message: str) -> dict:
    """
    Check user input for blocked topics and sensitive data.
    Returns: {"allowed": True/False, "reason": "..."}
    """
    # Check blocked topics
    message_lower = user_message.lower()
    for topic in BLOCKED_TOPICS:
        if topic in message_lower:
            return {"allowed": False, "reason": f"Blocked topic detected: '{topic}'"}

    # Check for sensitive data
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, user_message):
            return {"allowed": False, "reason": "Sensitive data detected (SSN, credit card, or email). Please remove before sending."}

    return {"allowed": True, "reason": "Input is safe"}


async def demo_input_guardrail():
    print("=" * 60)
    print("PART 1: Input Guardrail")
    print("=" * 60)

    agent = AssistantAgent(
        name="crm_assistant",
        model_client=model_client,
        system_message="You are a CRM assistant. Help with customer queries. Be concise.",
    )

    test_inputs = [
        "What is Acme Corp's account status?",               # Safe
        "How do I hack into the admin panel?",                # Blocked topic
        "Customer SSN is 123-45-6789, look them up",          # Sensitive data
        "Show me the sales report for Q1",                    # Safe
        "Can you delete all customer records?",               # Blocked topic
    ]

    for user_input in test_inputs:
        print(f"\n>> USER: {user_input}")

        # Run input guardrail BEFORE agent
        check = input_guardrail(user_input)

        if not check["allowed"]:
            print(f">> GUARDRAIL BLOCKED: {check['reason']}")
        else:
            response = await agent.on_messages(
                [TextMessage(content=user_input, source="user")],
                cancellation_token=CancellationToken(),
            )
            print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")
            await agent.on_reset(CancellationToken())


# =====================================================
# PART 2: Output Guardrail
# Check agent response BEFORE returning to user
# =====================================================

# Words that should never appear in agent responses
FORBIDDEN_OUTPUT = ["competitor", "CompetitorX", "switch provider", "cancel subscription"]

# Sensitive fields to redact
REDACT_PATTERNS = {
    r'\$[\d,]+': "[REDACTED_AMOUNT]",          # Dollar amounts
    r'\b\d{3}-\d{3}-\d{4}\b': "[REDACTED_PHONE]",  # Phone numbers
}


def output_guardrail(agent_response: str) -> dict:
    """
    Check and filter agent output before returning to user.
    Returns: {"allowed": True/False, "filtered_response": "...", "reason": "..."}
    """
    # Check forbidden words
    response_lower = agent_response.lower()
    for word in FORBIDDEN_OUTPUT:
        if word.lower() in response_lower:
            return {
                "allowed": False,
                "filtered_response": None,
                "reason": f"Response mentions forbidden topic: '{word}'",
            }

    # Redact sensitive data from output
    filtered = agent_response
    was_redacted = False
    for pattern, replacement in REDACT_PATTERNS.items():
        if re.search(pattern, filtered):
            filtered = re.sub(pattern, replacement, filtered)
            was_redacted = True

    return {
        "allowed": True,
        "filtered_response": filtered,
        "reason": "Redacted sensitive data" if was_redacted else "Output is safe",
    }


async def demo_output_guardrail():
    print("\n\n" + "=" * 60)
    print("PART 2: Output Guardrail")
    print("=" * 60)

    agent = AssistantAgent(
        name="crm_assistant",
        model_client=model_client,
        system_message="""You are a CRM assistant. When asked about customers, include their
        phone number and revenue details. Be concise (1-2 sentences).""",
    )

    test_queries = [
        "Tell me about customer John Smith. His phone is 555-123-4567 and his deal is worth $50,000",
        "Why should I use our product instead of CompetitorX?",
    ]

    for query in test_queries:
        print(f"\n>> USER: {query}")

        response = await agent.on_messages(
            [TextMessage(content=query, source="user")],
            cancellation_token=CancellationToken(),
        )
        raw_response = response.chat_message.content

        # Run output guardrail AFTER agent
        check = output_guardrail(raw_response)

        if not check["allowed"]:
            print(f">> GUARDRAIL BLOCKED OUTPUT: {check['reason']}")
            print(f">> (Agent tried to say: {raw_response.encode('ascii', 'replace').decode()[:80]}...)")
        else:
            print(f">> AGENT (filtered): {check['filtered_response'].encode('ascii', 'replace').decode()}")
            if check["reason"] != "Output is safe":
                print(f">> GUARDRAIL NOTE: {check['reason']}")

        await agent.on_reset(CancellationToken())


# =====================================================
# PART 3: Tool Guardrail
# Restrict what tools can do
# =====================================================

MAX_DISCOUNT = 20.0  # Business rule: max 20% discount


def apply_discount(customer_id: str, discount_percent: float) -> str:
    """
    Apply a discount to a customer's account.
    Args:
        customer_id: The customer ID
        discount_percent: Discount percentage (0-100)
    """
    # GUARDRAIL: Enforce business rules inside the tool
    if discount_percent > MAX_DISCOUNT:
        return f"DENIED: Cannot apply {discount_percent}% discount. Maximum allowed is {MAX_DISCOUNT}%. Please get manager approval for higher discounts."

    if discount_percent < 0:
        return "DENIED: Discount cannot be negative."

    return f"SUCCESS: {discount_percent}% discount applied to customer {customer_id}"


async def demo_tool_guardrail():
    print("\n\n" + "=" * 60)
    print("PART 3: Tool Guardrail (Business Rules)")
    print("=" * 60)

    agent = AssistantAgent(
        name="sales_agent",
        model_client=model_client,
        tools=[apply_discount],
        system_message="""You are a sales agent. When asked to apply discounts, use the apply_discount tool.
        Report the result back to the user. Be concise.""",
    )

    test_requests = [
        "Apply 10% discount to customer C001",   # Within limit
        "Apply 50% discount to customer C002",   # Exceeds limit - blocked!
        "Apply 15% discount to customer C003",   # Within limit
    ]

    for request in test_requests:
        print(f"\n>> USER: {request}")

        response = await agent.on_messages(
            [TextMessage(content=request, source="user")],
            cancellation_token=CancellationToken(),
        )
        print(f">> AGENT: {response.chat_message.content.encode('ascii', 'replace').decode()}")
        await agent.on_reset(CancellationToken())


async def main():
    await demo_input_guardrail()
    await demo_output_guardrail()
    await demo_tool_guardrail()


if __name__ == "__main__":
    asyncio.run(main())
