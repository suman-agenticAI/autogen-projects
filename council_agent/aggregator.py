"""
Aggregator — combines specialist email sections into one professional email.
Same logic as LangGraph version but using AutoGen AssistantAgent.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


AGGREGATOR_PROMPT = """You are composing a single professional email response to a citizen.

The citizen contacted the council about one or more issues. Below are the individual response sections from our specialist teams.
Combine them into ONE cohesive, professional email.

Rules:
- Start with "Dear <customer_name>,"
- Use a warm but professional tone
- Reference each issue clearly so the citizen can follow
- Include any SR reference numbers mentioned
- End with "Kind regards,\\nCity Council Customer Service"
- Do NOT add information that is not in the sections below
- If there is only one section, still format it as a complete email
- If a section is empty, skip it"""


async def aggregate_results(
    customer_name: str,
    agent_results: list[dict],
    model_client: AzureOpenAIChatCompletionClient,
) -> str:
    """Combine all specialist email sections into one professional email."""

    # Collect non-empty email sections
    sections = [r.get("email_section", "") for r in agent_results if r.get("email_section")]

    if not sections:
        return f"Dear {customer_name},\n\nThank you for contacting the City Council. Your request has been received and is being processed. A staff member will be in touch shortly.\n\nKind regards,\nCity Council Customer Service"

    sections_text = "\n---\n".join(sections)

    aggregator = AssistantAgent(
        name="aggregator",
        model_client=model_client,
        system_message=AGGREGATOR_PROMPT,
    )

    message = f"Customer name: {customer_name}\n\nResponse sections:\n{sections_text}"

    response = await aggregator.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=CancellationToken(),
    )

    email_body = response.chat_message.content
    if not isinstance(email_body, str):
        email_body = str(email_body)

    print(f"\n[AGGREGATOR] Combined {len(sections)} section(s) into final email")

    return email_body
