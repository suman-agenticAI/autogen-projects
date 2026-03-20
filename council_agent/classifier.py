"""
Classifier — Custom Agent that extracts intents from citizen email.
Uses structured output (Pydantic) — same as LangGraph version.

AutoGen conversion: Uses AssistantAgent with output_content_type
instead of LangChain's with_structured_output().
"""

import os
from typing import Literal
from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


# ---------------------------------------------------------------------------
# Pydantic models (SAME as LangGraph version)
# ---------------------------------------------------------------------------
class Intent(BaseModel):
    category: Literal[
        "Waste Management",
        "Roads and Streets",
        "Pets and Animals",
        "Building and Planning",
        "Rates and Property",
        "General",
    ]
    issue_description: str


class ClassificationResult(BaseModel):
    intents: list[Intent]


# ---------------------------------------------------------------------------
# Classifier prompt (SAME as LangGraph version)
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """You are a classification agent for a local council citizen request system.

A citizen has sent an email that may contain ONE or MULTIPLE issues.
Extract ALL distinct intents from the email.

For each intent, determine the category:
- "Waste Management" — bins, waste collection, recycling, green waste, missed bin, new bin, stolen bin
- "Roads and Streets" — potholes, road damage, streetlights, footpaths, signage, traffic
- "Pets and Animals" — pet registration, barking dogs, stray animals, animal complaints
- "Building and Planning" — development applications, building complaints, planning permits
- "Rates and Property" — rate payments, property valuations, pensioner rebates
- "General" — anything that does not clearly fit the above categories

Extract the relevant text for each intent as issue_description.
Do NOT combine multiple issues into one intent."""


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
FALLBACK_INTENT = {
    "category": "General",
    "issue_description": "Classification failed - routed to manual review",
}


# ---------------------------------------------------------------------------
# Classifier function
# ---------------------------------------------------------------------------
async def classify_email(email_text: str, model_client: AzureOpenAIChatCompletionClient) -> list[dict]:
    """Extract intents from citizen email using structured LLM output."""
    try:
        classifier = AssistantAgent(
            name="classifier",
            model_client=model_client,
            system_message=CLASSIFY_PROMPT,
            output_content_type=ClassificationResult,
        )

        response = await classifier.on_messages(
            [TextMessage(content=email_text, source="user")],
            cancellation_token=CancellationToken(),
        )

        result = response.chat_message.content
        intents = [intent.model_dump() for intent in result.intents]

        if not intents:
            print("[CLASSIFIER] No intents extracted - falling back to General")
            return [FALLBACK_INTENT]

        print(f"\n[CLASSIFIER] Found {len(intents)} intent(s):")
        for i, intent in enumerate(intents, 1):
            print(f"  {i}. {intent['category']}")

        return intents

    except Exception as e:
        print(f"[CLASSIFIER] Error: {str(e)} - falling back to General")
        return [FALLBACK_INTENT]
