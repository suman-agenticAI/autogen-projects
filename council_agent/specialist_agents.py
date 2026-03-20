"""
Specialist Agents — Waste, Roads, Pets, General.

AutoGen conversion notes:
- LangGraph: Each agent was a LangChain create_agent() with MCP tools
- AutoGen: Each agent is an AssistantAgent with MCP tools via StdioMcpToolAdapter
- The MCP server, prompts, and logic remain the SAME

For this conversion, we simulate the MCP tools as regular Python functions
since AutoGen's MCP adapter would need the same server setup.
In production, you'd use autogen_ext.tools.mcp.StdioMcpToolAdapter.
"""

import json
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {"intent", "category", "sub_category", "status", "child_sr_number", "email_section"}

AGENT_TIMEOUT = 60


def parse_agent_json(raw_response: str, fallback: dict) -> dict:
    """Extract and validate JSON from agent response text."""
    for i, ch in enumerate(raw_response):
        if ch != "{":
            continue
        try:
            parsed = json.loads(raw_response[i:])
        except json.JSONDecodeError:
            end = raw_response.rfind("}", i)
            if end == -1:
                continue
            try:
                parsed = json.loads(raw_response[i:end + 1])
            except json.JSONDecodeError:
                continue

        if not isinstance(parsed, dict):
            continue

        missing = REQUIRED_FIELDS - set(parsed.keys())
        if not missing:
            return parsed

    return fallback


# ---------------------------------------------------------------------------
# Simulated MCP tools (same as your mcp_server_cl.py)
# In production, use StdioMcpToolAdapter to connect to real MCP server
# ---------------------------------------------------------------------------
MOCK_PROPERTIES = {
    "42 banksia drive": {
        "id": "PROP-001",
        "address": "42 Banksia Drive",
        "zone": "A",
        "bins": ["general_waste", "recycling", "green_waste"],
    },
    "15 wattle street": {
        "id": "PROP-002",
        "address": "15 Wattle Street",
        "zone": "B",
        "bins": ["general_waste", "recycling"],
    },
    "8 eucalyptus court": {
        "id": "PROP-003",
        "address": "8 Eucalyptus Court",
        "zone": "C",
        "bins": [],
    },
}

SR_COUNTER = {"count": 100}


def search_property(query: str) -> str:
    """
    Search for a property by address.
    Args:
        query: The address to search for
    """
    query_lower = query.lower()
    for key, prop in MOCK_PROPERTIES.items():
        if key in query_lower:
            return json.dumps(prop)
    return json.dumps({"error": "Property not found"})


def get_property_bins(prop_id: str) -> str:
    """
    Get bins registered at a property.
    Args:
        prop_id: The property ID (e.g., PROP-001)
    """
    for prop in MOCK_PROPERTIES.values():
        if prop["id"] == prop_id:
            return json.dumps({"property_id": prop_id, "bins": prop["bins"]})
    return json.dumps({"error": "Property not found"})


def create_sr(title: str, problem_description: str, category_name: str,
              sub_category_name: str, contact_party_id: str,
              parent_sr_id: str) -> str:
    """
    Create a new Service Request in the CRM system.
    Args:
        title: SR title
        problem_description: Description of the issue
        category_name: Category (e.g., Waste Management)
        sub_category_name: Sub-category (e.g., Missed Bin)
        contact_party_id: Customer ID
        parent_sr_id: Parent SR number
    """
    SR_COUNTER["count"] += 1
    sr_number = f"SR-2024-{SR_COUNTER['count']}"
    return json.dumps({
        "SrNumber": sr_number,
        "Title": title,
        "CategoryName": category_name,
        "SubCategoryName": sub_category_name,
        "Status": "New",
    })


def search_kb(query: str, category: str = "") -> str:
    """
    Search the council knowledge base for articles.
    Args:
        query: Search term (e.g., pothole, streetlight)
        category: Optional category filter
    """
    kb_articles = {
        "pothole": {
            "title": "Pothole / Road Damage Report",
            "category": "Roads and Streets",
            "sub_category": "Road Damage",
            "required_fields": ["location", "description of damage"],
            "jurisdiction": "Council manages local roads only. State highways managed by State Roads Authority.",
            "sla": "Inspection within 5 business days",
            "form_url": "https://council.gov.au/forms/road-damage",
        },
        "streetlight": {
            "title": "Streetlight Fault Report",
            "category": "Roads and Streets",
            "sub_category": "Streetlight",
            "required_fields": ["location", "description of fault"],
            "jurisdiction": "Council manages streetlights in parks and car parks only. Public road streetlights managed by electricity provider.",
            "sla": "Assessment within 10 business days",
            "form_url": "https://council.gov.au/forms/streetlight",
        },
        "footpath": {
            "title": "Footpath Damage Report",
            "category": "Roads and Streets",
            "sub_category": "Footpath Damage",
            "required_fields": ["location", "description of damage", "hazard level"],
            "jurisdiction": "Council manages all public footpaths.",
            "sla": "Inspection within 5 business days",
            "form_url": "https://council.gov.au/forms/road-damage",
        },
        "barking": {
            "title": "Barking Dog Complaint",
            "category": "Pets and Animals",
            "sub_category": "Barking Dog",
            "required_fields": ["address of dog", "times of barking", "duration"],
            "jurisdiction": "Council handles all domestic animal complaints.",
            "sla": "Investigation within 10 business days",
            "form_url": "https://council.gov.au/forms/animal-complaint",
        },
        "stray": {
            "title": "Stray Animal Report",
            "category": "Pets and Animals",
            "sub_category": "Stray Animal",
            "required_fields": ["location", "animal description", "behaviour"],
            "jurisdiction": "Council handles stray domestic animals. Wildlife handled by state wildlife service.",
            "sla": "Response within 24 hours for aggressive animals",
            "form_url": "https://council.gov.au/forms/animal-complaint",
        },
        "registration": {
            "title": "Pet Registration",
            "category": "Pets and Animals",
            "sub_category": "Pet Registration",
            "required_fields": ["pet type", "breed", "owner address", "microchip number"],
            "jurisdiction": "All dogs and cats must be registered with council.",
            "sla": "Registration processed within 5 business days",
            "form_url": "https://council.gov.au/forms/pet-registration",
        },
    }

    query_lower = query.lower()
    results = []
    for key, article in kb_articles.items():
        if key in query_lower or (category and article["category"] == category):
            results.append(article)

    if results:
        return json.dumps(results)
    return json.dumps({"results": [], "message": "No KB articles found"})


WASTE_SCHEDULE = {
    "Zone A": {"general_waste": "Monday", "recycling": "Wednesday", "green_waste": "Friday"},
    "Zone B": {"general_waste": "Tuesday", "recycling": "Thursday", "green_waste": "Saturday"},
    "Zone C": {"general_waste": "Wednesday", "recycling": "Friday", "green_waste": "Monday"},
    "Zone D": {"general_waste": "Thursday", "recycling": "Monday", "green_waste": "Wednesday"},
}


def get_waste_schedule() -> str:
    """Get the weekly waste collection schedule by zone."""
    return json.dumps(WASTE_SCHEDULE)


COUNCIL_FORMS = {
    "Waste Management": {
        "Missed Bin": "https://council.gov.au/forms/missed-bin",
        "New Bin": "https://council.gov.au/forms/new-bin",
        "Additional Bin": "https://council.gov.au/forms/additional-bin",
        "Stolen/Lost Bin": "https://council.gov.au/forms/stolen-lost-bin",
    },
    "Roads & Streets": {
        "Road Damage": "https://council.gov.au/forms/road-damage",
        "Streetlight": "https://council.gov.au/forms/streetlight",
    },
    "Pets & Animals": {
        "Registration": "https://council.gov.au/forms/pet-registration",
        "Complaint": "https://council.gov.au/forms/animal-complaint",
    },
}


def get_council_forms() -> str:
    """Get online form URLs for citizen self-service."""
    return json.dumps(COUNCIL_FORMS)


# ---------------------------------------------------------------------------
# Specialist Agent Prompts
# ---------------------------------------------------------------------------
WASTE_PROMPT = """You are a Waste Management specialist agent for the City Council.

You receive a task with: parent_sr_id, customer_id, customer_name, category, issue_description.

You have access to tools: search_property, get_property_bins, create_sr, get_waste_schedule, get_council_forms.

Follow these steps:
1. Extract sub-category (Missed Bin, New Bin, Additional Bin, Stolen-Lost Bin, Cancel Bin)
2. Search for the property using search_property
3. Get bins at the property using get_property_bins
4. Check waste schedule using get_waste_schedule
5. If all info present and valid: create_sr and set status="child_sr_created"
6. If info missing or invalid: set status="missing_info", include form URL

Return ONLY this JSON:
{{"intent": "waste_management", "category": "Waste Management", "sub_category": "<>", "status": "<>", "child_sr_number": "<>", "email_section": "<email paragraph for citizen>", "form_url": "<>"}}"""


ROADS_PROMPT = """You are a Roads and Streets specialist agent for the City Council.

You receive a task with: parent_sr_id, customer_id, customer_name, category, issue_description.

You have access to tools: search_kb, create_sr.

Follow these steps:
1. Search the knowledge base using search_kb with keywords from the issue
2. Extract required fields based on KB rules
3. Check jurisdiction (council vs state authority)
4. If valid and within jurisdiction: create_sr, status="child_sr_created"
5. If missing info: status="missing_info"
6. If not council jurisdiction: status="not_council_jurisdiction", tell citizen who to contact

Return ONLY this JSON:
{{"intent": "roads_and_streets", "category": "Roads and Streets", "sub_category": "<from KB>", "status": "<>", "child_sr_number": "<>", "email_section": "<email paragraph for citizen>", "form_url": "<>"}}"""


PETS_PROMPT = """You are a Pets and Animals specialist agent for the City Council.

You receive a task with: parent_sr_id, customer_id, customer_name, category, issue_description.

You have access to tools: search_kb, create_sr.

Follow these steps:
1. Search the knowledge base using search_kb with keywords from the issue
2. Extract required fields based on KB rules
3. Check jurisdiction
4. If valid: create_sr, status="child_sr_created"
5. If missing info: status="missing_info", include form URL
6. For barking dog complaints, reassure citizen their details are confidential

Return ONLY this JSON:
{{"intent": "pets_and_animals", "category": "Pets and Animals", "sub_category": "<from KB>", "status": "<>", "child_sr_number": "<>", "email_section": "<email paragraph for citizen>", "form_url": "<>"}}"""


# ---------------------------------------------------------------------------
# Fallback responses
# ---------------------------------------------------------------------------
FALLBACKS = {
    "Waste Management": {
        "intent": "waste_management", "category": "Waste Management",
        "sub_category": "General", "status": "agent_error", "child_sr_number": "",
        "email_section": "We received your waste management request and it has been forwarded for manual review.",
        "form_url": "",
    },
    "Roads and Streets": {
        "intent": "roads_and_streets", "category": "Roads and Streets",
        "sub_category": "General", "status": "agent_error", "child_sr_number": "",
        "email_section": "We received your roads request and it has been forwarded for manual review.",
        "form_url": "",
    },
    "Pets and Animals": {
        "intent": "pets_and_animals", "category": "Pets and Animals",
        "sub_category": "General", "status": "agent_error", "child_sr_number": "",
        "email_section": "We received your pets request and it has been forwarded for manual review.",
        "form_url": "",
    },
}


# ---------------------------------------------------------------------------
# Run specialist agent
# ---------------------------------------------------------------------------
async def run_specialist(category: str, task_data: dict, model_client: AzureOpenAIChatCompletionClient) -> dict:
    """Run the appropriate specialist agent for the given category."""

    # Select prompt and tools based on category
    if category == "Waste Management":
        prompt = WASTE_PROMPT
        tools = [search_property, get_property_bins, create_sr, get_waste_schedule, get_council_forms]
        fallback = FALLBACKS["Waste Management"]
        node_name = "WASTE"
    elif category == "Roads and Streets":
        prompt = ROADS_PROMPT
        tools = [search_kb, create_sr]
        fallback = FALLBACKS["Roads and Streets"]
        node_name = "ROADS"
    elif category == "Pets and Animals":
        prompt = PETS_PROMPT
        tools = [search_kb, create_sr]
        fallback = FALLBACKS["Pets and Animals"]
        node_name = "PETS"
    else:
        # General queue — no LLM needed (Custom Agent pattern from Lesson 12!)
        return await run_general_queue(task_data)

    print(f"\n[{node_name} AGENT] Processing: {task_data['issue_description'][:80]}...")

    try:
        agent = AssistantAgent(
            name=f"{node_name.lower()}_agent",
            model_client=model_client,
            tools=tools,
            system_message=prompt,
        )

        message = f"Process this task:\n{json.dumps(task_data, indent=2)}"

        response = await asyncio.wait_for(
            agent.on_messages(
                [TextMessage(content=message, source="user")],
                cancellation_token=CancellationToken(),
            ),
            timeout=AGENT_TIMEOUT,
        )

        raw_response = response.chat_message.content
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        result = parse_agent_json(raw_response, fallback)
        print(f"[{node_name} AGENT] Status: {result.get('status', 'unknown')}")
        return result

    except asyncio.TimeoutError:
        print(f"[{node_name} AGENT] Timeout after {AGENT_TIMEOUT}s")
        return fallback
    except Exception as e:
        print(f"[{node_name} AGENT] Error: {str(e)}")
        return fallback


async def run_general_queue(task_data: dict) -> dict:
    """General queue — no LLM, just create SR for manual review."""
    category = task_data["category"]
    print(f"\n[GENERAL QUEUE] Creating child SR for manual review: {category}")

    try:
        result = json.loads(create_sr(
            title=f"{category} - Manual Review",
            problem_description=task_data.get("issue_description", ""),
            category_name=category,
            sub_category_name="General",
            contact_party_id=task_data.get("customer_id", ""),
            parent_sr_id=task_data.get("parent_sr_id", ""),
        ))
        sr_number = result.get("SrNumber", "")
        print(f"[GENERAL QUEUE] Child SR created: {sr_number}")
    except Exception as e:
        print(f"[GENERAL QUEUE] Failed: {str(e)}")
        sr_number = ""

    return {
        "intent": "general",
        "category": category,
        "sub_category": "General",
        "status": "child_sr_created" if sr_number else "queued_for_manual_review",
        "child_sr_number": sr_number,
        "email_section": "",
        "form_url": "",
    }
