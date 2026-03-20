# AutoGen Projects

Multi-agent AI projects built with Microsoft AutoGen framework.

## About

Learning and building agentic AI systems using AutoGen, focusing on enterprise use cases like CRM automation, customer support, and sales workflows.

## Tech Stack

- Python 3.12
- AutoGen 0.7.5
- Azure OpenAI (GPT-4o)

## Lessons - AutoGen Core Patterns

| # | Lesson | Pattern | File |
|---|--------|---------|------|
| 1 | First Agent | Single agent answering questions | `01_first_agent.py` |
| 2 | Two-Agent Chat | Coder + Reviewer collaboration | `02_two_agent_chat.py` |
| 3 | Tool / Function Calling | Agent using custom tools (CRM lookup, calculator) | `03_tool_use.py` |
| 4 | Group Chat | Round-robin multi-agent discussion | `04_group_chat.py` |
| 5 | Fan-Out / Fan-In | Parallel agents + aggregator | `05_fanout_fanin.py` |
| 6 | Selector Group Chat | LLM dynamically picks next speaker | `06_selector_group_chat.py` |
| 7 | Supervisor-Worker | Boss agent delegates to workers (Swarm) | `07_supervisor_worker.py` |
| 8 | Human-in-the-Loop | Agent proposes, human approves | `08_human_in_loop.py` |
| 9 | Agent Memory | Short-term and long-term memory | `09_agent_memory.py` |
| 10 | Structured Output | Force agents to return Pydantic models | `10_structured_output.py` |
| 11 | Guardrails & Safety | Input/output/tool guardrails | `11_guardrails.py` |
| 12 | Custom Agent | Build your own agent class (no LLM) | `12_custom_agent.py` |

## Agent Orchestration Patterns Covered

```
1. Round Robin        - Fixed turn order (Lesson 4)
2. Selector           - LLM picks next speaker (Lesson 6)
3. Supervisor-Worker  - Boss delegates to workers (Lesson 7)
4. Fan-Out / Fan-In   - Parallel execution + aggregation (Lesson 5)
5. Human-in-the-Loop  - Human approval gates (Lesson 8)
6. Swarm / Handoff    - Agents transfer control to each other (Lesson 7)
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install autogen-agentchat autogen-ext[openai] python-dotenv

# Configure Azure OpenAI (create .env file)
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Run any lesson
python 01_first_agent.py
```

## Author

**Suman Rao Balumuri**
Solution Architect | Agentic AI Enthusiast
