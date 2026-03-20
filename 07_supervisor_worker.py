"""
AutoGen - Lesson 7: Supervisor-Worker Pattern
A supervisor agent explicitly delegates tasks to worker agents using Handoffs.

Pattern:
  Task --> Supervisor --> "Worker A, do this" --> result back to Supervisor
                      --> "Worker B, do this" --> result back to Supervisor
                      --> Supervisor creates final output

Key difference from Selector: Here ONE agent is the BOSS and controls the flow.
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
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

# ---- Worker Agents ----

researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    system_message="""You are a Research Worker.
    When the supervisor asks you to research something, provide detailed findings.
    Keep responses under 100 words.
    When done, transfer back to the supervisor using the transfer function.""",
    handoffs=["supervisor"],  # Can hand back to supervisor
)

writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    system_message="""You are a Content Writer Worker.
    When the supervisor asks you to write something, create professional content.
    Keep responses under 100 words.
    When done, transfer back to the supervisor using the transfer function.""",
    handoffs=["supervisor"],  # Can hand back to supervisor
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="""You are a Quality Review Worker.
    When the supervisor asks you to review, check for quality and accuracy.
    Keep responses under 100 words.
    When done, transfer back to the supervisor using the transfer function.""",
    handoffs=["supervisor"],  # Can hand back to supervisor
)

# ---- Supervisor Agent ----
# The boss - delegates work to the right worker

supervisor = AssistantAgent(
    name="supervisor",
    model_client=model_client,
    system_message="""You are a Supervisor managing a team of 3 workers:
    - researcher: For gathering information and analysis
    - writer: For creating content and documents
    - reviewer: For quality checks and reviews

    Your job:
    1. Break the task into subtasks
    2. Delegate each subtask to the RIGHT worker using transfer functions
    3. After all workers have contributed, create a FINAL summary
    4. Say COMPLETED at the very end

    Always delegate to workers. Do NOT do the work yourself.
    Transfer to one worker at a time.""",
    handoffs=["researcher", "writer", "reviewer"],  # Can delegate to any worker
)

# ---- Termination ----
termination = TextMentionTermination("COMPLETED") | MaxMessageTermination(12)

# ---- Swarm: Agents hand off to each other ----
team = Swarm(
    participants=[supervisor, researcher, writer, reviewer],
    termination_condition=termination,
)


async def main():
    print("=" * 60)
    print("LESSON 7: Supervisor-Worker Pattern")
    print("Supervisor delegates tasks to specialized workers")
    print("=" * 60)

    task = """
    Create a customer win-back email for Acme Corp.
    They churned 3 months ago because of poor support response times.
    We've since improved our support SLA from 24hrs to 4hrs.
    """

    result = await team.run(task=task)

    # Print the conversation
    for message in result.messages:
        print(f"\n{'-' * 50}")
        print(f">> {message.source.upper()}")
        print(f"{'-' * 50}")
        content = message.content
        if isinstance(content, str):
            print(content.encode("ascii", "replace").decode())


if __name__ == "__main__":
    asyncio.run(main())
