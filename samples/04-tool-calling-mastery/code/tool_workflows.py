# Sample 04: Multi-Step Tool Workflow Example
# Demonstrates complex workflows with multiple sequential tool calls

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_oci import ChatOCIGenAI
import os

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")


# Research workflow tools
@tool
def search_articles(topic: str) -> str:
    """Search for articles on a topic. Returns article IDs."""
    # Simulated search results
    return f"Found articles on '{topic}': [article_001, article_002, article_003]"


@tool
def get_article_content(article_id: str) -> str:
    """Get the content of a specific article."""
    articles = {
        "article_001": "AI is transforming healthcare with new diagnostic tools...",
        "article_002": "Machine learning models now predict patient outcomes...",
        "article_003": "Hospitals adopting AI see 30% improvement in efficiency...",
    }
    return articles.get(article_id, f"Article {article_id} not found")


@tool
def summarize_text(text: str) -> str:
    """Summarize a piece of text."""
    # Simulated summarization
    return f"Summary: {text[:100]}... (key points extracted)"


@tool
def save_research_note(note: str) -> str:
    """Save a research note to the database."""
    return f"Note saved successfully: '{note[:50]}...'"


def run_workflow(
    llm_with_tools, messages: list, tools_dict: dict, max_iterations: int = 10
):
    """Run a multi-step tool workflow until completion."""

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            # No more tool calls - we have the final answer
            print("Final answer reached!")
            return response.content

        print(f"Tool calls: {[tc['name'] for tc in response.tool_calls]}")

        # Execute tools
        messages.append(response)
        for tc in response.tool_calls:
            tool_func = tools_dict[tc["name"]]
            result = tool_func.invoke(tc["args"])
            print(f"  {tc['name']} -> {result[:60]}...")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return "Max iterations reached"


def main():
    # Create chat model with workflow-appropriate settings
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        max_sequential_tool_calls=10,  # Allow multi-step workflows
        tool_result_guidance=True,  # Help model use results
    )

    tools = [search_articles, get_article_content, summarize_text, save_research_note]
    tools_dict = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    # Complex research request
    print("Request: Research AI in healthcare, summarize findings, and save notes")
    print("=" * 60)

    messages = [
        HumanMessage(
            content="Research AI in healthcare. Get the first article, "
            "summarize it, and save a note with the key findings."
        )
    ]

    final_answer = run_workflow(llm_with_tools, messages, tools_dict)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(final_answer)


if __name__ == "__main__":
    main()
