# Sample 03: Building AI Agents

Learn how to create autonomous AI agents that can use tools, maintain memory, and interact with users.

## What You'll Build

By the end of this sample, you'll be able to:
- Create a ReAct agent with `create_oci_agent()`
- Define tools for your agent to use
- Add memory with checkpointing
- Implement human-in-the-loop workflows
- Integrate with LangGraph for complex agent patterns

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- Install LangGraph: `pip install langgraph`

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `create_oci_agent()` | Factory function to create agents |
| `@tool` decorator | Define tools for agents |
| Checkpointing | Persist conversation state |
| Human-in-the-loop | Pause for user approval |
| LangGraph | Graph-based agent orchestration |

---

## Part 1: What is an AI Agent?

An **AI agent** is a system that:
1. **Reasons** about what to do (using an LLM)
2. **Acts** by calling tools
3. **Observes** the results
4. **Repeats** until the task is complete

This is called the **ReAct** pattern (Reason + Act).

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Reason  │────▶│   Act   │────▶│ Observe │
│  (LLM)  │◀────│ (Tools) │◀────│(Results)│
└─────────┘     └─────────┘     └─────────┘
```

---

## Part 2: Creating Your First Agent

### Step 1: Define Tools

Tools are functions your agent can call. Use the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., "Chicago", "Paris")
    """
    # In production, call a real weather API
    return f"Weather in {city}: 72°F, sunny"

@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query
    """
    # In production, call a real search API
    return f"Search results for '{query}': Found 10 relevant articles."
```

### Step 2: Create the Agent

```python
from langchain_oci import create_oci_agent

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[get_weather, search_web],
    compartment_id="ocid1.compartment.oc1..xxx",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    system_prompt="You are a helpful assistant with access to weather and search tools.",
)
```

### Step 3: Run the Agent

```python
from langchain_core.messages import HumanMessage

result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in Chicago?")]
})

# Get the final response
final_message = result["messages"][-1]
print(final_message.content)
```

**Output:**
```
The weather in Chicago is 72°F and sunny!
```

---

## Part 3: Understanding Agent Execution

### What Happens Inside

When you invoke an agent:

1. **User message** → Agent receives "What's the weather in Chicago?"
2. **LLM reasons** → "I need to call the get_weather tool"
3. **Tool called** → `get_weather("Chicago")` returns "72°F, sunny"
4. **LLM responds** → "The weather in Chicago is 72°F and sunny!"

### View All Messages

```python
result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in Chicago?")]
})

for msg in result["messages"]:
    print(f"{msg.type}: {msg.content[:100] if msg.content else '(tool call)'}")
```

---

## Part 4: Environment Variables for Convenience

Instead of passing credentials every time, use environment variables:

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
export OCI_REGION="us-chicago-1"
# or
export OCI_SERVICE_ENDPOINT="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
```

Then create agents simply:

```python
from langchain_oci import create_oci_agent

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[get_weather],
    # No need for compartment_id or service_endpoint!
)
```

---

## Part 5: Adding Memory with Checkpointing

Checkpointing allows your agent to remember previous conversations:

```python
from langgraph.checkpoint.memory import MemorySaver
from langchain_oci import create_oci_agent

# Create a checkpointer
checkpointer = MemorySaver()

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[get_weather],
    checkpointer=checkpointer,
)

# First conversation
result1 = agent.invoke(
    {"messages": [HumanMessage(content="What's the weather in Chicago?")]},
    config={"configurable": {"thread_id": "user_123"}},
)

# Later, continue the same conversation
result2 = agent.invoke(
    {"messages": [HumanMessage(content="What about New York?")]},
    config={"configurable": {"thread_id": "user_123"}},
)
# Agent remembers the previous context!
```

### Persistent Checkpointers

For production, use a database-backed checkpointer:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite (local)
checkpointer = SqliteSaver.from_conn_string("agent_memory.db")

# PostgreSQL (production)
# from langgraph.checkpoint.postgres import PostgresSaver
# checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

---

## Part 6: Human-in-the-Loop Workflows

Pause the agent for human approval before taking actions:

```python
from langchain_oci import create_oci_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[get_weather, dangerous_action],
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # Pause before executing tools
)

# Start the conversation
result = agent.invoke(
    {"messages": [HumanMessage(content="Delete all files")]},
    config={"configurable": {"thread_id": "review_123"}},
)

# Agent is paused before calling tools
# Review the pending tool call
pending_tool = result["messages"][-1].tool_calls[0]
print(f"Agent wants to call: {pending_tool['name']}")
print(f"With args: {pending_tool['args']}")

# If approved, continue execution
if user_approves():
    result = agent.invoke(
        None,  # Continue from where we left off
        config={"configurable": {"thread_id": "review_123"}},
    )
```

---

## Part 7: Multi-Tool Orchestration

Agents can use multiple tools in sequence:

```python
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    return f"{ticker}: $150.00"

@tool
def get_company_news(company: str) -> str:
    """Get recent news about a company."""
    return f"Latest news for {company}: Q4 earnings exceeded expectations."

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))  # Use a safe evaluator in production

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[get_stock_price, get_company_news, calculate],
    max_sequential_tool_calls=8,  # Allow up to 8 tool calls per turn
)

result = agent.invoke({
    "messages": [HumanMessage(
        content="What's Apple's stock price and what's the latest news? "
        "Also, if I buy 10 shares, how much would that cost?"
    )]
})
```

The agent will:
1. Call `get_stock_price("AAPL")`
2. Call `get_company_news("Apple")`
3. Call `calculate("150.00 * 10")`
4. Synthesize a final answer

---

## Part 8: Preventing Infinite Loops

Sometimes agents get stuck calling the same tool repeatedly. Use these safeguards:

```python
agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[my_tools],
    max_sequential_tool_calls=8,     # Stop after 8 tool calls
    tool_result_guidance=True,       # Guide model to use tool results
)
```

### How It Works

- **`max_sequential_tool_calls`**: Limits the number of tool calls per conversation turn
- **`tool_result_guidance`**: Injects a system message telling the model to respond naturally after receiving tool results
- **Infinite loop detection**: Automatically detects when the same tool is called with the same arguments repeatedly

---

## Summary

In this sample, you learned:

1. **ReAct pattern** - How agents reason and act
2. **`create_oci_agent()`** - Factory function for creating agents
3. **Tools** - Functions agents can call with `@tool`
4. **Checkpointing** - Memory persistence across conversations
5. **Human-in-the-loop** - Pausing for approval
6. **Multi-tool orchestration** - Complex workflows with multiple tools
7. **Loop prevention** - Safeguards against infinite loops

## Next Steps

- **[Sample 04: Tool Calling Mastery](../04-tool-calling-mastery/)** - Deep dive into tool calling
- **[Sample 05: Structured Output](../05-structured-output/)** - Get structured responses

## API Reference

| Function/Parameter | Description |
|--------------------|-------------|
| `create_oci_agent()` | Create a ReAct agent |
| `model_id` | OCI model identifier |
| `tools` | List of tools the agent can use |
| `checkpointer` | LangGraph checkpointer for persistence |
| `interrupt_before` | Nodes to pause before |
| `max_sequential_tool_calls` | Maximum tool calls per turn |
| `tool_result_guidance` | Guide model to use tool results |

## Troubleshooting

### "Agent keeps calling the same tool"
- Increase `max_sequential_tool_calls` if legitimate
- Enable `tool_result_guidance=True`
- Check that tool return values are informative

### "Tool not found"
- Ensure tool has a docstring (required for description)
- Check tool is in the `tools` list

### "Agent doesn't remember previous messages"
- Ensure you're using the same `thread_id`
- Verify checkpointer is configured correctly
