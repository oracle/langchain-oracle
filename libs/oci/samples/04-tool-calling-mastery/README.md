# Sample 04: Tool Calling Mastery

Master the art of tool calling - from basic definitions to parallel execution and complex workflows.

## What You'll Build

By the end of this sample, you'll be able to:
- Define tools with the `@tool` decorator
- Use `bind_tools()` for chat models
- Handle tool call responses
- Enable parallel tool execution
- Configure tool calling behavior
- Implement complex multi-step workflows

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- Completed [Sample 03: Building AI Agents](../03-building-ai-agents/) (recommended)

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `@tool` decorator | Define tools from functions |
| `bind_tools()` | Attach tools to chat models |
| `tool_calls` | Tool call requests from AI |
| `parallel_tool_calls` | Execute multiple tools at once |
| `max_sequential_tool_calls` | Limit consecutive tool calls |
| `tool_result_guidance` | Help model use tool results |

---

## Part 1: Defining Tools

### Using the `@tool` Decorator

The simplest way to create a tool:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., "Chicago", "Paris")

    Returns:
        Weather description
    """
    return f"Weather in {city}: 72°F, sunny"
```

**Important:** The docstring becomes the tool's description - make it clear!

### With Type Hints (Pydantic)

For more complex parameters:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchParams(BaseModel):
    """Parameters for web search."""
    query: str = Field(description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")
    language: str = Field(default="en", description="Result language code")

@tool(args_schema=SearchParams)
def search_web(query: str, max_results: int = 10, language: str = "en") -> str:
    """Search the web for information."""
    return f"Found {max_results} results for '{query}' in {language}"
```

---

## Part 2: Binding Tools to Chat Models

### The `bind_tools()` Method

Attach tools to a chat model:

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Bind tools to the model
llm_with_tools = llm.bind_tools([get_weather, search_web])
```

### Invoking with Tools

```python
from langchain_core.messages import HumanMessage

response = llm_with_tools.invoke([
    HumanMessage(content="What's the weather in Tokyo?")
])

# Check if the model wants to call a tool
if response.tool_calls:
    print(f"Tool to call: {response.tool_calls[0]['name']}")
    print(f"Arguments: {response.tool_calls[0]['args']}")
else:
    print(f"Direct response: {response.content}")
```

---

## Part 3: Handling Tool Responses

### The Tool Calling Flow

```
1. User message
2. Model returns tool_calls (what to call)
3. You execute the tool
4. Send ToolMessage with result
5. Model uses result to respond
```

### Complete Example

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_oci import ChatOCIGenAI

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))  # Use safe eval in production!

llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

llm_with_tools = llm.bind_tools([calculate])

# Step 1: User asks a question
messages = [HumanMessage(content="What is 25 * 47?")]
response = llm_with_tools.invoke(messages)

# Step 2: Model wants to call a tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"Calling: {tool_call['name']}({tool_call['args']})")

    # Step 3: Execute the tool
    result = calculate.invoke(tool_call['args'])

    # Step 4: Add AI message and tool result to history
    messages.append(response)  # AI message with tool_calls
    messages.append(ToolMessage(
        content=result,
        tool_call_id=tool_call['id']
    ))

    # Step 5: Get final response
    final_response = llm_with_tools.invoke(messages)
    print(f"Answer: {final_response.content}")
```

**Output:**
```
Calling: calculate({'expression': '25 * 47'})
Answer: 25 multiplied by 47 equals 1175.
```

---

## Part 4: Parallel Tool Execution

Some models (like Llama 4) can call multiple tools at once:

### Enabling Parallel Tool Calls

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

llm_with_tools = llm.bind_tools(
    [get_weather, get_time, search_web],
    parallel_tool_calls=True,  # Enable parallel calls
)
```

### Handling Multiple Tool Calls

```python
response = llm_with_tools.invoke([
    HumanMessage(content="What's the weather and time in Chicago and New York?")
])

# Multiple tool calls may be returned
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")
```

**Output:**
```
Tool: get_weather, Args: {'city': 'Chicago'}
Tool: get_weather, Args: {'city': 'New York'}
Tool: get_time, Args: {'city': 'Chicago'}
Tool: get_time, Args: {'city': 'New York'}
```

### Execute All Tools in Parallel

```python
import asyncio

async def execute_tools_parallel(response, tools_dict):
    """Execute multiple tool calls in parallel."""
    results = []

    for tool_call in response.tool_calls:
        tool = tools_dict[tool_call['name']]
        result = tool.invoke(tool_call['args'])
        results.append(ToolMessage(
            content=result,
            tool_call_id=tool_call['id']
        ))

    return results
```

---

## Part 5: Controlling Tool Calling Behavior

### `max_sequential_tool_calls`

Limit how many tools can be called in a single conversation turn:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    max_sequential_tool_calls=5,  # Stop after 5 tool calls
)
```

### `tool_result_guidance`

Help models use tool results naturally (especially useful for Meta Llama):

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    tool_result_guidance=True,  # Inject guidance after tool results
)
```

This injects a system message telling the model:
> "Respond with a helpful, natural language answer that incorporates the tool results."

### `tool_choice`

Force or prevent tool usage:

```python
# Force a specific tool
llm_with_tools = llm.bind_tools(tools, tool_choice="get_weather")

# Force any tool to be called
llm_with_tools = llm.bind_tools(tools, tool_choice="required")

# Prevent tool calls
llm_with_tools = llm.bind_tools(tools, tool_choice="none")

# Let model decide (default)
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
```

---

## Part 6: Infinite Loop Detection

The system automatically detects when the same tool is called repeatedly:

### How It Works

```python
# If the model calls get_weather("Chicago") twice in a row
# with the exact same arguments, the system will:
# 1. Detect the infinite loop
# 2. Force the model to stop calling tools
# 3. Require a natural language response
```

### Customize Loop Prevention

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    max_sequential_tool_calls=8,   # Higher limit for complex workflows
    tool_result_guidance=True,     # Help model use results
)
```

---

## Part 7: Complex Multi-Step Workflows

### Research Assistant Example

```python
from langchain_core.tools import tool

@tool
def search_papers(topic: str) -> str:
    """Search for academic papers on a topic."""
    return f"Found 5 papers on '{topic}': [Paper1, Paper2, ...]"

@tool
def get_paper_summary(paper_id: str) -> str:
    """Get the summary of a specific paper."""
    return f"Summary of {paper_id}: This paper discusses..."

@tool
def save_notes(content: str) -> str:
    """Save research notes."""
    return f"Notes saved: {content[:50]}..."

# Create workflow
llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    max_sequential_tool_calls=10,
)

llm_with_tools = llm.bind_tools([search_papers, get_paper_summary, save_notes])

# The model will:
# 1. Search for papers
# 2. Get summaries of interesting ones
# 3. Save notes
# All in sequence!
```

---

## Part 8: Best Practices

### Tool Design

1. **Clear docstrings** - The description is crucial for the model
2. **Typed arguments** - Use type hints for clarity
3. **Reasonable defaults** - Make tools easy to use
4. **Informative returns** - Return enough context for the model

### Performance

1. **Batch similar operations** - Group related tool calls
2. **Use parallel calls** - When tools are independent
3. **Set limits** - Use `max_sequential_tool_calls` to prevent runaway

### Debugging

```python
# Print tool calls for debugging
def debug_tool_calls(response):
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"[DEBUG] Tool: {tc['name']}")
            print(f"[DEBUG] Args: {tc['args']}")
            print(f"[DEBUG] ID: {tc['id']}")
```

---

## Summary

In this sample, you learned:

1. **Tool definition** - `@tool` decorator with docstrings
2. **`bind_tools()`** - Attach tools to chat models
3. **Tool call flow** - Request → Execute → ToolMessage → Response
4. **Parallel execution** - Multiple tools at once
5. **Behavior control** - `max_sequential_tool_calls`, `tool_result_guidance`
6. **Loop prevention** - Automatic infinite loop detection
7. **Complex workflows** - Multi-step tool orchestration

## Next Steps

- **[Sample 05: Structured Output](../05-structured-output/)** - Get typed responses
- **[Sample 07: Async for Production](../07-async-for-production/)** - Async tool execution

## API Reference

| Method/Parameter | Description |
|------------------|-------------|
| `@tool` | Decorator to create tools |
| `bind_tools(tools)` | Attach tools to model |
| `parallel_tool_calls` | Enable parallel tool calls |
| `tool_choice` | Control tool selection |
| `max_sequential_tool_calls` | Limit tool calls per turn |
| `tool_result_guidance` | Guide model to use results |

## Troubleshooting

### "Tool not being called"
- Check the tool's docstring is clear and descriptive
- Verify the user's request clearly relates to the tool

### "Wrong arguments passed"
- Add `Field(description=...)` to parameters
- Use Pydantic models for complex arguments

### "Model ignores tool results"
- Enable `tool_result_guidance=True`
- Check tool returns informative results
