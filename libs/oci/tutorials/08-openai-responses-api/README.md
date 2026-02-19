# Tutorial 08: OpenAI Responses API

Use OpenAI-compatible models through OCI with conversation persistence and advanced features.

## What You'll Learn

- Configure `ChatOCIOpenAI` for OpenAI Responses API
- Use conversation stores for persistent memory
- Authenticate with various OCI methods
- Access web search and MCP tools
- Migrate from OpenAI to OCI

## Prerequisites

- Completed [Tutorial 01: Getting Started](../01-getting-started/)
- Access to OCI Generative AI with OpenAI-compatible models
- Additional packages installed

```bash
pip install oci-openai langchain-openai langchain-oci
```

## Concepts Covered

| Class/Feature | Description |
|---------------|-------------|
| `ChatOCIOpenAI` | OCI client for OpenAI Responses API |
| Conversation Store | Persistent conversation memory |
| Web Search | Built-in web search tool |
| Hosted MCP | Model Context Protocol integration |

---

## Part 1: What is ChatOCIOpenAI?

`ChatOCIOpenAI` provides access to OpenAI-compatible models through OCI's Generative AI service. It extends LangChain's `ChatOpenAI` with:

- OCI authentication (API keys, session tokens, principals)
- Conversation stores for persistent memory
- Regional endpoints across OCI
- Access to web search and hosted MCP tools

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Your Application                       │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │              ChatOCIOpenAI                        │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │  │
│   │  │ OCI Auth │  │ Conv.    │  │ OpenAI       │   │  │
│   │  │ Handler  │  │ Store    │  │ Responses API│   │  │
│   │  └──────────┘  └──────────┘  └──────────────┘   │  │
│   └──────────────────────────────────────────────────┘  │
│                          │                               │
└──────────────────────────┼───────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  OCI Generative AI     │
              │  OpenAI-Compatible     │
              │  Endpoint              │
              └────────────────────────┘
```

---

## Part 2: Authentication Setup

`ChatOCIOpenAI` uses the `oci-openai` package for authentication.

### Session Token Authentication

```python
from oci_openai import OciSessionAuth
from langchain_oci import ChatOCIOpenAI

# Authenticate with OCI CLI session
# First run: oci session authenticate --profile-name MY_PROFILE
auth = OciSessionAuth(profile_name="MY_PROFILE")

client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)
```

### Resource Principal Authentication

For OCI Functions, Jobs, and other OCI resources:

```python
from oci_openai import OciResourcePrincipalAuth
from langchain_oci import ChatOCIOpenAI

auth = OciResourcePrincipalAuth()

client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)
```

### Instance Principal Authentication

For OCI Compute instances:

```python
from oci_openai import OciInstancePrincipalAuth
from langchain_oci import ChatOCIOpenAI

auth = OciInstancePrincipalAuth()

client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)
```

---

## Part 3: Basic Usage

### Simple Invocation

```python
from oci_openai import OciSessionAuth
from langchain_oci import ChatOCIOpenAI

client = ChatOCIOpenAI(
    auth=OciSessionAuth(profile_name="DEFAULT"),
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)

# Simple message
response = client.invoke("What is the capital of France?")
print(response.content)
```

### With System Message

```python
messages = [
    ("system", "You are a helpful translator. Translate user messages to French."),
    ("human", "Hello, how are you?"),
]

response = client.invoke(messages)
print(response.content)  # "Bonjour, comment allez-vous?"
```

### Prompt Chaining

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{input}"),
])

chain = prompt | client

response = chain.invoke({
    "input_language": "English",
    "output_language": "German",
    "input": "I love programming.",
})
print(response.content)  # "Ich liebe Programmieren."
```

---

## Part 4: Conversation Stores

Persist conversations across sessions using OCI Conversation Stores.

### Creating a Conversation Store

First, create a conversation store in OCI Console or via CLI:

```bash
oci generative-ai conversation-store create \
    --compartment-id ocid1.compartment.oc1..xxx \
    --display-name "My Conversation Store" \
    --region us-chicago-1
```

### Using Conversation Store

```python
from oci_openai import OciSessionAuth
from langchain_oci import ChatOCIOpenAI

# With conversation store for persistent memory
client = ChatOCIOpenAI(
    auth=OciSessionAuth(profile_name="DEFAULT"),
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
    conversation_store_id="ocid1.conversationstore.oc1..xxx",
)

# First conversation
response1 = client.invoke("My name is Alice.")
print(response1.content)

# Later conversation - model remembers
response2 = client.invoke("What is my name?")
print(response2.content)  # "Your name is Alice."
```

---

## Part 5: Function Calling

Bind tools to the model for function calling.

```python
from pydantic import BaseModel, Field
from langchain_oci import ChatOCIOpenAI

class GetWeather(BaseModel):
    """Get weather for a location."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)

# Bind tools
llm_with_tools = client.bind_tools([GetWeather])

# Model will call the function
response = llm_with_tools.invoke("What is the weather like in San Francisco?")

# Access tool calls
for tool_call in response.tool_calls:
    print(f"Function: {tool_call['name']}")
    print(f"Arguments: {tool_call['args']}")
```

---

## Part 6: Web Search

Use the built-in web search capability.

```python
from langchain_oci import ChatOCIOpenAI

client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)

# Enable web search
web_search_tool = {"type": "web_search_preview"}
llm_with_search = client.bind_tools([web_search_tool])

# Ask about current events
response = llm_with_search.invoke("What was a positive news story from today?")
print(response.content)
```

---

## Part 7: Hosted MCP (Model Context Protocol)

Access external knowledge sources via hosted MCP servers.

```python
from langchain_oci import ChatOCIOpenAI

client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
)

# Bind MCP tool
mcp_tool = {
    "type": "mcp",
    "server_label": "deepwiki",
    "server_url": "https://mcp.deepwiki.com/mcp",
    "require_approval": "never",
}

llm_with_mcp = client.bind_tools([mcp_tool])

# Query external knowledge
response = llm_with_mcp.invoke(
    "What transport protocols does the 2025-03-26 version of the MCP "
    "spec (modelcontextprotocol/modelcontextprotocol) support?"
)
print(response.content)
```

---

## Part 8: Endpoint Configuration

Configure custom endpoints for different regions or setups.

### Using Region

```python
client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",  # Automatically constructs endpoint
    model="openai.gpt-4.1",
)
```

### Using Service Endpoint

```python
client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    model="openai.gpt-4.1",
)
```

### Using Base URL

```python
client = ChatOCIOpenAI(
    auth=auth,
    compartment_id="ocid1.compartment.oc1..xxx",
    base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/v1",
    model="openai.gpt-4.1",
)
```

---

## Part 9: Migration from OpenAI

Migrating from OpenAI to OCI is straightforward.

### Before (OpenAI)

```python
from langchain_openai import ChatOpenAI

client = ChatOpenAI(
    api_key="sk-...",
    model="gpt-4",
)

response = client.invoke("Hello!")
```

### After (OCI)

```python
from oci_openai import OciSessionAuth
from langchain_oci import ChatOCIOpenAI

client = ChatOCIOpenAI(
    auth=OciSessionAuth(profile_name="DEFAULT"),
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",  # OCI model name
)

response = client.invoke("Hello!")
```

### Key Differences

| Aspect | OpenAI | OCI |
|--------|--------|-----|
| Authentication | API key | OCI auth (session, principal) |
| Model names | `gpt-4` | `openai.gpt-4.1` |
| Endpoint | OpenAI servers | OCI regional endpoints |
| Conversation store | N/A | Built-in support |

---

## Summary

You learned how to:

- Configure `ChatOCIOpenAI` with various authentication methods
- Use conversation stores for persistent memory
- Perform function calling with custom tools
- Access web search capabilities
- Integrate hosted MCP servers
- Migrate from OpenAI to OCI

## Next Steps

- [Tutorial 09: Provider Deep Dive](../09-provider-deep-dive/) - Explore provider-specific features
- [Tutorial 10: Embeddings](../10-embeddings/) - Text and image embeddings

## API Reference

| Class/Function | Description |
|----------------|-------------|
| `ChatOCIOpenAI` | OpenAI Responses API client for OCI |
| `OciSessionAuth` | Session token authentication |
| `OciResourcePrincipalAuth` | Resource principal auth |
| `OciInstancePrincipalAuth` | Instance principal auth |

## Troubleshooting

### Import Error

```
ImportError: oci-openai not found
```
- Install: `pip install oci-openai`

### Authentication Failed

```
401 Unauthorized
```
- Refresh session: `oci session authenticate --profile-name MY_PROFILE`
- Check profile name matches config

### Conversation Store Not Found

```
NotAuthorizedOrNotFound: conversation_store_id
```
- Verify conversation store OCID is correct
- Ensure compartment has access to the store
- Check IAM policies for conversation store operations

### Model Not Available

```
Model 'openai.gpt-4.1' not found
```
- Check model is available in your region
- Verify compartment has GenAI access
- Use `oci generative-ai model list` to see available models
