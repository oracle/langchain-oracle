# langchain-oci

[![PyPI version](https://img.shields.io/pypi/v/langchain-oci)](https://pypi.org/project/langchain-oci/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-oci)](https://pypi.org/project/langchain-oci/)
[![License](https://img.shields.io/badge/License-UPL%201.0-green)](https://opensource.org/licenses/UPL)

LangChain integrations for Oracle Cloud Infrastructure (OCI) Generative AI.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Chat Models](#chat-models)
- [Vision & Multimodal](#vision--multimodal)
- [Embeddings](#embeddings)
- [Async Support](#async-support)
- [Tool Calling](#tool-calling)
- [Structured Output](#structured-output)
- [AI Agents](#ai-agents)
- [OpenAI Responses API](#openai-responses-api)
- [OCI Data Science Deployments](#oci-data-science-deployments)
- [Provider Reference](#provider-reference)
- [Tutorials](#tutorials)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install langchain-oci oci
```

---

## Quick Start

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..your-compartment-id",
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

---

## Authentication

Four authentication methods are supported:

### API Key (Default)

Uses credentials from `~/.oci/config`:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    auth_type="API_KEY",        # Default
    auth_profile="DEFAULT",      # Profile name in ~/.oci/config
)
```

### Security Token (Session-Based)

```bash
oci session authenticate --profile-name MY_PROFILE
```

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    auth_type="SECURITY_TOKEN",
    auth_profile="MY_PROFILE",
    ...
)
```

### Instance Principal

For applications running on OCI compute instances:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    auth_type="INSTANCE_PRINCIPAL",
    ...
)
```

### Resource Principal

For OCI Functions and other resources:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    auth_type="RESOURCE_PRINCIPAL",
    ...
)
```

---

## Chat Models

### On-Demand Models

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)
```

### DAC/Imported Models

For models deployed on Dedicated AI Clusters:

```python
llm = ChatOCIGenAI(
    model_id="ocid1.generativeaiendpoint.oc1.us-chicago-1.xxxxx",  # Endpoint OCID
    provider="meta",  # "cohere", "google", "meta", "generic"
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)
```

### Provider Matrix

> **Note:** For the most current model list, see the [OCI Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm).

| Provider | Example Models | Features |
|----------|----------------|----------|
| **Meta** | Llama 3.2, 3.3, 4 | Vision, parallel tools |
| **Google** | Gemini 2.0/2.5 Flash | Multimodal (PDF, video, audio) |
| **xAI** | Grok 4 | Vision, reasoning |
| **Cohere** | Command R+, Command A | RAG, vision (V2) |
| **OpenAI** | GPT-4, o1 | Reasoning (via ChatOCIOpenAI) |
| **Mistral** | Mistral models | Fast inference |

---

## Vision & Multimodal

### Vision-Capable Models

```python
from langchain_oci.utils.vision import VISION_MODELS

# 13+ vision-capable models
print(VISION_MODELS)
# ['meta.llama-3.2-90b-vision-instruct', 'google.gemini-2.5-flash', 'xai.grok-4', ...]
```

### Image Analysis

```python
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI, load_image

llm = ChatOCIGenAI(
    model_id="meta.llama-3.2-90b-vision-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image"},
        load_image("./photo.jpg"),
    ]
)

response = llm.invoke([message])
```

### Utility Functions

| Function | Description |
|----------|-------------|
| `load_image(path)` | Load image file as content block |
| `encode_image(bytes, mime_type)` | Encode bytes as content block |
| `to_data_uri(image)` | Convert to data URI string |
| `is_vision_model(model_id)` | Check if model supports vision |

### Gemini Multimodal

Gemini models support PDF, video, and audio:

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(model_id="google.gemini-2.5-flash", ...)

# PDF
with open("doc.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode()

message = HumanMessage(content=[
    {"type": "text", "text": "Summarize this PDF"},
    {"type": "media", "data": pdf_data, "mime_type": "application/pdf"}
])
```

---

## Embeddings

### Text Embeddings

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Single query
vector = embeddings.embed_query("What is machine learning?")

# Multiple documents
vectors = embeddings.embed_documents(["Doc 1", "Doc 2"])
```

### Image Embeddings

```python
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",  # Multimodal model
    ...
)

# Single image
vector = embeddings.embed_image("./photo.jpg")

# Batch
vectors = embeddings.embed_image_batch(["img1.jpg", "img2.jpg"])
```

### Embedding Models

| Model | Type | Dimensions |
|-------|------|------------|
| `cohere.embed-english-v3.0` | Text | 1024 |
| `cohere.embed-multilingual-v3.0` | Text | 1024 |
| `cohere.embed-v4.0` | Text + Image | 256-1536 |

---

## Async Support

All chat models support async operations via LangChain's base classes:

```python
import asyncio
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(...)

async def main():
    # Single async request
    response = await llm.ainvoke("Hello!")

    # Async streaming
    async for chunk in llm.astream("Tell me a story"):
        print(chunk.content, end="")

    # Concurrent requests
    results = await asyncio.gather(
        llm.ainvoke("Question 1"),
        llm.ainvoke("Question 2"),
        llm.ainvoke("Question 3"),
    )

asyncio.run(main())
```

---

## Tool Calling

### Basic Tools

```python
from langchain_core.tools import tool
from langchain_oci import ChatOCIGenAI

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72F, sunny"

llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", ...)
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in Chicago?")
```

### Parallel Tool Calls (Llama 4+)

```python
llm_with_tools = llm.bind_tools(
    [get_weather, get_time],
    parallel_tool_calls=True,  # Enable parallel execution
)
```

### Tool Configuration

| Parameter | Description |
|-----------|-------------|
| `parallel_tool_calls` | Enable parallel tool execution (Llama 4+) |
| `max_sequential_tool_calls` | Limit consecutive tool calls (default: 8) |
| `tool_result_guidance` | Guide model to use tool results naturally |
| `tool_choice` | "auto", "required", "none", or tool name |

---

## Structured Output

```python
from pydantic import BaseModel
from langchain_oci import ChatOCIGenAI

class Contact(BaseModel):
    name: str
    email: str

llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", ...)
structured_llm = llm.with_structured_output(Contact)

result = structured_llm.invoke("Extract: John Doe john@example.com")
print(result.name)   # "John Doe"
print(result.email)  # "john@example.com"
```

### Methods

| Method | Description |
|--------|-------------|
| `function_calling` | Default, most reliable |
| `json_mode` | Simple schemas |
| `json_schema` | Native OCI support |

---

## AI Agents

### create_oci_agent()

```python
from langchain_core.tools import tool
from langchain_oci import create_oci_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[search],
    compartment_id="ocid1.compartment.oc1..xxx",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    system_prompt="You are a helpful assistant.",
)

from langchain_core.messages import HumanMessage
result = agent.invoke({
    "messages": [HumanMessage(content="Search for Python tutorials")]
})
```

### With Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[search],
    checkpointer=MemorySaver(),
    ...
)

# Conversations persist by thread_id
result = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "user_123"}},
)
```

### Human-in-the-Loop

```python
agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[dangerous_action],
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # Pause before tool execution
)
```

---

## OpenAI Responses API

```python
from oci_openai import OciSessionAuth
from langchain_oci import ChatOCIOpenAI

client = ChatOCIOpenAI(
    auth=OciSessionAuth(profile_name="MY_PROFILE"),
    compartment_id="ocid1.compartment.oc1..xxx",
    region="us-chicago-1",
    model="openai.gpt-4.1",
    conversation_store_id="ocid1.conversationstore...",  # Required if store=True
)

response = client.invoke([
    ("system", "You are a helpful assistant."),
    ("human", "Hello!"),
])
```

---

## OCI Data Science Deployments

### ChatOCIModelDeployment

```python
from langchain_oci.chat_models import ChatOCIModelDeployment

endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

chat = ChatOCIModelDeployment(
    endpoint=endpoint,
    streaming=True,
    model_kwargs={"temperature": 0.2, "max_tokens": 512},
)

response = chat.invoke("Hello!")
```

### vLLM/TGI Deployments

```python
from langchain_oci.chat_models import ChatOCIModelDeploymentVLLM

chat = ChatOCIModelDeploymentVLLM(endpoint=endpoint)
response = chat.invoke("Hello!")
```

---

## Provider Reference

### Meta Llama

```python
# Vision models
"meta.llama-3.2-90b-vision-instruct"
"meta.llama-3.2-11b-vision-instruct"

# Text models
"meta.llama-3.3-70b-instruct"

# Llama 4 (parallel tools)
"meta.llama-4-scout-17b-16e-instruct"
"meta.llama-4-maverick-17b-128e-instruct-fp8"
```

### Google Gemini

```python
"google.gemini-2.5-flash"       # Fast, multimodal
"google.gemini-2.5-flash"       # Latest
"google.gemini-2.5-pro"         # Most capable
```

### xAI Grok

```python
"xai.grok-4"                    # Vision + reasoning
"xai.grok-4-fast-reasoning"     # Optimized reasoning
```

### Cohere

```python
"cohere.command-r-plus"         # Powerful reasoning
"cohere.command-a-03-2025"      # Latest
"cohere.command-a-vision"       # Vision support (V2 API)
```

---

## Tutorials

Comprehensive tutorials covering all features:

| Tutorial | Description |
|----------|-------------|
| [01. Getting Started](./tutorials/01-getting-started/) | Authentication, basic chat |
| [02. Vision & Multimodal](./tutorials/02-vision-and-multimodal/) | Images, PDFs, video, audio |
| [03. Building AI Agents](./tutorials/03-building-ai-agents/) | create_oci_agent, checkpointing |
| [04. Tool Calling Mastery](./tutorials/04-tool-calling-mastery/) | Parallel tools, workflows |
| [05. Structured Output](./tutorials/05-structured-output/) | Pydantic, JSON modes |
| [06. Model Deployments](./tutorials/06-model-deployments/) | vLLM, TGI, custom endpoints |
| [07. Async for Production](./tutorials/07-async-for-production/) | ainvoke, astream, FastAPI |
| [08. OpenAI Responses API](./tutorials/08-openai-responses-api/) | ChatOCIOpenAI, conversation stores |
| [09. Provider Deep Dive](./tutorials/09-provider-deep-dive/) | Provider-specific features |
| [10. Embeddings](./tutorials/10-embeddings/) | Text & image embeddings, RAG |

See [tutorials/README.md](./tutorials/README.md) for the full learning path.

---

## Troubleshooting

### Authentication Errors

```
AuthenticationError: Could not authenticate
```
- Verify `~/.oci/config` exists and is valid
- Check profile name matches `auth_profile`
- For session auth: `oci session authenticate --profile-name MY_PROFILE`

### Model Not Found

```
NotAuthorizedOrNotFound: model_id
```
- Verify model ID spelling
- Check model is available in your region
- Ensure compartment has GenAI access

### Tool Calling Issues

```
Model keeps calling the same tool
```
- Enable `tool_result_guidance=True`
- Set `max_sequential_tool_calls` limit
- Check tool returns informative results

### Vision Not Working

```
Content type not supported
```
- Verify using a vision-capable model (`is_vision_model()`)
- Check image format (PNG, JPEG, GIF, WebP)
- Reduce image size if too large

---

## API Reference

### Chat Models

| Class | Description |
|-------|-------------|
| `ChatOCIGenAI` | Main chat model for OCI GenAI |
| `ChatOCIOpenAI` | OpenAI Responses API compatibility |
| `ChatOCIModelDeployment` | Custom OCI Data Science deployments |
| `ChatOCIModelDeploymentVLLM` | vLLM-specific deployment handler |
| `ChatOCIModelDeploymentTGI` | TGI-specific deployment handler |

### Embeddings

| Class | Description |
|-------|-------------|
| `OCIGenAIEmbeddings` | Text and image embeddings |
| `OCIModelDeploymentEndpointEmbeddings` | Custom deployment embeddings |

### Agents

| Function | Description |
|----------|-------------|
| `create_oci_agent()` | Create ReAct agent with tools |

### Utilities

| Function | Description |
|----------|-------------|
| `load_image()` | Load image for vision models |
| `encode_image()` | Encode bytes for vision models |
| `to_data_uri()` | Convert to data URI |
| `is_vision_model()` | Check vision support |

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the [Universal Permissive License (UPL) 1.0](https://opensource.org/licenses/UPL).
