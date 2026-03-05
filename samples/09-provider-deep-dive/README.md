# Sample 09: Provider Deep Dive

Understand the provider architecture and master provider-specific features.

## What You'll Learn

- Understand the provider abstraction pattern
- Master Meta Llama features (vision, parallel tools)
- Use Google Gemini multimodal capabilities
- Work with Cohere Command models (RAG, V2 API)
- Leverage xAI Grok reasoning features
- Use OpenAI models via ChatOCIOpenAI
- Handle provider-specific quirks

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- Completed [Sample 02: Vision & Multimodal](../02-vision-and-multimodal/)
- Completed [Sample 04: Tool Calling Mastery](../04-tool-calling-mastery/)

## Concepts Covered

| Provider | Key Features |
|----------|--------------|
| `GenericProvider` | Base for Meta, xAI, Mistral |
| `MetaProvider` | Llama 3.2/3.3/4, vision, parallel tools |
| `GeminiProvider` | Multimodal (PDF, video, audio) |
| `CohereProvider` | RAG, citations, V2 vision API |
| `ChatOCIOpenAI` | OpenAI models (gpt-4.1, o1), conversation stores |

---

## Part 1: Provider Architecture

The provider system abstracts model-specific behaviors behind a common interface.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   ChatOCIGenAI                          │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │              Provider Selection                   │  │
│   │                                                    │  │
│   │  model_id="meta.llama-3.3-70b-instruct"          │  │
│   │       ↓                                           │  │
│   │  provider="meta" → MetaProvider()                │  │
│   │                                                    │  │
│   │  model_id="cohere.command-r-plus"                │  │
│   │       ↓                                           │  │
│   │  provider="cohere" → CohereProvider()            │  │
│   └──────────────────────────────────────────────────┘  │
│                                                          │
│   Each provider handles:                                 │
│   • Message format conversion                            │
│   • Tool calling format                                  │
│   • Response parsing                                     │
│   • Streaming events                                     │
│   • Provider-specific features                           │
└─────────────────────────────────────────────────────────┘
```

### Provider Hierarchy

```
Provider (base)
├── GenericProvider (Meta, xAI, OpenAI, Mistral)
│   └── GeminiProvider (Gemini-specific)
└── CohereProvider (Cohere-specific)
```

> **Note:** `MetaProvider` is deprecated. Use `GenericProvider` (or let the library auto-detect) for Llama models.

### Auto-Detection

Providers are auto-detected from model IDs:

```python
from langchain_oci import ChatOCIGenAI

# Auto-detects MetaProvider
llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", ...)

# Auto-detects CohereProvider
llm = ChatOCIGenAI(model_id="cohere.command-r-plus", ...)

# Auto-detects GeminiProvider
llm = ChatOCIGenAI(model_id="google.gemini-2.5-flash", ...)
```

### Manual Override

For DAC/imported models, specify the provider:

```python
llm = ChatOCIGenAI(
    model_id="ocid1.generativeaiendpoint.oc1...",  # Endpoint OCID
    provider="meta",  # "meta", "cohere", "google", "generic"
    ...
)
```

---

## Part 2: Meta Llama Models

Meta provides Llama models with vision and advanced tool calling.

### Available Models

| Model | Features |
|-------|----------|
| `meta.llama-3.2-11b-vision-instruct` | Vision |
| `meta.llama-3.2-90b-vision-instruct` | Vision |
| `meta.llama-3.3-70b-instruct` | Text, tools |
| `meta.llama-4-scout-17b-16e-instruct` | Parallel tools |
| `meta.llama-4-maverick-17b-128e-instruct-fp8` | Parallel tools |

### Basic Usage

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

response = llm.invoke("Explain quantum computing.")
print(response.content)
```

### Vision with Llama 3.2

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
        {"type": "text", "text": "Describe this image in detail."},
        load_image("./photo.jpg"),
    ]
)

response = llm.invoke([message])
print(response.content)
```

### Parallel Tool Calls (Llama 4+)

Llama 4 models support calling multiple tools simultaneously:

```python
from langchain_core.tools import tool
from langchain_oci import ChatOCIGenAI

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72F"

@tool
def get_time(city: str) -> str:
    """Get current time in a city."""
    return f"Time in {city}: 3:00 PM"

llm = ChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    ...
)

# Enable parallel tool calls
llm_with_tools = llm.bind_tools(
    [get_weather, get_time],
    parallel_tool_calls=True,
)

# Both tools called in single response
response = llm_with_tools.invoke(
    "What's the weather and time in New York and London?"
)

for tc in response.tool_calls:
    print(f"Tool: {tc['name']}, Args: {tc['args']}")
```

### Tool Result Guidance

Help Meta models use tool results naturally:

```python
llm_with_tools = llm.bind_tools(
    [get_weather],
    tool_result_guidance=True,  # Helps model synthesize results
    max_sequential_tool_calls=5,  # Prevents infinite loops
)
```

---

## Part 3: Google Gemini Models

Gemini offers advanced multimodal capabilities.

### Available Models

| Model | Features |
|-------|----------|
| `google.gemini-2.5-flash` | Fast, multimodal |
| `google.gemini-2.5-pro` | Most capable |

### Basic Usage

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

response = llm.invoke("What are the key features of Python 3.12?")
print(response.content)
```

### PDF Processing

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(model_id="google.gemini-2.5-flash", ...)

# Load PDF
with open("document.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode()

message = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize the key points from this document."},
        {"type": "media", "data": pdf_data, "mime_type": "application/pdf"},
    ]
)

response = llm.invoke([message])
print(response.content)
```

### Video Analysis

```python
import base64
from langchain_core.messages import HumanMessage

# Load video
with open("video.mp4", "rb") as f:
    video_data = base64.b64encode(f.read()).decode()

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe what happens in this video."},
        {"type": "media", "data": video_data, "mime_type": "video/mp4"},
    ]
)

response = llm.invoke([message])
```

### Audio Analysis

```python
import base64
from langchain_core.messages import HumanMessage

# Load audio
with open("audio.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode()

message = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe and summarize this audio."},
        {"type": "media", "data": audio_data, "mime_type": "audio/mp3"},
    ]
)

response = llm.invoke([message])
```

### Gemini-Specific Parameters

```python
llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    model_kwargs={
        "max_tokens": 1024,  # Note: max_tokens, not max_output_tokens
        "temperature": 0.7,
    },
    ...
)
```

**Note:** The OCI API uses `max_tokens` for Gemini, not `max_output_tokens`. The provider automatically maps `max_output_tokens` to `max_tokens` with a warning.

---

## Part 4: Cohere Command Models

Cohere excels at RAG and provides citations.

### Available Models

| Model | Features |
|-------|----------|
| `cohere.command-r-plus` | Powerful reasoning |
| `cohere.command-a-03-2025` | Latest |
| `cohere.command-a-vision` | Vision (V2 API, DAC only) |

### Basic Usage

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="cohere.command-r-plus",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

response = llm.invoke("Explain the theory of relativity.")
print(response.content)
```

### RAG with Citations

Cohere returns citations when documents are provided:

```python
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOCIGenAI(model_id="cohere.command-r-plus", ...)

# RAG-style prompt with context
messages = [
    SystemMessage(content="""Use the following documents to answer:

Document 1: Python was created by Guido van Rossum in 1991.
Document 2: Python 3.0 was released on December 3, 2008.
Document 3: Python is known for its clear syntax and readability."""),
    HumanMessage(content="When was Python created and by whom?"),
]

response = llm.invoke(messages)
print(response.content)

# Access citations in generation info
if response.response_metadata.get("citations"):
    print("\nCitations:", response.response_metadata["citations"])
```

### Tool Calling with Cohere

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """Search documents for information."""
    return f"Results for: {query}"

llm = ChatOCIGenAI(model_id="cohere.command-r-plus", ...)
llm_with_tools = llm.bind_tools([search_docs])

# Note: Cohere doesn't support parallel_tool_calls
response = llm_with_tools.invoke("Search for Python samples")
```

### Cohere V2 Vision (DAC Only)

Vision support requires dedicated AI cluster:

```python
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI, load_image

# V2 API for vision - requires DAC
llm = ChatOCIGenAI(
    model_id="ocid1.generativeaiendpoint.oc1..xxx",  # DAC endpoint
    provider="cohere",
    ...
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        load_image("./image.jpg"),
    ]
)

response = llm.invoke([message])
```

---

## Part 5: xAI Grok Models

Grok offers reasoning capabilities and vision.

### Available Models

| Model | Features |
|-------|----------|
| `xai.grok-4` | Vision, reasoning |
| `xai.grok-4-fast-reasoning` | Optimized reasoning |

### Basic Usage

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="xai.grok-4",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

response = llm.invoke("Solve this logic puzzle: ...")
print(response.content)
```

### Accessing Reasoning Content

Grok reasoning models expose their thinking:

```python
llm = ChatOCIGenAI(model_id="xai.grok-4-fast-reasoning", ...)

response = llm.invoke("What is 23 * 47? Show your reasoning.")

# Access reasoning from response metadata
if response.response_metadata.get("reasoning_content"):
    print("Reasoning:", response.response_metadata["reasoning_content"])

print("Answer:", response.content)
```

### Vision with Grok

```python
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI, load_image

llm = ChatOCIGenAI(model_id="xai.grok-4", ...)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze this chart and explain the trends."},
        load_image("./chart.png"),
    ]
)

response = llm.invoke([message])
print(response.content)
```

---

## Part 6: OpenAI Models (via ChatOCIOpenAI)

For OpenAI models deployed in OCI, use `ChatOCIOpenAI` instead of `ChatOCIGenAI`.

### Available Models

| Model | Features |
|-------|----------|
| `openai.gpt-4.1` | Tools, reasoning |
| `openai.o1` | Advanced reasoning |

### Basic Usage

```python
from langchain_oci import ChatOCIOpenAI
from oci.auth.signers import get_resource_principals_signer

# Using resource principal auth (OCI Functions, Jobs)
signer = get_resource_principals_signer()

llm = ChatOCIOpenAI(
    auth=signer,
    compartment_id="ocid1.compartment.oc1..xxx",
    model="openai.gpt-4.1",
    region="us-chicago-1",
)

response = llm.invoke("What are the benefits of cloud computing?")
print(response.content)
```

### With Conversation Store

OpenAI models support persistent conversation memory:

```python
llm = ChatOCIOpenAI(
    auth=signer,
    compartment_id="ocid1.compartment.oc1..xxx",
    model="openai.gpt-4.1",
    conversation_store_id="ocid1.generativeaiagentconversation.oc1..xxx",
    region="us-chicago-1",
)
```

> **Note:** `ChatOCIOpenAI` uses the OpenAI Responses API and has different initialization parameters than `ChatOCIGenAI`.

---

## Part 7: Provider Comparison

### Feature Matrix

| Feature | Meta | Gemini | Cohere | xAI | OpenAI |
|---------|------|--------|--------|-----|--------|
| Vision | ✅ Llama 3.2 | ✅ All | ✅ V2/DAC | ✅ | ✅ GPT-4o |
| PDF | ❌ | ✅ | ❌ | ❌ | ❌ |
| Video | ❌ | ✅ | ❌ | ❌ | ❌ |
| Audio | ❌ | ✅ | ❌ | ❌ | ❌ |
| Parallel Tools | ✅ Llama 4+ | ❌ | ❌ | ❌ | ✅ |
| Citations | ❌ | ❌ | ✅ | ❌ | ❌ |
| Reasoning | ❌ | ❌ | ❌ | ✅ | ✅ o1 |
| tool_choice | ✅ | ✅ | ❌ | ✅ | ✅ |

### Performance Characteristics

| Provider | Latency | Throughput | Best For |
|----------|---------|------------|----------|
| Meta Llama 4 | Low | High | Production, tools |
| Gemini Flash | Very Low | Very High | Multimodal, speed |
| Cohere Command | Medium | Medium | RAG, search |
| xAI Grok | Medium | Medium | Reasoning tasks |
| OpenAI GPT-4 | Medium | Medium | General tasks, tools |

---

## Part 8: Best Practices

### Choosing a Provider

```python
# For vision tasks → Llama 3.2, Gemini, or Grok
if need_vision and not need_pdf:
    model = "meta.llama-3.2-90b-vision-instruct"
elif need_multimodal:  # PDF, video, audio
    model = "google.gemini-2.5-flash"

# For tool-heavy workflows → Llama 4 (parallel tools)
if many_tools and need_parallel:
    model = "meta.llama-4-scout-17b-16e-instruct"

# For RAG with citations → Cohere
if need_citations:
    model = "cohere.command-r-plus"

# For reasoning tasks → Grok
if need_reasoning:
    model = "xai.grok-4-fast-reasoning"
```

### Handling Provider Differences

```python
from langchain_oci import ChatOCIGenAI

def get_llm_for_task(task_type: str) -> ChatOCIGenAI:
    """Get appropriate LLM for task type."""
    configs = {
        "vision": {
            "model_id": "meta.llama-3.2-90b-vision-instruct",
        },
        "multimodal": {
            "model_id": "google.gemini-2.5-flash",
        },
        "rag": {
            "model_id": "cohere.command-r-plus",
        },
        "reasoning": {
            "model_id": "xai.grok-4-fast-reasoning",
        },
        "tools": {
            "model_id": "meta.llama-4-scout-17b-16e-instruct",
        },
    }

    config = configs.get(task_type, configs["tools"])

    return ChatOCIGenAI(
        **config,
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..xxx",
    )
```

---

## Summary

You learned:

- How the provider architecture abstracts model differences
- Meta Llama features: vision, parallel tools, guidance
- Google Gemini multimodal: PDF, video, audio
- Cohere features: RAG, citations, V2 vision
- xAI Grok: reasoning content access
- How to choose providers for different tasks

## Next Steps

- [Sample 10: Embeddings](../10-embeddings/) - Text and image embeddings

## API Reference

| Provider Class | Models |
|----------------|--------|
| `GenericProvider` | Meta, xAI, OpenAI, Mistral |
| `MetaProvider` | Meta Llama (extends Generic) |
| `GeminiProvider` | Google Gemini (extends Generic) |
| `CohereProvider` | Cohere Command |

## Troubleshooting

### Wrong Provider Selected

```
Unexpected response format
```
- For DAC endpoints, explicitly set `provider="meta"` etc.

### Tool Choice Not Supported

```
ValueError: Tool choice is not supported for Cohere
```
- Cohere doesn't support `tool_choice` parameter
- Remove it or switch to Meta/Gemini

### Parallel Tools Error

```
Parallel tool calls not supported
```
- Only Llama 4+ supports `parallel_tool_calls=True`
- Use sequential calls for other models

### Vision Not Working

```
Content type not supported
```
- Check model supports vision (`is_vision_model()`)
- Cohere vision requires V2 API on DAC
