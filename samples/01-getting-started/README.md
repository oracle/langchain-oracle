# Sample 01: Getting Started with OCI GenAI

Welcome to langchain-oci! This sample will get you up and running with Oracle Cloud Infrastructure's Generative AI service integrated with LangChain.

## What You'll Build

By the end of this sample, you'll be able to:
- Configure authentication for OCI Generative AI
- Create your first chat conversation
- Understand providers and model selection
- Use streaming responses for real-time output

## Prerequisites

- An OCI account with access to Generative AI service
- Python 3.9 or higher
- OCI CLI configured (for API key authentication)

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `ChatOCIGenAI` | Main chat model class |
| Authentication | 4 methods: API Key, Instance Principal, Resource Principal, Session Token |
| Providers | Meta, Cohere, Google (Gemini), xAI (Grok) |
| `invoke()` | Send a message and get a response |
| `stream()` | Get streaming responses |

---

## Part 1: Installation & Setup

### Install the Package

```bash
pip install langchain-oci oci
```

### Configure OCI CLI (API Key Authentication)

If you haven't already, set up the OCI CLI:

```bash
oci setup config
```

This creates `~/.oci/config` with your credentials. The default profile is named `DEFAULT`.

---

## Part 2: Your First Chat

Let's start with the simplest possible example:

```python
from langchain_oci import ChatOCIGenAI

# Create a chat model
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..your-compartment-id",
)

# Send a message
response = llm.invoke("What is the capital of France?")
print(response.content)
```

**Output:**
```
The capital of France is Paris.
```

### Understanding the Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_id` | Yes | The model to use (e.g., `meta.llama-3.3-70b-instruct`) |
| `service_endpoint` | Yes | Regional endpoint for GenAI service |
| `compartment_id` | Yes | Your OCI compartment OCID |
| `auth_type` | No | Authentication method (default: `API_KEY`) |
| `auth_profile` | No | Profile name in `~/.oci/config` (default: `DEFAULT`) |

### Service Endpoints by Region

| Region | Endpoint |
|--------|----------|
| Chicago | `https://inference.generativeai.us-chicago-1.oci.oraclecloud.com` |
| Frankfurt | `https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com` |

---

## Part 3: Authentication Methods

### Method 1: API Key (Default)

Uses credentials from `~/.oci/config`. This is the default and simplest method:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    auth_type="API_KEY",          # Optional, this is the default
    auth_profile="DEFAULT",        # Optional, uses DEFAULT profile
)
```

### Method 2: Security Token (Session-Based)

For interactive sessions with temporary credentials:

```bash
# First, authenticate
oci session authenticate --profile-name MY_PROFILE
```

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    auth_type="SECURITY_TOKEN",
    auth_profile="MY_PROFILE",
)
```

### Method 3: Instance Principal

For applications running on OCI compute instances:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    auth_type="INSTANCE_PRINCIPAL",
)
```

### Method 4: Resource Principal

For OCI Functions and other resources:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    auth_type="RESOURCE_PRINCIPAL",
)
```

---

## Part 4: Choosing a Provider & Model

### Available Providers (Examples)

> **Note:** This is not a comprehensive list. See the [OCI Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) for all available models.

| Provider | Example Models | Strengths |
|----------|----------------|-----------|
| **Meta** | Llama 3.2, 3.3, 4 | Excellent general-purpose, tool calling |
| **Cohere** | Command R+, Command A | RAG, document processing |
| **Google** | Gemini 2.0 Flash, 2.5 | Multimodal (PDF, video, audio) |
| **xAI** | Grok 4 | Fast reasoning, vision |

> **OpenAI Models:** For OpenAI models (GPT-4.1, o1, etc.), use `ChatOCIOpenAI` instead of `ChatOCIGenAI`. See [Sample 08: OpenAI Responses API](../08-openai-responses-api/).

### Popular Model IDs

```python
# Meta Llama models
"meta.llama-3.3-70b-instruct"           # Latest text model
"meta.llama-3.2-90b-vision-instruct"    # Vision-capable
"meta.llama-4-scout-17b-16e-instruct"   # Llama 4 with parallel tools

# Cohere models
"cohere.command-r-plus"                  # Powerful reasoning
"cohere.command-a-03-2025"              # Latest, with vision

# Google Gemini models
"google.gemini-2.5-flash"               # Fast, multimodal

# xAI Grok models
"xai.grok-4"                            # Reasoning and vision
```

### Provider Detection

The provider is automatically detected from the model ID:

```python
# Auto-detects "meta" provider
llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", ...)

# Or specify explicitly
llm = ChatOCIGenAI(model_id="my-custom-model", provider="meta", ...)
```

---

## Part 5: Conversations with Messages

For multi-turn conversations, use LangChain message types:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Multi-turn conversation
messages = [
    SystemMessage(content="You are a helpful cooking assistant."),
    HumanMessage(content="I have chicken, rice, and vegetables."),
    AIMessage(content="Great! You could make a stir-fry or a chicken rice bowl."),
    HumanMessage(content="How do I make a stir-fry?"),
]

response = llm.invoke(messages)
print(response.content)
```

---

## Part 6: Streaming Responses

For real-time output, use streaming:

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Stream the response
for chunk in llm.stream("Tell me a short story about a robot."):
    print(chunk.content, end="", flush=True)
```

---

## Part 7: Model Parameters

Fine-tune model behavior with `model_kwargs`:

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    model_kwargs={
        "temperature": 0.7,      # Creativity (0.0 = deterministic, 1.0 = creative)
        "max_tokens": 500,       # Maximum response length
        "top_p": 0.9,           # Nucleus sampling
        "top_k": 50,            # Top-k sampling
    }
)
```

### Parameter Reference

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0.0 - 1.0 | Higher = more creative, lower = more focused |
| `max_tokens` | Model-dependent | Maximum tokens in the response (see [OCI docs](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)) |
| `top_p` | 0.0 - 1.0 | Nucleus sampling cutoff |
| `top_k` | Model-dependent | Number of top tokens to consider (typically 1-500, varies by model) |

> **Note:** Maximum token limits vary by model. Check the [OCI Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) for specific model limits.

---

## Summary

In this sample, you learned:

1. **Installation** - `pip install langchain-oci oci`
2. **Basic chat** - Using `ChatOCIGenAI` with `invoke()`
3. **Authentication** - 4 methods (API Key, Session, Instance Principal, Resource Principal)
4. **Providers** - Meta, Cohere, Google, xAI and their model families
5. **Conversations** - Multi-turn chat with message types
6. **Streaming** - Real-time responses with `stream()`
7. **Parameters** - Fine-tuning with `model_kwargs`

## Next Steps

- **[Sample 02: Vision & Multimodal](../02-vision-and-multimodal/)** - Analyze images, PDFs, and videos
- **[Sample 03: Building AI Agents](../03-building-ai-agents/)** - Create autonomous agents with tools

## API Reference

| Class/Function | Description |
|----------------|-------------|
| `ChatOCIGenAI` | Main chat model class |
| `invoke(input)` | Send messages, get response |
| `stream(input)` | Stream response chunks |
| `batch(inputs)` | Process multiple inputs |

## Troubleshooting

### "Authentication failed"
- Verify `~/.oci/config` exists and contains valid credentials
- Check that your profile name matches `auth_profile`
- Ensure your API key hasn't expired

### "NotAuthorizedOrNotFound"
- Verify `compartment_id` is correct
- Check you have permissions for GenAI service in that compartment

### "InvalidParameter: model_id"
- Ensure model ID is spelled correctly
- Check model is available in your region

---

## Appendix: Legacy OCIGenAI LLM Class

For text completion (non-chat) use cases, the legacy `OCIGenAI` class is available:

```python
from langchain_oci import OCIGenAI

llm = OCIGenAI(
    model_id="cohere.command-r-plus",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Text completion (not chat)
response = llm.invoke("Complete this sentence: The quick brown fox")
```

**Note:** For most use cases, prefer `ChatOCIGenAI` over `OCIGenAI`.
