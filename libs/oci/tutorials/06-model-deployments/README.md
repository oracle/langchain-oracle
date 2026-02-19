# Tutorial 06: Custom Model Deployments

Deploy and use custom models on OCI Data Science Model Deployments with LangChain.

## What You'll Learn

- Deploy custom models using OCI Data Science
- Use `ChatOCIModelDeployment` for chat interfaces
- Configure vLLM and TGI deployments
- Handle authentication with `oracle-ads`
- Enable streaming and async operations

## Prerequisites

- Completed [Tutorial 01: Getting Started](../01-getting-started/)
- OCI Data Science Model Deployment endpoint
- `oracle-ads` and `langchain-openai` installed

```bash
pip install oracle-ads langchain-openai langchain-oci
```

## Concepts Covered

| Class | Description |
|-------|-------------|
| `ChatOCIModelDeployment` | Base chat model for OCI deployments |
| `ChatOCIModelDeploymentVLLM` | vLLM-specific deployment |
| `ChatOCIModelDeploymentTGI` | Text Generation Inference deployment |
| `OCIModelDeploymentLLM` | Legacy LLM interface |

---

## Part 1: Understanding OCI Model Deployments

OCI Data Science Model Deployments allow you to deploy custom models (fine-tuned LLMs, open-source models) on dedicated infrastructure. Unlike the managed GenAI service, you have full control over:

- Model selection (any Hugging Face model)
- Infrastructure sizing (GPU types, memory)
- Inference framework (vLLM, TGI, custom)

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  OCI Data Science                        │
│                                                          │
│   ┌──────────────┐    ┌──────────────┐                  │
│   │   vLLM       │    │     TGI      │                  │
│   │  Deployment  │    │  Deployment  │                  │
│   └──────┬───────┘    └──────┬───────┘                  │
│          │                   │                           │
│          └─────────┬─────────┘                          │
│                    │                                     │
│         ┌──────────┴──────────┐                         │
│         │  /v1/chat/completions│                         │
│         │    (OpenAI-compatible)│                        │
│         └──────────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

---

## Part 2: Authentication Setup

OCI Model Deployments use `oracle-ads` for authentication.

### Configure ADS Authentication

```python
import ads

# Option 1: API Key (default, uses ~/.oci/config)
ads.set_auth("api_key")

# Option 2: Resource Principal (for OCI Functions, Jobs)
ads.set_auth("resource_principal")

# Option 3: Instance Principal (for Compute instances)
ads.set_auth("instance_principal")

# Option 4: Security Token (session-based)
ads.set_auth("security_token")
```

### Verify Authentication

```python
# Check current auth method
print(ads.common.auth.default_signer())
```

---

## Part 3: Basic ChatOCIModelDeployment

The base class works with any OpenAI-compatible endpoint.

```python
from langchain_oci import ChatOCIModelDeployment

# Create chat model pointing to your deployment
chat = ChatOCIModelDeployment(
    endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
    model="odsc-llm",  # Default model name for AQUA deployments
    streaming=True,
    model_kwargs={
        "max_tokens": 512,
        "temperature": 0.2,
    },
)

# Simple invocation
response = chat.invoke("What is machine learning?")
print(response.content)
```

### With Custom Headers

```python
chat = ChatOCIModelDeployment(
    endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
    model="odsc-llm",
    default_headers={
        "route": "/v1/chat/completions",
        "X-Custom-Header": "value",
    },
)
```

### Message Formats

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantum computing in simple terms."),
]

response = chat.invoke(messages)
```

---

## Part 4: vLLM Deployments

vLLM is optimized for high-throughput LLM inference with PagedAttention.

```python
from langchain_oci import ChatOCIModelDeploymentVLLM

chat = ChatOCIModelDeploymentVLLM(
    endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
    model="odsc-llm",

    # vLLM-specific parameters
    temperature=0.2,
    max_tokens=512,
    top_p=0.95,
    top_k=40,

    # Sampling parameters
    frequency_penalty=0.1,
    presence_penalty=0.1,
    repetition_penalty=1.1,

    # Beam search (optional)
    use_beam_search=False,
    best_of=1,

    # Token control
    min_tokens=10,
    ignore_eos=False,
    skip_special_tokens=True,
)

response = chat.invoke("Write a haiku about coding.")
print(response.content)
```

### vLLM Tool Calling

If your vLLM deployment is configured with `--tool-call-parser`:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72F, sunny"

chat = ChatOCIModelDeploymentVLLM(
    endpoint="...",
    tool_choice="auto",  # Enable tool calling
)

chat_with_tools = chat.bind_tools([get_weather])
response = chat_with_tools.invoke("What's the weather in Chicago?")
```

### Custom Chat Templates

```python
chat = ChatOCIModelDeploymentVLLM(
    endpoint="...",
    chat_template="{% for message in messages %}...",  # Jinja2 template
)
```

---

## Part 5: TGI Deployments

Text Generation Inference (TGI) from Hugging Face offers production-ready serving.

```python
from langchain_oci import ChatOCIModelDeploymentTGI

chat = ChatOCIModelDeploymentTGI(
    endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
    model="odsc-llm",

    # TGI parameters
    temperature=0.2,
    max_tokens=512,
    top_p=0.9,

    # Reproducibility
    seed=42,

    # Penalties
    frequency_penalty=0.0,
    presence_penalty=0.0,

    # Log probabilities
    logprobs=True,
    top_logprobs=5,
)

response = chat.invoke("Explain the theory of relativity.")
print(response.content)
```

---

## Part 6: Streaming Responses

All deployment classes support streaming.

### Synchronous Streaming

```python
chat = ChatOCIModelDeployment(
    endpoint="...",
    streaming=True,
)

for chunk in chat.stream("Tell me a story about a robot."):
    print(chunk.content, end="", flush=True)
```

### Async Streaming

```python
import asyncio

async def stream_response():
    chat = ChatOCIModelDeployment(
        endpoint="...",
        streaming=True,
    )

    async for chunk in chat.astream("Tell me a story about a robot."):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_response())
```

---

## Part 7: Async Operations

Full async support for high-concurrency applications.

```python
import asyncio
from langchain_oci import ChatOCIModelDeployment

async def main():
    chat = ChatOCIModelDeployment(endpoint="...", model="odsc-llm")

    # Single async request
    response = await chat.ainvoke("Hello!")
    print(response.content)

    # Concurrent requests
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    tasks = [chat.ainvoke(q) for q in questions]
    responses = await asyncio.gather(*tasks)

    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r.content[:100]}...")

asyncio.run(main())
```

---

## Part 8: Structured Output

Use JSON mode for structured responses.

```python
from pydantic import BaseModel, Field
from langchain_oci import ChatOCIModelDeployment

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief review summary")

chat = ChatOCIModelDeployment(endpoint="...", model="odsc-llm")

# Use JSON mode
structured_chat = chat.with_structured_output(MovieReview, method="json_mode")

response = structured_chat.invoke(
    "Review the movie 'Inception'. Respond in JSON with "
    "title, rating (1-10), and summary fields."
)

print(f"Title: {response.title}")
print(f"Rating: {response.rating}/10")
print(f"Summary: {response.summary}")
```

---

## Part 9: Custom Endpoint Handling

Extend the base class for custom inference endpoints.

```python
from langchain_oci import ChatOCIModelDeployment
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage

class MyCustomDeployment(ChatOCIModelDeployment):
    """Custom deployment with non-standard response format."""

    def _construct_json_body(self, messages: list, params: dict) -> dict:
        """Custom request payload."""
        return {
            "inputs": messages,
            "parameters": params,
            "custom_field": "value",
        }

    def _process_response(self, response_json: dict) -> ChatResult:
        """Custom response parsing."""
        # Extract text from custom response format
        text = response_json.get("output", {}).get("generated_text", "")

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info={"custom": True},
                )
            ]
        )

# Use custom deployment
chat = MyCustomDeployment(endpoint="...", model="my-model")
response = chat.invoke("Hello!")
```

---

## Part 10: Legacy LLM Interface

For text completion (non-chat) workloads.

```python
from langchain_oci import OCIModelDeploymentLLM

llm = OCIModelDeploymentLLM(
    endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
    model="odsc-llm",
    streaming=True,
    model_kwargs={
        "max_tokens": 256,
        "temperature": 0.7,
    },
)

# Text completion
response = llm.invoke("Complete this sentence: The future of AI is")
print(response)

# Streaming
for chunk in llm.stream("Write a poem about:"):
    print(chunk, end="", flush=True)
```

---

## Summary

You learned how to:

- Configure authentication with `oracle-ads`
- Use `ChatOCIModelDeployment` for any OpenAI-compatible endpoint
- Configure vLLM deployments with `ChatOCIModelDeploymentVLLM`
- Configure TGI deployments with `ChatOCIModelDeploymentTGI`
- Stream responses synchronously and asynchronously
- Get structured output with JSON mode
- Extend base classes for custom endpoints

## Next Steps

- [Tutorial 07: Async for Production](../07-async-for-production/) - Scale with async patterns
- [Tutorial 08: OpenAI Responses API](../08-openai-responses-api/) - Use ChatOCIOpenAI

## API Reference

| Class | Description |
|-------|-------------|
| `ChatOCIModelDeployment` | Base class for OCI model deployments |
| `ChatOCIModelDeploymentVLLM` | vLLM-specific parameters |
| `ChatOCIModelDeploymentTGI` | TGI-specific parameters |
| `OCIModelDeploymentLLM` | Text completion interface |
| `OCIModelDeploymentVLLM` | vLLM text completion |
| `OCIModelDeploymentTGI` | TGI text completion |

## Troubleshooting

### Authentication Errors

```
Could not authenticate
```
- Run `ads.set_auth("api_key")` with correct profile
- Verify `~/.oci/config` is properly configured
- Check IAM policies for model deployment access

### Connection Timeout

```
ConnectTimeout
```
- Verify endpoint URL is correct
- Check network connectivity to OCI
- Ensure deployment is in ACTIVE state

### 401 Unauthorized

```
TokenExpiredError
```
- Refresh security token: `oci session authenticate`
- Check resource principal permissions
- Verify compartment access policies

### Model Not Found

```
Model 'xxx' not found
```
- Use `"odsc-llm"` for AQUA deployments
- Check model name matches deployment configuration
