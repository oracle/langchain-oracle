# API Reference

Complete API reference for langchain-oci.

## Chat Models

### ChatOCIGenAI

Main chat model for OCI Generative AI service.

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id: str,                          # Model ID or endpoint OCID
    service_endpoint: str,                  # OCI service endpoint URL
    compartment_id: str,                    # OCI compartment OCID
    provider: Optional[str] = None,         # "meta", "cohere", "google", "generic"
    auth_type: str = "API_KEY",             # "API_KEY", "INSTANCE_PRINCIPAL", etc.
    auth_profile: str = "DEFAULT",          # Profile name in ~/.oci/config
    model_kwargs: Optional[Dict] = None,    # Model parameters (temperature, etc.)
    streaming: bool = False,                # Enable streaming
)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `invoke(input)` | Synchronous generation |
| `ainvoke(input)` | Async generation |
| `stream(input)` | Synchronous streaming |
| `astream(input)` | Async streaming |
| `batch(inputs)` | Batch generation |
| `abatch(inputs)` | Async batch generation |
| `bind_tools(tools, **kwargs)` | Bind tools for function calling |
| `with_structured_output(schema)` | Get structured output |

**bind_tools Parameters:**

```python
llm.bind_tools(
    tools: List[BaseTool],                  # Tools to bind
    tool_choice: Optional[str] = None,      # "auto", "required", "none", or tool name
    parallel_tool_calls: bool = False,      # Enable parallel execution (Llama 4+)
    tool_result_guidance: bool = False,     # Guide model to use results naturally
    max_sequential_tool_calls: int = 8,     # Limit consecutive calls
)
```

**with_structured_output Parameters:**

```python
llm.with_structured_output(
    schema: Type[BaseModel],                # Pydantic model or dict
    method: str = "function_calling",       # "function_calling", "json_mode", "json_schema"
    include_raw: bool = False,              # Include raw response
)
```

---

### ChatOCIOpenAI

OpenAI Responses API compatibility for OCI.

```python
from langchain_oci import ChatOCIOpenAI

llm = ChatOCIOpenAI(
    auth: httpx.Auth,                       # OCI auth handler
    compartment_id: str,                    # OCI compartment OCID
    model: str,                             # Model name (e.g., "openai.gpt-4.1")
    conversation_store_id: Optional[str],   # For persistent memory
    region: Optional[str] = None,           # OCI region
    service_endpoint: Optional[str] = None, # Custom endpoint
    base_url: Optional[str] = None,         # Full URL override
)
```

---

### ChatOCIModelDeployment

Chat model for OCI Data Science Model Deployments.

```python
from langchain_oci import ChatOCIModelDeployment

llm = ChatOCIModelDeployment(
    endpoint: str,                          # Deployment predict URL
    model: str = "odsc-llm",                # Model name
    streaming: bool = False,                # Enable streaming
    max_retries: int = 3,                   # Retry count
    model_kwargs: Optional[Dict] = None,    # Model parameters
    default_headers: Optional[Dict] = None, # Custom headers
)
```

### ChatOCIModelDeploymentVLLM

vLLM-specific deployment parameters.

```python
from langchain_oci import ChatOCIModelDeploymentVLLM

llm = ChatOCIModelDeploymentVLLM(
    endpoint: str,
    model: str = "odsc-llm",
    # vLLM-specific
    temperature: float = 0.2,
    max_tokens: int = 256,
    top_p: float = 1.0,
    top_k: int = -1,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    use_beam_search: bool = False,
    best_of: int = 1,
    min_tokens: int = 0,
    tool_choice: Optional[str] = None,
)
```

### ChatOCIModelDeploymentTGI

TGI-specific deployment parameters.

```python
from langchain_oci import ChatOCIModelDeploymentTGI

llm = ChatOCIModelDeploymentTGI(
    endpoint: str,
    model: str = "odsc-llm",
    # TGI-specific
    temperature: float = 0.2,
    max_tokens: int = 256,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
)
```

---

## Embeddings

### OCIGenAIEmbeddings

Text and image embeddings.

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id: str,                          # Embedding model ID
    service_endpoint: str,                  # OCI service endpoint
    compartment_id: str,                    # OCI compartment OCID
    auth_type: str = "API_KEY",             # Authentication type
    auth_profile: str = "DEFAULT",          # Profile name
    truncate: str = "END",                  # Truncation strategy
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `embed_query(text)` | Embed single query |
| `embed_documents(texts)` | Embed multiple documents |
| `embed_image(path)` | Embed single image |
| `embed_image_batch(paths)` | Embed multiple images |

---

## Agents

### create_oci_agent

Factory function to create a ReAct agent.

```python
from langchain_oci import create_oci_agent

agent = create_oci_agent(
    model_id: str,                          # Model ID
    tools: List[BaseTool],                  # Tools for the agent
    compartment_id: str,                    # OCI compartment OCID
    service_endpoint: str,                  # OCI service endpoint
    system_prompt: Optional[str] = None,    # System instructions
    checkpointer: Optional[BaseCheckpointSaver] = None,  # Memory
    interrupt_before: Optional[List[str]] = None,        # Human-in-loop
    interrupt_after: Optional[List[str]] = None,
    auth_type: str = "API_KEY",
    auth_profile: str = "DEFAULT",
    model_kwargs: Optional[Dict] = None,
)
```

**Returns:** `CompiledStateGraph` (LangGraph agent)

**Usage:**

```python
from langchain_core.messages import HumanMessage

result = agent.invoke({
    "messages": [HumanMessage(content="Search for Python tutorials")]
})
```

---

## Vision Utilities

### load_image

Load image file for vision models.

```python
from langchain_oci import load_image

content_block = load_image(
    path: str,                              # Path to image file
    detail: str = "auto",                   # "auto", "low", "high"
)
# Returns: Dict with type="image_url" and base64 data
```

### encode_image

Encode bytes as image content.

```python
from langchain_oci import encode_image

content_block = encode_image(
    data: bytes,                            # Raw image bytes
    mime_type: str,                         # "image/jpeg", "image/png", etc.
    detail: str = "auto",
)
```

### to_data_uri

Convert image to data URI string.

```python
from langchain_oci import to_data_uri

uri = to_data_uri(
    path: str,                              # Path to image file
)
# Returns: "data:image/jpeg;base64,..."
```

### is_vision_model

Check if model supports vision.

```python
from langchain_oci import is_vision_model

supports_vision = is_vision_model(model_id: str)
# Returns: bool
```

---

## Constants

### VISION_MODELS

List of vision-capable model IDs.

```python
from langchain_oci import VISION_MODELS

print(VISION_MODELS)
# ['meta.llama-3.2-90b-vision-instruct', 'google.gemini-2.5-flash', ...]
```

### IMAGE_EMBEDDING_MODELS

List of models supporting image embeddings.

```python
from langchain_oci import IMAGE_EMBEDDING_MODELS

print(IMAGE_EMBEDDING_MODELS)
# ['cohere.embed-v4.0']
```

---

## Authentication Types

```python
from langchain_oci import OCIAuthType

OCIAuthType.API_KEY              # Default, uses ~/.oci/config
OCIAuthType.INSTANCE_PRINCIPAL   # For OCI Compute instances
OCIAuthType.RESOURCE_PRINCIPAL   # For OCI Functions, Jobs
OCIAuthType.SECURITY_TOKEN       # Session-based authentication
```

---

## Legacy LLM Classes

### OCIGenAI

Legacy LLM interface (text completion).

```python
from langchain_oci import OCIGenAI

llm = OCIGenAI(
    model_id: str,
    service_endpoint: str,
    compartment_id: str,
    model_kwargs: Optional[Dict] = None,
)

response = llm.invoke("Complete this: ")
```

### OCIModelDeploymentLLM

Legacy LLM for model deployments.

```python
from langchain_oci import OCIModelDeploymentLLM

llm = OCIModelDeploymentLLM(
    endpoint: str,
    model: str = "odsc-llm",
    model_kwargs: Optional[Dict] = None,
)
```
