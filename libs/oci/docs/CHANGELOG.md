# Changelog

All notable changes to langchain-oci are documented here.

## [Unreleased]

### Added
- Comprehensive tutorial suite (10 tutorials)
- API reference documentation
- Model reference guide

---

## [0.2.0] - 2025

### Added

#### Vision & Multimodal
- Vision support for 13 models via `load_image()`, `encode_image()`, `to_data_uri()`
- `VISION_MODELS` registry for vision-capable model discovery
- `is_vision_model()` utility function
- Gemini multimodal support (PDF, video, audio)
- Cohere V2 API for vision models (DAC deployments)

#### Agents
- `create_oci_agent()` factory function for ReAct agents
- LangGraph integration with checkpointing
- Human-in-the-loop support via `interrupt_before`/`interrupt_after`

#### Tool Calling
- Parallel tool calls for Llama 4+ models (`parallel_tool_calls=True`)
- `tool_result_guidance` to help models synthesize tool results
- `max_sequential_tool_calls` for infinite loop prevention
- Intelligent `tool_choice` management

#### Embeddings
- Image embeddings via `embed_image()` and `embed_image_batch()`
- `IMAGE_EMBEDDING_MODELS` registry
- Support for `cohere.embed-v4.0` multimodal embeddings

#### Async Support
- Full async support via LangChain base classes
- `ainvoke()`, `astream()`, `abatch()` methods
- Async support for `ChatOCIModelDeployment`

#### Providers
- `GeminiProvider` with `max_output_tokens` → `max_tokens` mapping
- Enhanced `CohereProvider` with V2 API support
- `XAIProvider` with reasoning content extraction

### Changed
- Improved error handling for tool calling
- Better message format validation
- Enhanced streaming reliability

### Fixed
- Issue #28: Meta models outputting raw JSON instead of natural responses
- Issue #78: NullPointerException with empty AI message content
- Infinite loop detection for repeated tool calls

---

## [0.1.0] - 2024

### Added
- Initial release
- `ChatOCIGenAI` for OCI Generative AI chat models
- `ChatOCIOpenAI` for OpenAI Responses API compatibility
- `ChatOCIModelDeployment` for OCI Data Science deployments
- `ChatOCIModelDeploymentVLLM` for vLLM deployments
- `ChatOCIModelDeploymentTGI` for TGI deployments
- `OCIGenAIEmbeddings` for text embeddings
- `OCIModelDeploymentEndpointEmbeddings` for deployment embeddings
- Support for Meta, Cohere, Google, xAI, OpenAI, Mistral providers
- Four authentication methods (API Key, Instance Principal, Resource Principal, Security Token)
- Structured output via `with_structured_output()`
- Basic tool calling via `bind_tools()`

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.0 | 2025 | Vision, agents, parallel tools, async |
| 0.1.0 | 2024 | Initial release |

---

## Migration Guide

### From 0.1.x to 0.2.x

#### Vision Support

```python
# Old: No vision support
# New: Use load_image()
from langchain_oci import ChatOCIGenAI, load_image
from langchain_core.messages import HumanMessage

llm = ChatOCIGenAI(model_id="meta.llama-3.2-90b-vision-instruct", ...)

message = HumanMessage(content=[
    {"type": "text", "text": "Describe this image."},
    load_image("photo.jpg"),
])
response = llm.invoke([message])
```

#### Agents

```python
# Old: Manual agent setup
# New: Use create_oci_agent()
from langchain_oci import create_oci_agent

agent = create_oci_agent(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    tools=[my_tool],
    compartment_id="...",
    service_endpoint="...",
)
```

#### Parallel Tools

```python
# Old: Sequential tool calls only
# New: Enable parallel calls (Llama 4+)
llm_with_tools = llm.bind_tools(
    [tool1, tool2],
    parallel_tool_calls=True,
)
```

---

## Deprecations

None currently planned.

---

## Security

See [SECURITY.md](../../../SECURITY.md) for vulnerability reporting.
