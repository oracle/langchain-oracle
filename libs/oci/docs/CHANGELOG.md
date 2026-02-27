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
| 0.2.0 | 2025 | Vision, agents, parallel tools, embeddings |
| 0.1.0 | 2024 | Initial release |

---

## Deprecations

None currently planned.

---

## Security

See [SECURITY.md](../../../SECURITY.md) for vulnerability reporting.
