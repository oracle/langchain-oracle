# OCI Generative AI integration roadmap

This package currently provides text-chat `invoke()` and `stream()` support
for OCI Generative AI. The following work is intentionally tracked as
follow-up scope rather than a prerequisite for the initial release.

## Priority follow-ups

- [ ] Return provider response metadata and token usage consistently for both
  non-streaming and streaming responses. Preserve finish reasons for every
  supported model family.
- [ ] Add Cohere V1 tool-result round trips. Generic chat supports LangChain
  tool calls and `ToolMessage`; Cohere V1 needs a dedicated adapter because its
  tool calls have no provider-generated call IDs and its request shape separates
  the current human message from tool results.
- [ ] Support multimodal message content where the OCI model/API supports it.
- [ ] Add `OciGenAiEmbeddings` for OCI embedding models, with unit and
  authenticated integration coverage.
- [ ] Add OCI Cohere V2 API support. `OciGenAiCohereChat` currently uses the
  legacy Cohere V1 API format only.

## Ongoing quality work

- [ ] Expand authenticated integration coverage across supported OCI regions,
  model families, serving modes, and authentication providers.
- [ ] Document feature support and model/API compatibility in a matrix as V2,
  tools, multimodal input, and embeddings are added.

## Completed reliability baseline

- [x] Use `BaseChatModel` and return `ChatResult`/`AIMessage` values with OCI
  usage and response metadata.
- [x] Support Generic tool binding, tool-call responses, and `ToolMessage`
  request conversion. This also enables LangChain's standard structured-output
  flow for Generic models.
- [x] Parse SSE events independently when a network chunk contains multiple
  events; retain incomplete event data for the next chunk.
- [x] Decode streaming bytes with `TextDecoder` streaming mode so split UTF-8
  sequences are preserved.
- [x] Keep streamed and non-streamed generic text assembly consistent.
- [x] Require the final Cohere V1 input message to be human, preserving the
  preceding history order.
- [x] Support configuration-file, instance-principal, resource-principal,
  session, and custom OCI authentication providers.
- [x] Run the package unit tests, lint, format check, and build in CI.
