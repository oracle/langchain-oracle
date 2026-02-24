# Model Reference

> **Note:** OCI Generative AI models are updated frequently. For the most current and comprehensive model list, refer to the [OCI Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm).
>
> This document provides examples and guidance for using models with langchain-oci.

## Chat Models

### Meta Llama

| Model ID | Type | Features | Context |
|----------|------|----------|---------|
| `meta.llama-3.2-11b-vision-instruct` | Vision | Image analysis | 128K |
| `meta.llama-3.2-90b-vision-instruct` | Vision | Image analysis | 128K |
| `meta.llama-3.3-70b-instruct` | Text | Tools, reasoning | 128K |
| `meta.llama-4-scout-17b-16e-instruct` | Text | Parallel tools | 128K |
| `meta.llama-4-maverick-17b-128e-instruct-fp8` | Text | Parallel tools | 128K |

**Key Features:**
- Vision: Llama 3.2 11B/90B
- Parallel tool calls: Llama 4 models only
- `tool_choice` support: All models
- `tool_result_guidance`: Recommended for tool workflows

---

### Google Gemini

| Model ID | Type | Features | Context |
|----------|------|----------|---------|
| `google.gemini-2.5-flash` | Multimodal | PDF, video, audio, image | 1M |
| `google.gemini-2.5-flash` | Multimodal | PDF, video, audio, image | 1M |
| `google.gemini-2.5-pro` | Multimodal | Most capable | 1M |

**Key Features:**
- PDF processing via `application/pdf` mime type
- Video analysis via `video/mp4`, `video/webm`, etc.
- Audio transcription via `audio/mp3`, `audio/wav`, etc.
- Use `max_tokens` (not `max_output_tokens`)

**Supported Media Types:**

| Type | MIME Types |
|------|------------|
| PDF | `application/pdf` |
| Video | `video/mp4`, `video/mpeg`, `video/mov`, `video/avi`, `video/webm` |
| Audio | `audio/mp3`, `audio/wav`, `audio/aac`, `audio/ogg`, `audio/flac` |
| Image | `image/jpeg`, `image/png`, `image/gif`, `image/webp` |

---

### xAI Grok

| Model ID | Type | Features | Context |
|----------|------|----------|---------|
| `xai.grok-4` | Vision | Image, reasoning | 128K |
| `xai.grok-4-fast-reasoning` | Text | Optimized reasoning | 128K |

**Key Features:**
- `reasoning_content` in response metadata
- Vision support
- `tool_choice` support

---

### Cohere Command

| Model ID | Type | Features | Context |
|----------|------|----------|---------|
| `cohere.command-r-plus` | Text | RAG, citations | 128K |
| `cohere.command-a-03-2025` | Text | Latest release | 128K |
| `cohere.command-a-vision` | Vision | V2 API, DAC only | 128K |

**Key Features:**
- Citations in response metadata
- RAG-optimized
- V2 API for vision (requires Dedicated AI Cluster)
- No `tool_choice` support
- No parallel tool calls

---

### OpenAI (via ChatOCIOpenAI)

| Model ID | Type | Features |
|----------|------|----------|
| `openai.gpt-4.1` | Text | Tools, reasoning |
| `openai.o1` | Text | Advanced reasoning |

---

## Embedding Models

### Text Embeddings

| Model ID | Languages | Dimensions |
|----------|-----------|------------|
| `cohere.embed-english-v3.0` | English | 1024 |
| `cohere.embed-multilingual-v3.0` | 100+ languages | 1024 |

### Multimodal Embeddings

| Model ID | Types | Dimensions |
|----------|-------|------------|
| `cohere.embed-v4.0` | Text + Image | 256-1536 (configurable) |

**Usage:**

```python
from langchain_oci import OCIGenAIEmbeddings

# Text embeddings
embeddings = OCIGenAIEmbeddings(model_id="cohere.embed-english-v3.0", ...)
vector = embeddings.embed_query("Hello world")

# Image embeddings (cohere.embed-v4.0 only)
embeddings = OCIGenAIEmbeddings(model_id="cohere.embed-v4.0", ...)
vector = embeddings.embed_image("photo.jpg")
```

---

## Feature Matrix

| Feature | Meta | Gemini | Cohere | xAI |
|---------|------|--------|--------|-----|
| Text Generation | ✅ | ✅ | ✅ | ✅ |
| Vision (Images) | ✅ 3.2 | ✅ | ✅ DAC | ✅ |
| PDF Processing | ❌ | ✅ | ❌ | ❌ |
| Video Analysis | ❌ | ✅ | ❌ | ❌ |
| Audio Analysis | ❌ | ✅ | ❌ | ❌ |
| Tool Calling | ✅ | ✅ | ✅ | ✅ |
| Parallel Tools | ✅ 4+ | ❌ | ❌ | ❌ |
| `tool_choice` | ✅ | ✅ | ❌ | ✅ |
| Citations | ❌ | ❌ | ✅ | ❌ |
| Reasoning Content | ❌ | ❌ | ❌ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Async | ✅ | ✅ | ✅ | ✅ |

---

## Regions

OCI Generative AI is available in these regions:

| Region | Endpoint |
|--------|----------|
| us-chicago-1 | `https://inference.generativeai.us-chicago-1.oci.oraclecloud.com` |
| eu-frankfurt-1 | `https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com` |
| ap-tokyo-1 | `https://inference.generativeai.ap-tokyo-1.oci.oraclecloud.com` |

Check [OCI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm) for the latest region availability.

---

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model |
|----------|-------------------|
| General chat | `meta.llama-3.3-70b-instruct` |
| Image analysis | `meta.llama-3.2-90b-vision-instruct` |
| PDF/document processing | `google.gemini-2.5-flash` |
| Video understanding | `google.gemini-2.5-flash` |
| Audio transcription | `google.gemini-2.5-flash` |
| Tool-heavy workflows | `meta.llama-4-scout-17b-16e-instruct` |
| RAG with citations | `cohere.command-r-plus` |
| Complex reasoning | `xai.grok-4-fast-reasoning` |
| Fast responses | `google.gemini-2.5-flash` |
| Embeddings (text) | `cohere.embed-english-v3.0` |
| Embeddings (multimodal) | `cohere.embed-v4.0` |

### By Performance

| Priority | Model |
|----------|-------|
| Lowest latency | `google.gemini-2.5-flash` |
| Highest throughput | `google.gemini-2.5-flash` |
| Best quality | `meta.llama-3.3-70b-instruct`, `cohere.command-r-plus` |
| Best for tools | `meta.llama-4-scout-17b-16e-instruct` |
