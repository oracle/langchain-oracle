# langchain-oci Samples

Welcome to the langchain-oci samples! These samples will take you from beginner to expert, progressively building your skills with OCI Generative AI and LangChain.

## Learning Path

```
                           BEGINNER
                              │
                              ▼
                    ┌─────────────────┐
                    │  01. Getting    │  Authentication, ChatOCIGenAI
                    │     Started     │  First chat, provider intro
                    └────────┬────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  02. Vision &   │  Images, PDFs, video, audio
                   │   Multimodal    │
                   └────────┬────────┘
                            │
                       INTERMEDIATE
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  03. Building   │          │  04. Tool       │
    │    AI Agents    │          │     Calling     │
    └────────┬────────┘          └────────┬────────┘
             │                            │
             └─────────────┬──────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  05. Structured │
                 │     Output      │
                 └────────┬────────┘
                          │
                      ADVANCED
                          │
                          ▼
                 ┌──────────────┐
                 │ 07. Async &  │
                 │ Production   │
                 └──────────────┘
                          │
                   SPECIALIZED
                          │
              ┌───────────┴───────────┐
              ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ 09. Provider    │      │ 10. Embeddings  │
    │    Deep Dive    │      │  Text & Image   │
    └─────────────────┘      └─────────────────┘
```

## Sample Index

| # | Sample | Level | Description |
|---|----------|-------|-------------|
| 01 | [Getting Started](./01-getting-started/) | Beginner | Authentication, ChatOCIGenAI, providers |
| 02 | [Vision & Multimodal](./02-vision-and-multimodal/) | Beginner | Image analysis, PDF, video, audio |
| 03 | [Building AI Agents](./03-building-ai-agents/) | Intermediate | ReAct agents, tools, memory |
| 04 | [Tool Calling Mastery](./04-tool-calling-mastery/) | Intermediate | bind_tools, parallel calls, workflows |
| 05 | [Structured Output](./05-structured-output/) | Intermediate | Pydantic schemas, JSON modes |
| 07 | [Async for Production](./07-async-for-production/) | Advanced | ainvoke, astream, FastAPI |
| 09 | [Provider Deep Dive](./09-provider-deep-dive/) | Specialized | Meta, Gemini, Cohere, xAI |
| 10 | [Embeddings](./10-embeddings/) | Specialized | Text & image embeddings, RAG |

## Quick Start

If you're new to langchain-oci, start here:

```bash
pip install langchain-oci oci
```

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="your-compartment-id",
)

response = llm.invoke("Hello!")
print(response.content)
```

Then continue with [Sample 01: Getting Started](./01-getting-started/).

## Feature Coverage

| Feature | Sample(s) |
|---------|-------------|
| `ChatOCIGenAI` | 01, 02, 03, 04, 05, 07 |
| `OCIGenAIEmbeddings` | 10 |
| `create_oci_agent()` | 03 |
| Vision (13 models) | 02 |
| Gemini PDF/video/audio | 02 |
| Tool calling | 03, 04 |
| Parallel tool calls | 04 |
| Structured output | 05 |
| Async (ainvoke/astream/abatch) | 07 |
| Image embeddings | 10 |

## Prerequisites

All samples assume:
- Python 3.9+
- OCI CLI configured (`~/.oci/config`)
- Access to OCI Generative AI service
- A valid compartment ID

## Getting Help

- [API Reference](../docs/API_REFERENCE.md)
- [Main Documentation](../README.md)
- [GitHub Issues](https://github.com/oracle/langchain-oracle/issues)
