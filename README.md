<div align="center">
  <h1>
    <a href="https://docs.langchain.com/oss/python/integrations/providers/oci">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/langchain-ai/langchain/master/.github/images/logo-dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/langchain-ai/langchain/master/.github/images/logo-light.svg">
        <img alt="LangChain" src="https://raw.githubusercontent.com/langchain-ai/langchain/master/.github/images/logo-dark.svg" height="40">
      </picture>
    </a>
    🤝
    <a href="https://www.oracle.com/artificial-intelligence/">
      <img alt="Oracle" src=".github/images/oracle-logo.svg" height="28">
    </a>
  </h1>

  <h3>Build agents and AI applications with LangChain, LangGraph, and Oracle AI.</h3>

  <p>
    <a href="https://pypi.org/project/langchain-oci/" target="_blank"><img src="https://img.shields.io/pypi/pyversions/langchain-oci?logo=python&logoColor=white&label=python" alt="Python versions"></a>
    <a href="https://pypistats.org/packages/langchain-oci" target="_blank"><img src="https://img.shields.io/pepy/dt/langchain-oci?label=downloads&color=blue" alt="PyPI downloads"></a>
    <a href="https://oss.oracle.com/licenses/upl/" target="_blank"><img src="https://img.shields.io/badge/license-UPL--1.0-blue" alt="License: UPL-1.0"></a>
    <a href="https://github.com/oracle/langchain-oracle/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/oracle/langchain-oracle?style=flat&logo=github&label=stars&color=blue" alt="GitHub stars"></a>
  </p>

  <p>
    <a href="#-packages">Packages</a> ·
    <a href="#-quick-start">Quick start</a> ·
    <a href="#-samples">Samples</a> ·
    <a href="#-documentation">Documentation</a> ·
    <a href="https://docs.langchain.com/oss/python/integrations/providers/oci">LangChain docs</a>
  </p>
</div>

---

This is the official repository for the LangChain integrations with [Oracle Cloud Infrastructure (OCI)](https://cloud.oracle.com/) and [Oracle Database](https://www.oracle.com/database/). It provides native LangChain and LangGraph components — in Python and JavaScript/TypeScript — for [OCI Generative AI](https://www.oracle.com/artificial-intelligence/generative-ai/generative-ai-service/), [OCI Data Science](https://www.oracle.com/artificial-intelligence/data-science/), and [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/).

## 📦 Packages

| Package | Version | Install | What it provides |
|---|---|---|---|
| [`langchain-oci`](./libs/oci) | [![PyPI](https://img.shields.io/pypi/v/langchain-oci?label=%20&logo=pypi&logoColor=white)](https://pypi.org/project/langchain-oci/) | `pip install -U langchain-oci` | **OCI Generative AI & Data Science.** Chat models (`ChatOCIGenAI`, `ChatOCIOpenAI`, `ChatOCIModelDeployment`), embeddings (`OCIGenAIEmbeddings`), agents (`create_oci_agent`, `create_deepagents_agent`), guardrails (`OCIGuardrails`), and multimodal (vision, PDF, video, audio) support across Cohere, Google, Meta, OpenAI, and xAI models on OCI. |
| [`langchain-oracledb`](./libs/oracledb) | [![PyPI](https://img.shields.io/pypi/v/langchain-oracledb?label=%20&logo=pypi&logoColor=white)](https://pypi.org/project/langchain-oracledb/) | `pip install -U langchain-oracledb` | **Oracle AI Vector Search.** Vector store (`OracleVS`), hybrid & text-search retrievers, document loaders (`OracleDocLoader`), text splitter, in-database embeddings and summaries, semantic cache, and chat message history — all powered by Oracle Database. |
| [`langgraph-oracledb`](./libs/langgraph-oracledb) | [![PyPI](https://img.shields.io/pypi/v/langgraph-oracledb?label=%20&logo=pypi&logoColor=white)](https://pypi.org/project/langgraph-oracledb/) | `pip install -U langgraph-oracledb` | **LangGraph persistence on Oracle Database.** Checkpointers (`OracleSaver`, `AsyncOracleSaver`) for durable graph state and a key/value store (`OracleStore`, `AsyncOracleStore`) with optional vector search for long-term agent memory. |
| [`@oracle/langchain-oracledb`](./libs/js/langchain-oracledb) | [![npm](https://img.shields.io/npm/v/%40oracle%2Flangchain-oracledb?label=%20&logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/@oracle/langchain-oracledb) | `npm install @oracle/langchain-oracledb` | **LangChain.js for Oracle Database.** TypeScript/JavaScript counterparts of the Vector Search components: `OracleVS`, `OracleDocLoader`, `OracleEmbeddings`, `OracleSummary`, and `OracleTextSplitter`. |

> [!NOTE]
> This project merges and replaces the earlier OCI and Oracle AI Vector Search integrations from `langchain-community`. All packages assume you have credentials configured for the OCI and/or Oracle Database services you use.

## 🚀 Quick start

### Chat with OCI Generative AI

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxxxx",
)

print(llm.invoke("Sing a ballad of LangChain.").content)
```

`ChatOCIGenAI` supports streaming, tool calling, structured output, and multimodal inputs. See the [`langchain-oci` README](./libs/oci) for on-demand vs. dedicated AI cluster models, authentication options, and OCI Data Science model deployments.

### Vector search with Oracle Database

```python
import oracledb
from langchain_oci import OCIGenAIEmbeddings
from langchain_oracledb.vectorstores import OracleVS, DistanceStrategy

conn = oracledb.connect(user="user", password="password", dsn="dsn")

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxxxx",
)

vector_store = OracleVS(conn, embeddings, "my_docs", DistanceStrategy.COSINE)
vector_store.add_texts(["Oracle AI Vector Search is built into Oracle Database."])

results = vector_store.similarity_search("What is AI Vector Search?", k=1)
```

### Durable agents with LangGraph

```python
from langgraph_oracledb.checkpoint.oracle import OracleSaver

with OracleSaver.from_conn_string("user/password@localhost:1521/FREEPDB1") as checkpointer:
    checkpointer.setup()  # create tables & apply migrations once
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "1"}})
```

## 🎓 Samples

The [`samples/`](./samples) directory is a numbered, hands-on learning path — start at 01 and work up, or jump straight to the topic you need. Every track has its own README and runnable code; these same samples are featured on the [official LangChain OCI integration page](https://docs.langchain.com/oss/python/integrations/providers/oci#samples).

| Sample | Level | Topics |
|---|---|---|
| [01: Getting Started](./samples/01-getting-started) | Beginner | Authentication, basic chat, providers |
| [02: Vision & Multimodal](./samples/02-vision-and-multimodal) | Beginner | Image analysis, PDF, video, audio |
| [03: Building AI Agents](./samples/03-building-ai-agents) | Intermediate | ReAct agents, tools, memory |
| [04: Tool Calling Mastery](./samples/04-tool-calling-mastery) | Intermediate | Parallel tools, workflows |
| [05: Structured Output](./samples/05-structured-output) | Intermediate | Pydantic schemas, JSON modes |
| [07: Async for Production](./samples/07-async-for-production) | Advanced | `ainvoke`, `astream`, FastAPI |
| [09: Provider Deep Dive](./samples/09-provider-deep-dive) | Specialized | Meta, Gemini, Cohere, xAI specifics |
| [10: Embeddings](./samples/10-embeddings) | Specialized | Text & image embeddings, RAG |
| [11: Deepagents](./samples/11-deepagents) | Specialized | Deep agents with Autonomous Database & OpenSearch datastores |

## 📖 Documentation

**Official LangChain integration pages**

- [OCI provider page](https://docs.langchain.com/oss/python/integrations/providers/oci) — installation, authentication, chat models, embeddings, model deployments, and the samples above
- [`ChatOCIGenAI`](https://docs.langchain.com/oss/python/integrations/chat/oci_generative_ai) and [`OCIGenAIEmbeddings`](https://docs.langchain.com/oss/python/integrations/text_embedding/oci_generative_ai) component guides
- [Oracle AI Vector Search provider page](https://docs.langchain.com/oss/python/integrations/providers/oracleai) — document loaders, text splitter, embeddings, summaries, and the vector store
- [Oracle AI Vector Search end-to-end RAG demo](https://github.com/langchain-ai/langchain/blob/v0.3/cookbook/oracleai_demo.ipynb)

**In this repository**

- [`libs/oci/README.md`](./libs/oci/README.md) — full `langchain-oci` guide: on-demand and dedicated AI cluster models, multimodal content, structured output, tool calling, deepagents with datastores, and OCI Data Science deployments
- [`libs/oracledb/README.md`](./libs/oracledb/README.md) — full `langchain-oracledb` guide: connecting with `python-oracledb`, `OracleVS` with chunking and indexing, loaders, splitter, embeddings, and summaries
- [`libs/langgraph-oracledb/README.md`](./libs/langgraph-oracledb/README.md) — checkpointer and store quickstarts (sync and async), plus vector-search store configuration
- [`libs/js/langchain-oracledb/README.md`](./libs/js/langchain-oracledb/README.md) — LangChain.js usage for every component, with build and test instructions
- [`samples/README.md`](./samples/README.md) — the learning path index, feature coverage matrix, and prerequisites

## 💁 Contributing

This project welcomes contributions from the community. Contributors must sign the [Oracle Contributor Agreement](https://oca.opensource.oracle.com) and sign off each commit (`git commit -s`). See the [contribution guide](./CONTRIBUTING.md) for the full workflow.

## 🔐 Security

Please report security vulnerabilities privately to Oracle following the [security guide](./SECURITY.md) — not through public GitHub issues.

## 📕 License

Copyright (c) 2025 Oracle and/or its affiliates.

Released under the [Universal Permissive License v1.0](https://oss.oracle.com/licenses/upl/). See [LICENSE.txt](./LICENSE.txt) and [THIRD_PARTY_LICENSES.txt](./THIRD_PARTY_LICENSES.txt) for details.
