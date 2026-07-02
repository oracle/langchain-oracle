# 🦜️🔗 LangChain 🤝 Oracle

Welcome to the official repository for LangChain integration with [Oracle Cloud Infrastructure (OCI)](https://cloud.oracle.com/) and [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/). This project provides native LangChain components for interacting with Oracle's AI services—providing support for **OCI Generative AI**, **OCI Data Science** and **Oracle AI Vector Search**.

## Features

- **LLMs**: Includes LLM classes for OCI services like [Generative AI](https://cloud.oracle.com/ai-services/generative-ai) and [ModelDeployment Endpoints](https://cloud.oracle.com/ai-services/model-deployment), allowing you to leverage their language models within LangChain.
- **Agents**: Includes Runnables to support [Oracle Generative AI Agents](https://www.oracle.com/artificial-intelligence/generative-ai/agents/), allowing you to leverage Generative AI Agents within LangChain and LangGraph.
- **Vector Search**: Offers native integration with [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/) through LangChain-compatible components. This enables pipelines that can:
    - Load the documents from various sources using `OracleDocLoader`
    - Summarize content within/outside the database using `OracleSummary`
    - Generate embeddings within/outside the database using `OracleEmbeddings`
    - Chunk according to different requirements using advanced Oracle capabilities from `OracleTextSplitter`
    - Store, index, and query vectors using `OracleVS`
- **LangGraph Persistence**: Includes JavaScript integrations for Oracle-backed LangGraph checkpointing and long-term memory stores with optional Oracle VECTOR search.
- **More to come**: This repository will continue to expand and offer additional components for various OCI and Oracle AI services as development progresses.

> This project merges and replaces earlier OCI and Oracle AI Vector Search integrations from the `langchain-community` repository and unifies contributions from Oracle teams.
> All integrations in this package assume that you have credentials set up to connect to OCI and database services.

---

## Installation

For OCI services:

```bash
python -m pip install -U langchain-oci
```

For Oracle AI Vector Search services:

```bash
python -m pip install -U langchain-oracledb
```

For LangGraph.js Oracle persistence:

```bash
npm install @oracle/langgraph-oracledb @langchain/core @langchain/langgraph-checkpoint oracledb
```

---

## Quick Start

This repository includes four main integration categories. For detailed information, please refer to the respective libraries:

- [OCI Generative AI](https://github.com/oracle/langchain-oracle/tree/main/libs/oci)
- [OCI Data Science (Model Deployment)](https://github.com/oracle/langchain-oracle/tree/main/libs/oci)
- [Oracle AI Vector Search](https://github.com/oracle/langchain-oracle/tree/main/libs/oracledb)
- [LangGraph.js Oracle Persistence](https://github.com/oracle/langchain-oracle/tree/main/libs/js/langgraph-oracledb)

## Samples

Explore comprehensive code samples covering authentication, vision/multimodal, agents, tool calling, structured output, async patterns, and embeddings:

- [Samples](https://github.com/oracle/langchain-oracle/tree/main/samples)

## Documentation

For full documentation, see the official LangChain integration pages:

- [OCI Generative AI Provider](https://python.langchain.com/docs/integrations/providers/oci/)
- [ChatOCIGenAI](https://python.langchain.com/docs/integrations/chat/oci_generative_ai/)
- [OCIGenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/oci_generative_ai/)

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md)

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process

## License

Copyright (c) 2025 Oracle and/or its affiliates.

Released under the Universal Permissive License v1.0 as shown at
<https://oss.oracle.com/licenses/upl/>
