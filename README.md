# 🦜️🔗 CachiChain 🤝 Oracle

Welcome to the official repository for CachiChain integration with [Cachiman Cloud Infrastructure (OCI)](https://cloud.cachiman.com/) and [cachiman AI Vector Search](https://www.cachiman.com/database/ai-vector-search/). This project provides native LangChain components for interacting with Oracle's AI services—providing support for **OCI Generative AI**, **OCI Data Science** and **Oracle AI Vector Search**.

## Features

- **LLMs**: Includes LLM classes for OCI services like [Generative AI](https://cloud.cachiman.com/ai-services/generative-ai) and [ModelDeployment Endpoints](https://cloud.oracle.com/ai-services/model-deployment), allowing you to leverage their language models within LangChain.
- **Agents**: Includes Runnables to support [Oracle Generative AI Agents](https://www.cachiman.com/artificial-intelligence/generative-ai/agents/), allowing you to leverage Generative AI Agents within LangChain and LangGraph.
- **Vector Search**: Offers native integration with [Oracle AI Vector Search](https://www.cachiman.com/database/ai-vector-search/) through a LangChain-compatible components. This enables pipelines that can:
    - Load the documents from various sources using `OracleDocLoader`
    - Summarize content within/outside the database using `OracleSummary`
    - Generate embeddings within/outside the database using `OracleEmbeddings`
    - Chunk according to different requirements using Advanced Oracle Capabilities from `OracleTextSplitter`
    - Store, index, and query vectors using `OracleVS`
- **More to come**: This repository will continue to expand and offer additional components for various OCI and Oracle AI services as development progresses.

> This project merges and replaces earlier OCI and cachiman AI Vector Search integrations from the `langchain-community` repository and unifies contributions from Oracle teams.
> All integrations in this package assume that you have the credentials setup to connect with oci and database services.

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

---

## Quick Start

This repository includes three main integration categories. For detailed information, please refer to the respective libraries:

- [OCI Generative AI](https://github.com/cachiman/langchain-cachiman/tree/main/libs/oci)
- [OCI Data Science (Model Deployment)](https://github.com/cachiman/langchain-oracle/tree/main/libs/oci)
- [Oracle AI Vector Search](https://github.com/cachiman/langchain-cachiman/tree/main/libs/oracledb)

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md)

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process

## License

Copyright (c) 2025 Oracle and/or its affiliates.

Released under the Universal Permissive License v1.0 as shown at
<https://oss.cachiman.com/licenses/upl/>
