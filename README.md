# LangChain Oracle

[![PyPI - langchain-oci](https://img.shields.io/pypi/v/langchain-oci?label=langchain-oci)](https://pypi.org/project/langchain-oci/)
[![PyPI - langchain-oracledb](https://img.shields.io/pypi/v/langchain-oracledb?label=langchain-oracledb)](https://pypi.org/project/langchain-oracledb/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-oci)](https://pypi.org/project/langchain-oci/)
[![License](https://img.shields.io/badge/License-UPL%201.0-green)](https://opensource.org/licenses/UPL)

Official LangChain integrations for [Oracle Cloud Infrastructure (OCI)](https://cloud.oracle.com/) and [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/).

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| [**langchain-oci**](./libs/oci/) | OCI Generative AI & Data Science | `pip install langchain-oci` |
| [**langchain-oracledb**](./libs/oracledb/) | Oracle AI Vector Search | `pip install langchain-oracledb` |

---

## langchain-oci Features

Full-featured LangChain integration for OCI Generative AI services.

### Chat Models & Providers

| Provider | Models | Vision | Tool Calling |
|----------|--------|--------|--------------|
| **Meta** | Llama 3.2, 3.3, 4 | ✅ | ✅ (parallel in Llama 4) |
| **Google** | Gemini 2.0/2.5 Flash | ✅ | ✅ |
| **xAI** | Grok 4 | ✅ | ✅ |
| **Cohere** | Command R+, Command A | ✅ (V2) | ✅ |
| **OpenAI** | GPT-4, o1 | - | ✅ |

### Key Features

| Feature | Description |
|---------|-------------|
| **Vision & Multimodal** | 13+ vision models, Gemini PDF/video/audio support |
| **AI Agents** | `create_oci_agent()` with checkpointing & human-in-the-loop |
| **Tool Calling** | Parallel tools, infinite loop detection, `tool_result_guidance` |
| **Structured Output** | Pydantic schemas, `json_mode`, `json_schema` |
| **Async Support** | `ainvoke()`, `astream()`, `abatch()` |
| **Embeddings** | Text & image embeddings in same vector space |

### Quick Example

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

response = llm.invoke("Hello!")
print(response.content)
```

**[See full documentation →](./libs/oci/)**

**[Explore tutorials →](./libs/oci/tutorials/)**

---

## langchain-oracledb Features

Native integration with Oracle AI Vector Search.

| Component | Description |
|-----------|-------------|
| `OracleVS` | Vector store with similarity search |
| `OracleDocLoader` | Document loading from various sources |
| `OracleEmbeddings` | In-database embedding generation |
| `OracleTextSplitter` | Advanced text chunking |
| `OracleSummary` | In-database summarization |

### Quick Example

```python
from langchain_oracledb import OracleVS

vectorstore = OracleVS(
    client=connection,
    embedding_function=embeddings,
    table_name="my_vectors",
)

results = vectorstore.similarity_search("search query", k=5)
```

**[See full documentation →](./libs/oracledb/)**

---

## Installation

```bash
# OCI Generative AI
pip install langchain-oci oci

# Oracle AI Vector Search
pip install langchain-oracledb oracledb
```

---

## Documentation

- **[langchain-oci Documentation](./libs/oci/README.md)** - Chat models, embeddings, agents
- **[langchain-oci Tutorials](./libs/oci/tutorials/)** - Step-by-step learning path
- **[langchain-oracledb Documentation](./libs/oracledb/README.md)** - Vector search integration

---

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md).

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process.

## License

Copyright (c) 2025 Oracle and/or its affiliates.

Released under the Universal Permissive License v1.0 as shown at <https://oss.oracle.com/licenses/upl/>
