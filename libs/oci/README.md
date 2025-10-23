# langchain-oci

This package contains the LangChain integrations with oci.

## Installation

```bash
pip install -U langchain-oci
```
All integrations in this package assume that you have the credentials setup to connect with oci services.

---

## Quick Start

This repository includes two main integration categories:

- [OCI Generative AI](#oci-generative-ai-examples)
- [OCI Data Science (Model Deployment)](#oci-data-science-model-deployment-examples)


---

## OCI Generative AI Examples

### 1. Use a Chat Model

`ChatOCIGenAI` class exposes chat models from OCI Generative AI.

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI()
llm.invoke("Sing a ballad of LangChain.")
```

### 2. Use a Completion Model
`OCIGenAI` class exposes LLMs from OCI Generative AI.

```python
from langchain_oci import OCIGenAI

llm = OCIGenAI()
llm.invoke("The meaning of life is")
```

### 3. Use an Embedding Model
`OCIGenAIEmbeddings` class exposes embeddings from OCI Generative AI.

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```


## OCI Data Science Model Deployment Examples

### 1. Use a Chat Model

The `ChatOCIModelDeployment` class is designed for deployment with OpenAI compatible APIs.

```python
from langchain_oci import ChatOCIModelDeployment

# Create an instance of OCI Model Deployment Endpoint

# Replace the endpoint uri with your own
# For streaming, use the /predictWithResponseStream endpoint.
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predictWithResponseStream"

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

chat = ChatOCIModelDeployment(
    endpoint=endpoint,
    model="odsc-llm",
    max_tokens=512,
)
chat.stream(messages)

```

### 2. Use a Completion Model
The `OCIModelDeploymentLLM` class is designed for completion endpoints.

```python
from langchain_oci import OCIModelDeploymentLLM

# Create an instance of OCI Model Deployment Endpoint

# Replace the endpoint uri and model name with your own
# For streaming, use the /predictWithResponseStream endpoint.
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

llm = OCIModelDeploymentLLM(
    endpoint=endpoint,
    model="odsc-llm",
)
llm.invoke("Who is the first president of United States?")

```

### 3. Use an Embedding Model
You may instantiate the OCI Data Science model with the `OCIModelDeploymentEndpointEmbeddings`.

```python
from langchain_oci.embeddings import OCIModelDeploymentEndpointEmbeddings

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

embeddings = OCIModelDeploymentEndpointEmbeddings(
    endpoint=endpoint,
)

query = "Hello World!"
embeddings.embed_query(query)

documents = ["This is a sample document", "and here is another one"]
embeddings.embed_documents(documents)
```
