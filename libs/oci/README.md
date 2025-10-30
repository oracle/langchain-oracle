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

llm = ChatOCIGenAI(
        model_id="MY_MODEL_ID",
        service_endpoint="MY_SERVICE_ENDPOINT",
        compartment_id="MY_COMPARTMENT_ID",
        model_kwargs={"max_tokens": 1024}, # Use max_completion_tokens instead of max_tokens for OpenAI models
        auth_profile="MY_AUTH_PROFILE",
        is_stream=True,
        auth_type="SECURITY_TOKEN"
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

### 4. Use Structured Output
`ChatOCIGenAI` supports structured output. 

<sub>**Note:** The default method is `function_calling`. If default method returns `None` (e.g. for Gemini models), try `json_schema` or `json_mode`.</sub>

```python
from langchain_oci import ChatOCIGenAI
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str

llm = ChatOCIGenAI()
structured_llm = llm.with_structured_output(Joke)
structured_llm.invoke("Tell me a joke about programming")
```


## OCI Data Science Model Deployment Examples

### 1. Use a Chat Model

You may instantiate the OCI Data Science model with the generic `ChatOCIModelDeployment` or framework specific class like `ChatOCIModelDeploymentVLLM`.

```python
from langchain_oci.chat_models import ChatOCIModelDeployment, ChatOCIModelDeploymentVLLM

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

chat = ChatOCIModelDeployment(
    endpoint=endpoint,
    streaming=True,
    max_retries=1,
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 512,
    },  # other model params...
    default_headers={
        "route": "/v1/chat/completions",
        # other request headers ...
    },
)
chat.invoke(messages)

chat_vllm = ChatOCIModelDeploymentVLLM(endpoint=endpoint)
chat_vllm.invoke(messages)
```

### 2. Use a Completion Model
You may instantiate the OCI Data Science model with `OCIModelDeploymentLLM` or `OCIModelDeploymentVLLM`.

```python
from langchain_oci.llms import OCIModelDeploymentLLM, OCIModelDeploymentVLLM

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri and model name with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

llm = OCIModelDeploymentLLM(
    endpoint=endpoint,
    model="odsc-llm",
)
llm.invoke("Who is the first president of United States?")

vllm = OCIModelDeploymentVLLM(
    endpoint=endpoint,
)
vllm.invoke("Who is the first president of United States?")
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
