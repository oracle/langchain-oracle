# Sample 10: Text and Image Embeddings

Create vector embeddings for semantic search, RAG, and similarity applications.

## What You'll Build

By the end of this sample, you'll be able to:
- Create text embeddings with `OCIGenAIEmbeddings`
- Create image embeddings for cross-modal search
- Use embeddings for semantic similarity
- Build a basic RAG pipeline

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- Basic understanding of vector embeddings

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `OCIGenAIEmbeddings` | Main embeddings class |
| `embed_documents()` | Embed multiple texts |
| `embed_query()` | Embed a search query |
| `embed_image()` | Embed a single image |
| `embed_image_batch()` | Embed multiple images |

---

## Part 1: What Are Embeddings?

Embeddings convert text (or images) into numerical vectors that capture semantic meaning:

```
"The cat sat on the mat"  →  [0.12, -0.34, 0.56, ...]  (1536 dimensions)
"A feline rested on a rug" →  [0.11, -0.33, 0.55, ...]  (similar vector!)
```

Similar meanings = similar vectors. This enables:
- **Semantic search** - Find relevant content by meaning, not keywords
- **RAG** - Retrieve context for LLMs
- **Clustering** - Group similar items
- **Recommendations** - Find similar products/content

---

## Part 2: Creating Text Embeddings

### Basic Usage

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Embed a single query
query_vector = embeddings.embed_query("What is machine learning?")
print(f"Vector dimension: {len(query_vector)}")  # 1024

# Embed multiple documents
doc_vectors = embeddings.embed_documents([
    "Machine learning is a branch of AI.",
    "Deep learning uses neural networks.",
    "Natural language processing handles text.",
])
print(f"Number of vectors: {len(doc_vectors)}")  # 3
```

### Available Models

| Model | Dimensions | Best For |
|-------|------------|----------|
| `cohere.embed-english-v3.0` | 1024 | English text |
| `cohere.embed-multilingual-v3.0` | 1024 | Multiple languages |
| `cohere.embed-v4.0` | 256-1536 | Text + Images (multimodal) |

---

## Part 3: Input Types

Different input types optimize embeddings for specific use cases:

```python
# For documents being indexed
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    input_type="SEARCH_DOCUMENT",  # Default
)

# For search queries
embeddings_query = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    input_type="SEARCH_QUERY",
)
```

### Input Type Reference

| Input Type | Use Case |
|------------|----------|
| `SEARCH_DOCUMENT` | Documents being indexed |
| `SEARCH_QUERY` | User search queries |
| `CLASSIFICATION` | Text classification |
| `CLUSTERING` | Text clustering |
| `IMAGE` | Image inputs (use embed_image instead) |

---

## Part 4: Image Embeddings

Multimodal models like `cohere.embed-v4.0` can embed both text and images into the same vector space:

```python
from langchain_oci import OCIGenAIEmbeddings

# Use multimodal model
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Embed an image
image_vector = embeddings.embed_image("./photo.jpg")
print(f"Image vector dimensions: {len(image_vector)}")

# Embed multiple images
image_vectors = embeddings.embed_image_batch([
    "./photo1.jpg",
    "./photo2.jpg",
    "./photo3.jpg",
])
```

### Image Input Formats

```python
# From file path
vector = embeddings.embed_image("./photo.jpg")

# From bytes
with open("photo.png", "rb") as f:
    vector = embeddings.embed_image(f.read(), mime_type="image/png")

# From data URI
vector = embeddings.embed_image("data:image/png;base64,iVBORw0...")
```

---

## Part 5: Cross-Modal Search

With multimodal embeddings, you can:
- Search images using text queries
- Search text using image queries

```python
from langchain_oci import OCIGenAIEmbeddings
import numpy as np

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Index images
image_paths = ["sunset.jpg", "beach.jpg", "mountain.jpg", "city.jpg"]
image_vectors = embeddings.embed_image_batch(image_paths)

# Search with text query
query = "A beautiful sunset over the ocean"
query_vector = embeddings.embed_query(query)

# Find most similar image
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_vector, img_vec) for img_vec in image_vectors]
best_match_idx = np.argmax(similarities)
print(f"Best match: {image_paths[best_match_idx]}")  # sunset.jpg
```

---

## Part 6: Output Dimensions

For `cohere.embed-v4.0`, you can control the output dimensions:

```python
# Smaller vectors (faster, less accurate)
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
    output_dimensions=256,  # Options: 256, 512, 1024, 1536
)
```

| Dimensions | Trade-off |
|------------|-----------|
| 256 | Fastest, lowest storage, less accurate |
| 512 | Balanced |
| 1024 | Good accuracy |
| 1536 | Most accurate (default for embed-v4.0) |

---

## Part 7: RAG Pattern

Use embeddings for Retrieval Augmented Generation:

```python
from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings
from langchain_core.messages import HumanMessage
import numpy as np

# Setup
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Knowledge base
documents = [
    "Oracle Cloud Infrastructure provides AI services.",
    "OCI Generative AI supports multiple LLM providers.",
    "LangChain is a framework for LLM applications.",
]
doc_vectors = embeddings.embed_documents(documents)

# Query and retrieve
query = "What AI services does Oracle offer?"
query_vector = embeddings.embed_query(query)

# Find most relevant documents
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_vector, dv) for dv in doc_vectors]
top_indices = np.argsort(similarities)[-2:][::-1]  # Top 2
context = "\n".join([documents[i] for i in top_indices])

# Generate answer with context
prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

response = llm.invoke(prompt)
print(response.content)
```

---

## Part 8: Vector Store Integration

Use with LangChain vector stores:

```python
from langchain_community.vectorstores import FAISS
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Create vector store from documents
documents = [
    "Document 1 content...",
    "Document 2 content...",
    "Document 3 content...",
]

vectorstore = FAISS.from_texts(documents, embeddings)

# Search
results = vectorstore.similarity_search("search query", k=2)
for doc in results:
    print(doc.page_content)
```

---

## Summary

In this sample, you learned:

1. **What embeddings are** - Vector representations of meaning
2. **Text embeddings** - `embed_documents()`, `embed_query()`
3. **Image embeddings** - `embed_image()`, `embed_image_batch()`
4. **Input types** - SEARCH_DOCUMENT, SEARCH_QUERY, etc.
5. **Cross-modal search** - Text-to-image and image-to-text
6. **RAG pattern** - Retrieval Augmented Generation
7. **Vector stores** - FAISS integration

## Next Steps

- **[Sample 02: Vision & Multimodal](../02-vision-and-multimodal/)** - Image analysis
- **[Sample 07: Async for Production](../07-async-for-production/)** - Async embedding operations

## API Reference

| Method | Description |
|--------|-------------|
| `embed_query(text)` | Embed a single query |
| `embed_documents(texts)` | Embed multiple documents |
| `embed_image(image)` | Embed a single image |
| `embed_image_batch(images)` | Embed multiple images |

### Models

| Model | Type | Dimensions |
|-------|------|------------|
| `cohere.embed-english-v3.0` | Text only | 1024 |
| `cohere.embed-multilingual-v3.0` | Text only | 1024 |
| `cohere.embed-v4.0` | Text + Image | 256-1536 |

## Troubleshooting

### "Model does not support images"
- Use `cohere.embed-v4.0` for image embeddings
- Check `IMAGE_EMBEDDING_MODELS` registry

### "Embedding dimension mismatch"
- Ensure same model and `output_dimensions` for indexing and querying
- Store dimension metadata with your vectors

### "Batch too large"
- Reduce batch size (default is 96)
- Use `batch_size` parameter
