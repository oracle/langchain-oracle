# Sample 07: Async for Production

Build high-performance applications using async patterns with OCI Generative AI.

## What You'll Build

By the end of this sample, you'll be able to:
- Use `ainvoke()` for async single requests
- Use `astream()` for async streaming
- Use `abatch()` for parallel batch processing
- Build FastAPI endpoints with OCI GenAI
- Handle errors in async code

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- Basic understanding of Python async/await
- Install: `pip install fastapi uvicorn`

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `ainvoke()` | Async single request |
| `astream()` | Async streaming |
| `abatch()` | Async batch processing |
| `asyncio.gather()` | Run multiple requests concurrently |
| FastAPI integration | Production web service |

---

## Part 1: Why Async?

### The Problem with Sync

```python
# Synchronous - blocks while waiting
response1 = llm.invoke("Question 1")  # Wait ~1s
response2 = llm.invoke("Question 2")  # Wait ~1s
response3 = llm.invoke("Question 3")  # Wait ~1s
# Total: ~3 seconds
```

### The Async Solution

```python
# Asynchronous - runs concurrently
responses = await asyncio.gather(
    llm.ainvoke("Question 1"),
    llm.ainvoke("Question 2"),
    llm.ainvoke("Question 3"),
)
# Total: ~1 second (3x faster!)
```

---

## Part 2: Basic Async Operations

### Single Async Request (`ainvoke`)

```python
import asyncio
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

async def main():
    response = await llm.ainvoke("What is the capital of France?")
    print(response.content)

# Run the async function
asyncio.run(main())
```

### Async Streaming (`astream`)

```python
async def stream_response():
    async for chunk in llm.astream("Tell me a story"):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_response())
```

### Async Batch (`abatch`)

```python
async def batch_process():
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    responses = await llm.abatch(questions)

    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r.content[:100]}...\n")

asyncio.run(batch_process())
```

---

## Part 3: Concurrent Requests

### Using `asyncio.gather()`

Process multiple independent requests concurrently:

```python
import asyncio
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

async def process_concurrent():
    # All three run at the same time
    results = await asyncio.gather(
        llm.ainvoke("Explain machine learning in one sentence"),
        llm.ainvoke("Explain deep learning in one sentence"),
        llm.ainvoke("Explain neural networks in one sentence"),
    )

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.content}")

asyncio.run(process_concurrent())
```

### Handling Errors in Concurrent Requests

```python
async def safe_invoke(llm, prompt: str):
    """Wrapper that catches errors for individual requests."""
    try:
        return await llm.ainvoke(prompt)
    except Exception as e:
        return f"Error: {e}"

async def process_with_error_handling():
    prompts = ["Good prompt", "Another good prompt", ""]  # Empty will fail

    results = await asyncio.gather(
        *[safe_invoke(llm, p) for p in prompts],
        return_exceptions=True  # Don't fail all on one error
    )

    for prompt, result in zip(prompts, results):
        if isinstance(result, Exception):
            print(f"Failed: {prompt} - {result}")
        else:
            print(f"Success: {prompt}")
```

---

## Part 4: Rate Limiting and Throttling

### Semaphore for Concurrency Control

```python
import asyncio

async def process_with_limit(llm, prompts: list, max_concurrent: int = 5):
    """Process prompts with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_invoke(prompt: str):
        async with semaphore:
            return await llm.ainvoke(prompt)

    return await asyncio.gather(*[limited_invoke(p) for p in prompts])

# Process 100 prompts, but only 5 at a time
prompts = [f"Question {i}" for i in range(100)]
results = await process_with_limit(llm, prompts, max_concurrent=5)
```

---

## Part 5: Async Streaming Patterns

### Collect Full Response from Stream

```python
async def stream_to_string(llm, prompt: str) -> str:
    """Stream response and collect full text."""
    chunks = []
    async for chunk in llm.astream(prompt):
        chunks.append(chunk.content)
    return "".join(chunks)
```

### Stream with Progress

```python
async def stream_with_progress(llm, prompt: str):
    """Stream with token counting."""
    token_count = 0
    async for chunk in llm.astream(prompt):
        token_count += 1
        print(chunk.content, end="", flush=True)
    print(f"\n[Received {token_count} chunks]")
```

---

## Part 6: ChatOCIModelDeployment Async

For custom model deployments, async methods are fully supported:

```python
from langchain_oci import ChatOCIModelDeployment

# vLLM or TGI deployment
chat = ChatOCIModelDeployment(
    endpoint="https://your-deployment.oci.oraclecloud.com/predict",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
)

async def use_deployment():
    # Async invoke
    response = await chat.ainvoke("Hello!")

    # Async stream
    async for chunk in chat.astream("Tell me a story"):
        print(chunk.content, end="")
```

---

## Part 7: FastAPI Integration

### Basic FastAPI Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_oci import ChatOCIGenAI

app = FastAPI()

# Initialize model once (reuse across requests)
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    response = await llm.ainvoke(request.message)
    return ChatResponse(response=response.content)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    async def generate():
        async for chunk in llm.astream(request.message):
            yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")


# Run with: uvicorn main:app --reload
```

### Concurrent Batch Endpoint

```python
class BatchRequest(BaseModel):
    messages: list[str]


@app.post("/chat/batch")
async def chat_batch(request: BatchRequest):
    """Process multiple messages concurrently."""
    responses = await llm.abatch(request.messages)
    return {"responses": [r.content for r in responses]}
```

---

## Part 8: Best Practices

### 1. Reuse Client Instances

```python
# Good: Create once, reuse
llm = ChatOCIGenAI(...)  # Module level

async def handler():
    return await llm.ainvoke(...)

# Bad: Create new client per request
async def handler():
    llm = ChatOCIGenAI(...)  # Expensive!
    return await llm.ainvoke(...)
```

### 2. Use Timeouts

```python
import asyncio

async def with_timeout(llm, prompt: str, timeout: float = 30.0):
    try:
        return await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return "Request timed out"
```

### 3. Graceful Shutdown

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    yield
    # Shutdown
    print("Shutting down...")
    # Cleanup resources if needed

app = FastAPI(lifespan=lifespan)
```

### 4. Error Handling

```python
from fastapi import HTTPException

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await llm.ainvoke(request.message)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Summary

In this sample, you learned:

1. **Why async** - Concurrent requests for better performance
2. **`ainvoke()`** - Async single requests
3. **`astream()`** - Async streaming
4. **`abatch()`** - Async batch processing
5. **Concurrency control** - Semaphores and rate limiting
6. **FastAPI integration** - Production web services
7. **Best practices** - Reuse, timeouts, error handling

## Next Steps

- **[Sample 06: Model Deployments](../06-model-deployments/)** - Custom model endpoints
- **[Sample 10: Embeddings](../10-embeddings/)** - Async embedding operations

## API Reference

| Method | Description |
|--------|-------------|
| `ainvoke(input)` | Async single request |
| `astream(input)` | Async streaming response |
| `abatch(inputs)` | Async batch processing |
| `asyncio.gather()` | Run multiple coroutines concurrently |

## Troubleshooting

### "RuntimeError: Event loop is already running"
- Use `await` instead of `asyncio.run()` inside async contexts
- In Jupyter: `await llm.ainvoke(...)` directly

### "Too many concurrent requests"
- Implement rate limiting with semaphores
- Use `abatch()` instead of many `ainvoke()` calls

### "Connection timeout"
- Increase timeout: `asyncio.wait_for(..., timeout=60.0)`
- Check network connectivity
