# Sample 07: FastAPI Integration Example
# Demonstrates building a production chat API with OCI GenAI
#
# Run with: uvicorn fastapi_app:app --reload

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_oci import ChatOCIGenAI

import os

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

# Global LLM instance (reused across requests)
llm: ChatOCIGenAI = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global llm
    # Startup: Initialize the LLM
    print("Initializing OCI GenAI client...")
    llm = ChatOCIGenAI(
    auth_profile=AUTH_PROFILE,
    model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )
    print("Client initialized!")
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="OCI GenAI Chat API",
    description="Chat API powered by OCI Generative AI",
    lifespan=lifespan,
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 500


class ChatResponse(BaseModel):
    response: str


class BatchRequest(BaseModel):
    messages: list[str]


class BatchResponse(BaseModel):
    responses: list[str]


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        response = await llm.ainvoke(request.message)
        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for chunk in llm.astream(request.message):
                yield chunk.content
        except Exception as e:
            yield f"\n\nError: {e}"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )


@app.post("/chat/batch", response_model=BatchResponse)
async def chat_batch(request: BatchRequest):
    """Process multiple messages concurrently."""
    try:
        responses = await llm.abatch(request.messages)
        return BatchResponse(responses=[r.content for r in responses])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example usage when running directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
