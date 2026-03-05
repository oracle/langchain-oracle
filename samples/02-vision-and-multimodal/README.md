# Sample 02: Vision & Multimodal

Learn how to analyze images, documents, videos, and audio with OCI Generative AI vision-capable models.

## What You'll Build

By the end of this sample, you'll be able to:
- Identify vision-capable models
- Load and encode images for analysis
- Analyze single and multiple images
- Process PDFs with Gemini models
- Handle video and audio content

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- An OCI compartment with Generative AI access

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `VISION_MODELS` | Registry of vision-capable models |
| `load_image()` | Load an image file as a content block |
| `encode_image()` | Encode raw bytes as a content block |
| `to_data_uri()` | Convert image to data URI string |
| `is_vision_model()` | Check if a model supports vision |

---

## Part 1: Vision-Capable Models

Not all models can process images. Here are the vision-capable models available in OCI Generative AI:

### Model Registry

```python
from langchain_oci import VISION_MODELS

print(VISION_MODELS)
```

**Output:**
```python
[
    # Meta Llama Vision
    "meta.llama-3.2-90b-vision-instruct",
    "meta.llama-3.2-11b-vision-instruct",
    "meta.llama-4-scout-17b-16e-instruct",
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    # Google Gemini
    "google.gemini-2.5-flash",
    "google.gemini-2.5-pro",
    "google.gemini-2.5-flash-lite",
    # xAI Grok
    "xai.grok-4",
    "xai.grok-4-1-fast-reasoning",
    "xai.grok-4-1-fast-non-reasoning",
    "xai.grok-4-fast-reasoning",
    "xai.grok-4-fast-non-reasoning",
    # Cohere Command A
    "cohere.command-a-vision",
]
```

### Check If a Model Supports Vision

```python
from langchain_oci.utils.vision import is_vision_model

# Returns True
is_vision_model("meta.llama-3.2-90b-vision-instruct")

# Returns False
is_vision_model("meta.llama-3.3-70b-instruct")
```

---

## Part 2: Loading Images

### Method 1: From File Path (`load_image`)

The simplest way to use images:

```python
from langchain_oci import load_image

# Load from file path
image_block = load_image("./photo.jpg")

# Returns: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
```

### Method 2: From Bytes (`encode_image`)

For images from HTTP responses, PIL, or other sources:

```python
import requests
from langchain_oci import encode_image

# From HTTP response
response = requests.get("https://example.com/image.png")
image_block = encode_image(response.content, mime_type="image/png")

# From PIL Image
from PIL import Image
import io

pil_image = Image.open("photo.jpg")
buffer = io.BytesIO()
pil_image.save(buffer, format="PNG")
image_block = encode_image(buffer.getvalue(), mime_type="image/png")
```

### Method 3: Direct Data URI (`to_data_uri`)

For lower-level control:

```python
from langchain_oci.utils.vision import to_data_uri

# From file path
uri = to_data_uri("photo.jpg")
# "data:image/jpeg;base64,/9j/4AAQ..."

# From bytes
uri = to_data_uri(image_bytes, mime_type="image/png")
# "data:image/png;base64,iVBORw0KGgo..."

# Passthrough existing data URIs
uri = to_data_uri("data:image/png;base64,iVBORw0...")
# "data:image/png;base64,iVBORw0..."
```

---

## Part 3: Single Image Analysis

Let's analyze an image with Meta Llama Vision:

```python
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI, load_image

# Create vision-capable model
llm = ChatOCIGenAI(
    model_id="meta.llama-3.2-90b-vision-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Create message with text and image
message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image? Describe it in detail."},
        load_image("./sunset.jpg"),
    ]
)

# Get response
response = llm.invoke([message])
print(response.content)
```

**Output:**
```
The image shows a beautiful sunset over the ocean. The sky is painted
in shades of orange, pink, and purple, with wispy clouds scattered
across the horizon. The sun is partially visible, casting a warm
golden glow across the calm water...
```

---

## Part 4: Comparing Multiple Images

Vision models can analyze multiple images in one request:

```python
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI, load_image

llm = ChatOCIGenAI(
    model_id="meta.llama-3.2-90b-vision-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Compare two images
message = HumanMessage(
    content=[
        {"type": "text", "text": "Compare these two images. What are the similarities and differences?"},
        load_image("./living_room_before.jpg"),
        load_image("./living_room_after.jpg"),
    ]
)

response = llm.invoke([message])
print(response.content)
```

### Product Comparison Example

```python
# Compare product images
message = HumanMessage(
    content=[
        {"type": "text", "text": "Which laptop appears more suitable for gaming? Why?"},
        load_image("./laptop_a.jpg"),
        load_image("./laptop_b.jpg"),
    ]
)
```

---

## Part 5: Gemini Multimodal - PDF Processing

Google Gemini models can process PDFs natively:

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Load and encode PDF
with open("document.pdf", "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize this PDF document."},
        {
            "type": "media",
            "data": pdf_data,
            "mime_type": "application/pdf"
        },
    ]
)

response = llm.invoke([message])
print(response.content)
```

### PDF Use Cases

```python
# Extract key points
"Extract the main points from this contract."

# Data extraction
"Extract all dates, amounts, and party names from this invoice."

# Question answering
"According to this document, what are the payment terms?"

# Translation
"Translate this PDF from Spanish to English."
```

---

## Part 6: Video Analysis with Gemini

Gemini models can analyze video content:

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Load and encode video
with open("clip.mp4", "rb") as f:
    video_data = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe what's happening in this video."},
        {
            "type": "media",
            "data": video_data,
            "mime_type": "video/mp4"
        },
    ]
)

response = llm.invoke([message])
print(response.content)
```

### Video Analysis Use Cases

```python
# Action recognition
"What activities are shown in this video?"

# Safety analysis
"Are there any safety hazards visible in this workplace footage?"

# Content moderation
"Does this video contain any inappropriate content?"

# Event summarization
"Summarize the key moments from this meeting recording."
```

---

## Part 7: Audio Analysis with Gemini

Gemini can also transcribe and analyze audio:

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Load and encode audio
with open("recording.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe this audio and summarize the key points."},
        {
            "type": "media",
            "data": audio_data,
            "mime_type": "audio/mp3"
        },
    ]
)

response = llm.invoke([message])
print(response.content)
```

---

## Part 8: Provider-Specific Vision Support

### Meta Llama Vision

Best for: General image analysis, detailed descriptions

```python
llm = ChatOCIGenAI(
    model_id="meta.llama-3.2-90b-vision-instruct",  # or 11b for faster
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)
```

### Google Gemini

Best for: Multimodal (PDF, video, audio), complex reasoning

```python
llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)
```

### xAI Grok Vision

Best for: Fast reasoning with vision

```python
llm = ChatOCIGenAI(
    model_id="xai.grok-4",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)
```

### Cohere Command A Vision

Best for: Document understanding, RAG with images

```python
llm = ChatOCIGenAI(
    model_id="cohere.command-a-vision",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)
```

---

## Summary

In this sample, you learned:

1. **Vision models** - 13+ models that support image input
2. **Loading images** - `load_image()` for files, `encode_image()` for bytes
3. **Image analysis** - Single and multi-image analysis
4. **Gemini multimodal** - PDF, video, and audio processing
5. **Provider differences** - Choosing the right model for your use case

## Next Steps

- **[Sample 03: Building AI Agents](../03-building-ai-agents/)** - Create autonomous agents with tools
- **[Sample 10: Embeddings](../10-embeddings/)** - Image embeddings for search

## API Reference

| Function | Description |
|----------|-------------|
| `load_image(path)` | Load image file as content block |
| `encode_image(bytes, mime_type)` | Encode bytes as content block |
| `to_data_uri(image, mime_type)` | Convert to data URI string |
| `is_vision_model(model_id)` | Check if model supports vision |
| `VISION_MODELS` | List of vision-capable models |

## Troubleshooting

### "Content type not supported"
- Ensure you're using a vision-capable model
- Check the image format is supported (PNG, JPEG, GIF, WebP)

### "Image too large"
- Resize the image before encoding
- Maximum size varies by model (typically 20MB)

### "PDF not rendering"
- PDF support is Gemini-only
- Ensure the file is a valid PDF

### "Video analysis slow"
- Video analysis is computationally intensive
- Consider extracting key frames for faster processing
