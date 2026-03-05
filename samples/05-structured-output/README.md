# Sample 05: Structured Output

Get predictable, typed responses from language models using schemas.

## What You'll Build

By the end of this sample, you'll be able to:
- Use `with_structured_output()` for typed responses
- Define schemas with Pydantic models
- Use JSON mode for flexible output
- Handle validation errors gracefully
- Build real-world data extraction pipelines

## Prerequisites

- Completed [Sample 01: Getting Started](../01-getting-started/)
- Completed [Sample 04: Tool Calling Mastery](../04-tool-calling-mastery/) (recommended)

## Concepts Covered

| Concept | Description |
|---------|-------------|
| `with_structured_output()` | Get typed responses |
| Pydantic schemas | Define output structure |
| `json_mode` | Flexible JSON output |
| `json_schema` | JSON Schema-based output |
| `include_raw` | Access raw response |

---

## Part 1: Why Structured Output?

Without structured output:
```python
response = llm.invoke("Extract the name and email from: John Doe john@example.com")
# Output: "The name is John Doe and the email is john@example.com"
# Hard to parse programmatically!
```

With structured output:
```python
response = structured_llm.invoke("Extract: John Doe john@example.com")
# Output: Contact(name="John Doe", email="john@example.com")
# Directly usable in code!
```

---

## Part 2: Using Pydantic Schemas

### Define Your Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Contact(BaseModel):
    """A contact with name and email."""
    name: str = Field(description="The person's full name")
    email: str = Field(description="The email address")
    phone: Optional[str] = Field(default=None, description="Phone number if available")
```

### Create Structured Model

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxx",
)

# Create structured version
structured_llm = llm.with_structured_output(Contact)
```

### Get Typed Responses

```python
result = structured_llm.invoke(
    "Extract contact info: John Doe, john.doe@example.com, 555-123-4567"
)

print(type(result))    # <class 'Contact'>
print(result.name)     # "John Doe"
print(result.email)    # "john.doe@example.com"
print(result.phone)    # "555-123-4567"
```

---

## Part 3: Complex Schemas

### Nested Structures

```python
from pydantic import BaseModel, Field
from typing import List

class Address(BaseModel):
    """A physical address."""
    street: str
    city: str
    state: str
    zip_code: str

class Company(BaseModel):
    """A company with employees."""
    name: str
    industry: str
    headquarters: Address
    employee_count: int

# Works with nested structures
structured_llm = llm.with_structured_output(Company)

result = structured_llm.invoke("""
    Extract company info: Acme Corp is a technology company based at
    123 Tech Blvd, San Francisco, CA 94102. They have about 500 employees.
""")

print(result.name)                    # "Acme Corp"
print(result.headquarters.city)       # "San Francisco"
print(result.employee_count)          # 500
```

### Lists and Enums

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Review(BaseModel):
    """A product review analysis."""
    summary: str = Field(description="Brief summary of the review")
    sentiment: Sentiment = Field(description="Overall sentiment")
    keywords: List[str] = Field(description="Key topics mentioned")
    rating: int = Field(ge=1, le=5, description="Rating 1-5")

structured_llm = llm.with_structured_output(Review)

result = structured_llm.invoke("""
    Analyze this review: "Great product! The battery life is amazing
    and the camera quality exceeded my expectations. Highly recommend!"
""")

print(result.sentiment)    # Sentiment.POSITIVE
print(result.keywords)     # ["battery life", "camera quality"]
print(result.rating)       # 5
```

---

## Part 4: Output Methods

### Method 1: Function Calling (Default)

Uses tool calling under the hood. Most reliable.

```python
structured_llm = llm.with_structured_output(
    Contact,
    method="function_calling",  # Default
)
```

### Method 2: JSON Mode

Returns raw JSON, parsed by Pydantic:

```python
structured_llm = llm.with_structured_output(
    Contact,
    method="json_mode",
)
```

### Method 3: JSON Schema

Uses OCI's native JSON schema support:

```python
structured_llm = llm.with_structured_output(
    Contact,
    method="json_schema",
)
```

### When to Use Each

| Method | Best For | Notes |
|--------|----------|-------|
| `function_calling` | Most use cases | Default, most reliable |
| `json_mode` | Simple schemas | Faster, less validation |
| `json_schema` | Complex schemas | Native OCI support |

---

## Part 5: Include Raw Response

Access both the parsed result and raw AI response:

```python
structured_llm = llm.with_structured_output(
    Contact,
    include_raw=True,
)

response = structured_llm.invoke("Extract: John Doe john@example.com")

# Response is a dict with both
print(response["parsed"])     # Contact(name="John Doe", email="john@example.com")
print(response["raw"])        # AIMessage with raw content
```

Useful for:
- Debugging
- Logging
- Accessing additional metadata

---

## Part 6: Error Handling

### Validation Errors

```python
from pydantic import ValidationError

try:
    result = structured_llm.invoke("This text has no contact info")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Handle gracefully
```

### Robust Extraction Pattern

```python
from typing import Optional
from pydantic import BaseModel, Field

class ExtractionResult(BaseModel):
    """Wrapper for extraction with confidence."""
    data: Optional[Contact] = Field(default=None)
    confidence: float = Field(ge=0, le=1, description="Extraction confidence")
    notes: str = Field(default="", description="Any issues or notes")

structured_llm = llm.with_structured_output(ExtractionResult)

result = structured_llm.invoke("Maybe John? Not sure about email")

if result.confidence > 0.8:
    process(result.data)
else:
    flag_for_review(result)
```

---

## Part 7: Real-World Examples

### Data Extraction from Documents

```python
class Invoice(BaseModel):
    """Extracted invoice data."""
    invoice_number: str
    date: str
    vendor_name: str
    total_amount: float
    line_items: List[LineItem]

structured_llm = llm.with_structured_output(Invoice)

# Extract from invoice text
invoice_data = structured_llm.invoke(invoice_text)
```

### Classification

```python
class Classification(BaseModel):
    """Document classification."""
    category: str = Field(description="Document category")
    subcategory: str = Field(description="Specific subcategory")
    confidence: float
    tags: List[str]

structured_llm = llm.with_structured_output(Classification)

result = structured_llm.invoke(f"Classify this document: {document_text}")
```

### Entity Extraction

```python
class Entities(BaseModel):
    """Named entities from text."""
    people: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)

structured_llm = llm.with_structured_output(Entities)

entities = structured_llm.invoke(article_text)
print(f"People mentioned: {entities.people}")
```

---

## Part 8: Best Practices

### Schema Design

1. **Clear descriptions** - Help the model understand each field
2. **Use Optional** - For fields that might not exist
3. **Add constraints** - `ge`, `le`, `min_length`, `max_length`
4. **Use enums** - For categorical fields

### Example: Well-Designed Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task(BaseModel):
    """A task extracted from text."""
    title: str = Field(
        min_length=1,
        max_length=200,
        description="Brief task title"
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed description if available"
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Task priority level"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Due date in ISO format (YYYY-MM-DD)"
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Person assigned to the task"
    )
```

---

## Summary

In this sample, you learned:

1. **Why structured output** - Predictable, typed responses
2. **Pydantic schemas** - Define output structure
3. **`with_structured_output()`** - Create structured models
4. **Output methods** - function_calling, json_mode, json_schema
5. **Error handling** - Validation and confidence
6. **Real-world patterns** - Extraction, classification, entities

## Next Steps

- **[Sample 07: Async for Production](../07-async-for-production/)** - Async structured extraction
- **[Sample 10: Embeddings](../10-embeddings/)** - Semantic search with extracted data

## API Reference

| Method/Parameter | Description |
|------------------|-------------|
| `with_structured_output(schema)` | Create structured model |
| `method` | "function_calling", "json_mode", "json_schema" |
| `include_raw` | Include raw response in output |
| Pydantic `Field()` | Add descriptions and constraints |

## Troubleshooting

### "Validation failed"
- Check if all required fields are extractable from input
- Use `Optional` for fields that might not exist
- Add `default=None` or `default_factory=list`

### "Wrong type returned"
- Verify Pydantic schema is correct
- Check `Field(description=...)` is clear
- Try `method="function_calling"` for better reliability

### "Incomplete extraction"
- Make input text clearer
- Add examples in the prompt
- Use `include_raw=True` to debug
