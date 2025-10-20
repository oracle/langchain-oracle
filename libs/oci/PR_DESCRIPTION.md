# Add Token Usage to AIMessage Response

## Summary
Adds `total_tokens` to `AIMessage.additional_kwargs` for non-streaming chat responses, enabling users to track token consumption when using `ChatOCIGenAI`.

## Problem
When using `ChatOCIGenAI.invoke()`, token usage information (prompt_tokens, completion_tokens, total_tokens) from the OCI Generative AI API was not accessible in the `AIMessage` response, even though the raw OCI API returns this data.

## Solution
Extract token usage from the OCI API response and add `total_tokens` to `additional_kwargs` in non-streaming mode.

### Changes Made
**File:** `langchain_oci/chat_models/oci_generative_ai.py`

1. **CohereProvider.chat_generation_info()** (lines 246-248)
   - Extract `usage.total_tokens` from `response.data.chat_response.usage`
   - Add to `generation_info["total_tokens"]`

2. **GenericProvider.chat_generation_info()** (lines 611-613)
   - Same extraction for Meta/Llama models

## Usage

### Before
```python
response = chat.invoke("What is the capital of France?")
# No way to access token usage
```

### After
```python
response = chat.invoke("What is the capital of France?")
print(response.additional_kwargs.get('total_tokens'))  # 26
```

## Limitations
- **Streaming mode:** Token usage is NOT available when `is_stream=True` because the OCI Generative AI streaming API does not include usage statistics in stream events.
- **Non-streaming only:** Use `is_stream=False` to get token usage information.

## Testing
Tested with:
- ✅ Cohere Command-R models (`cohere.command-r-plus-08-2024`)
- ✅ Meta Llama models (`meta.llama-3.3-70b-instruct`)
- ✅ Non-streaming mode (`is_stream=False`)
- ❌ Streaming mode (not supported by OCI API)

## Backward Compatibility
✅ Fully backward compatible - existing code continues to work unchanged.
