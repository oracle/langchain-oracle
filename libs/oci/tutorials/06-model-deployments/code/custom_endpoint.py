# Tutorial 06: Custom Endpoint Example
# Demonstrates extending ChatOCIModelDeployment for custom inference formats

import ads
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_oci import ChatOCIModelDeployment

# Configure authentication
ads.set_auth("api_key")


class CustomFormatDeployment(ChatOCIModelDeployment):
    """Custom deployment handler for non-standard response formats.

    This example shows how to extend the base class to handle
    custom request/response formats from your model deployment.
    """

    custom_param: str = "default_value"  # Add custom parameters

    def _construct_json_body(self, messages: list, params: dict) -> dict:
        """Construct custom request payload.

        Override this method if your endpoint expects a different
        request format than the standard OpenAI chat completions format.
        """
        # Convert LangChain messages to custom format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "content"):
                formatted_messages.append(
                    {
                        "role": msg.type,  # "human", "ai", "system"
                        "text": msg.content,
                    }
                )

        return {
            "conversation": formatted_messages,
            "config": {
                "max_tokens": params.get("max_tokens", 512),
                "temperature": params.get("temperature", 0.7),
                "custom_param": self.custom_param,
            },
        }

    def _process_response(self, response_json: dict) -> ChatResult:
        """Process custom response format.

        Override this method if your endpoint returns a different
        response format than the standard OpenAI chat completions format.
        """
        # Example custom response format:
        # {
        #     "output": {
        #         "generated_text": "...",
        #         "tokens_used": 100
        #     },
        #     "status": "success"
        # }

        output = response_json.get("output", {})
        text = output.get("generated_text", "")
        tokens = output.get("tokens_used", 0)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=text),
                    generation_info={
                        "tokens_used": tokens,
                        "custom_format": True,
                    },
                )
            ],
            llm_output={
                "model_name": self.model,
                "token_usage": {"total_tokens": tokens},
            },
        )


class BatchInferenceDeployment(ChatOCIModelDeployment):
    """Custom deployment for batch inference endpoints.

    Some deployments support batch processing of multiple prompts
    in a single request for efficiency.
    """

    batch_size: int = 5

    def _construct_json_body(self, messages: list, params: dict) -> dict:
        """Construct batch request payload."""
        # Extract user messages for batch processing
        prompts = []
        for msg in messages:
            if hasattr(msg, "content"):
                prompts.append(msg.content)

        return {
            "prompts": prompts,
            "batch_config": {
                "max_batch_size": self.batch_size,
                "return_all": True,
            },
            **params,
        }


def demo_custom_deployment():
    """Demonstrate custom deployment usage."""
    print("Custom Deployment Example")
    print("=" * 50)

    # Replace with your endpoint
    endpoint = "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict"

    chat = CustomFormatDeployment(
        endpoint=endpoint,
        model="my-custom-model",
        custom_param="special_value",
        model_kwargs={
            "max_tokens": 256,
            "temperature": 0.5,
        },
    )

    print("Custom deployment configured:")
    print(f"  - Endpoint: {endpoint}")
    print(f"  - Model: {chat.model}")
    print(f"  - Custom param: {chat.custom_param}")

    # Uncomment to actually invoke (requires real endpoint):
    # response = chat.invoke("Hello, custom model!")
    # print(f"Response: {response.content}")


def show_extension_patterns():
    """Show common extension patterns."""
    print("\nCommon Extension Patterns")
    print("=" * 50)

    patterns = """
    1. Custom Request Format:
       Override _construct_json_body() to change how messages
       are formatted in the HTTP request body.

    2. Custom Response Parsing:
       Override _process_response() to parse non-standard
       response formats from your model.

    3. Custom Streaming:
       Override _process_stream_response() for custom
       streaming response formats.

    4. Custom Headers:
       Use default_headers parameter or override _headers()
       for custom HTTP headers.

    5. Custom Parameters:
       Add Pydantic fields to your subclass for
       deployment-specific configuration.
    """
    print(patterns)


if __name__ == "__main__":
    print("Custom Endpoint Examples")
    print("Demonstrates extending ChatOCIModelDeployment\n")

    demo_custom_deployment()
    show_extension_patterns()
