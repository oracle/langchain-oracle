# @oracle/langchain-oci

Oracle Cloud Infrastructure (OCI) Generative AI is a fully managed service that provides a set of state-of-the-art, customizable large language models (LLMs) that cover a wide range of use cases and is available through a single API. Using the OCI Generative AI service, you can access ready-to-use pretrained models or create and host your own fine-tuned custom models based on your own data on dedicated AI clusters.

Detailed documentation of the OCI Generative AI service and API is available [here](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) and [here](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/).

This package enables you to use OCI Generative AI in your LangChain.js applications.

## Features

This package provides LangChain.js integrations for OCI Generative AI:

- `OciGenAiGenericChat` for models using OCI's Generic chat API
- `OciGenAiCohereChat` for legacy Cohere V1 chat models
- `OciGenAiEmbeddings` for OCI Generative AI text embeddings

`OciGenAiGenericChat` supports text chat, streaming, token usage and finish metadata, LangChain tool binding, and tool-message turns. This enables the standard LangChain structured-output flow for OCI Generic models.

> [!NOTE]
>
> `OciGenAiCohereChat` uses OCI's legacy Cohere V1 API format. Current Cohere chat models that require the V2 API are not supported by this class, and tool-result round trips are not supported for the legacy Cohere integration.

## Prerequisites

In order to use this integration you will need the following:

1. An OCI tenancy. If you do not already have an account, please create one [here](https://signup.cloud.oracle.com?sourceType=:ex:of:::::LangChainJSIntegration&SC=:ex:of:::::LangChainJSIntegration&pcode=).

2. An OCI authentication method. Using a [configuration file](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm) with [API Key authentication](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm#apisigningkey_topic_How_to_Generate_an_API_Signing_Key_Console) is the simplest option for local development. Other authentication options are described below.

3. A tenancy registered in one of the [supported regions](https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm#regions).

4. The OCID of a compartment in which your OCI user has [permission to use the Generative AI service](https://docs.oracle.com/en-us/iaas/Content/generative-ai/iam-policies.htm). You can use the `root` compartment or [create your own compartment](https://docs.oracle.com/en-us/iaas/Content/Identity/compartments/To_create_a_compartment.htm).

5. A model available in your selected region. See the [available models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) and make sure you select a model that is not deprecated.

6. Node.js 18 or later.

## Installation

The integration uses the [OCI TypeScript SDK](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/typescriptsdk.htm).

Install the package and its LangChain peer dependency:

```bash npm2yarn
npm install @oracle/langchain-oci @langchain/core
```

This package depends on `@langchain/core`. If you are using this package with other LangChain packages, make sure that all of the packages resolve to a compatible instance of `@langchain/core`.

The current package requires `@langchain/core` 1.x. For applications that need to force a single version across multiple LangChain packages, configure the appropriate package-manager override.

For example:

```json
{
  "name": "your-project",
  "version": "0.0.0",
  "dependencies": {
    "@langchain/core": "^1.0.0",
    "@oracle/langchain-oci": "^0.0.1"
  },
  "resolutions": {
    "@langchain/core": "^1.0.0"
  },
  "overrides": {
    "@langchain/core": "^1.0.0"
  },
  "pnpm": {
    "overrides": {
      "@langchain/core": "^1.0.0"
    }
  }
}
```

The field you need depends on the package manager you are using.

## Quick start

The simplest configuration uses OCI configuration-file authentication from `~/.oci/config`.

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
});

const response = await model.invoke(
  "Explain OCI Generative AI in one sentence."
);

console.log(response.content);
```

By default, the OCI SDK uses the `DEFAULT` profile from `~/.oci/config`.

## On-demand models and dedicated endpoints

Specify exactly one of `onDemandModelId` or `dedicatedEndpointId`.

### On-demand model

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
});
```

### Dedicated endpoint

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  dedicatedEndpointId: process.env.OCI_DEDICATED_ENDPOINT_ID!,
});
```

Do not specify both `onDemandModelId` and `dedicatedEndpointId`.

The `compartmentId` identifies the compartment in which the caller has permission to use OCI Generative AI.

## Authentication and OCI SDK client options

The chat and embeddings integrations share the OCI SDK client lifecycle and support multiple authentication configurations.

`OciGenAiNewClientAuthType` supports:

- Configuration-file authentication
- Instance Principal
- Resource Principal
- Session authentication
- A caller-provided OCI authentication provider

### Configuration-file authentication

The default configuration is equivalent to:

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
});
```

The OCI SDK reads credentials from `~/.oci/config` using the `DEFAULT` profile.

You can select a different config file and profile explicitly:

```ts
import {
  OciGenAiGenericChat,
  OciGenAiNewClientAuthType,
} from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  newClientParams: {
    authType: OciGenAiNewClientAuthType.ConfigFile,
    authParams: {
      clientConfigFilePath: "/my/path/config",
      clientProfile: "MY_PROFILE_IN_CONFIG_FILE",
    },
  },
});
```

### Instance Principal

Instance Principal authentication can be used when running in an OCI environment that is configured for instance principals.

```ts
import { MaxAttemptsTerminationStrategy, Region } from "oci-common";
import {
  OciGenAiGenericChat,
  OciGenAiNewClientAuthType,
} from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  newClientParams: {
    authType: OciGenAiNewClientAuthType.InstancePrincipal,
    regionId: Region.SA_SAOPAULO_1.regionId,
    clientConfiguration: {
      retryConfiguration: {
        terminationStrategy: new MaxAttemptsTerminationStrategy(3),
      },
    },
  },
});
```

See the OCI documentation for [authentication methods](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm) and [calling OCI services from instances](https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm).

### Resource Principal

Resource Principal authentication is intended for environments such as OCI Functions and Data Science where credentials are provided by the runtime environment.

```ts
import {
  OciGenAiGenericChat,
  OciGenAiNewClientAuthType,
} from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  newClientParams: {
    authType: OciGenAiNewClientAuthType.ResourcePrincipal,
  },
});
```

### Supplying your own OCI SDK client

You can create a `GenerativeAiInferenceClient` yourself when you need direct control over OCI SDK client creation and configuration.

```ts
import { ConfigFileAuthenticationDetailsProvider } from "oci-common";
import { GenerativeAiInferenceClient } from "oci-generativeaiinference";
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const client = new GenerativeAiInferenceClient({
  authenticationDetailsProvider: new ConfigFileAuthenticationDetailsProvider(),
});

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  client,
});
```

When a caller-provided client is supplied, the client remains owned by the caller.

## Invocation

`OciGenAiGenericChat` accepts additional OCI request parameters through the `requestParams` call option.

For OCI Generic chat requests, parameters such as `apiFormat`, `messages`, `isStream`, and `stop` are generated or inferred by the integration.

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const model = new OciGenAiGenericChat({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "meta.llama-3.3-70b-instruct",
});

const response = await model.invoke("Tell me a joke about beagles.", {
  requestParams: {
    temperature: 1,
    maxTokens: 300,
  },
});

console.log(response.content);
console.log(response.usage_metadata);
console.log(response.response_metadata);
```

The returned value is a LangChain `AIMessage`. Depending on the model and response, it can contain content, tool calls, token usage metadata, and provider response metadata.

For the full OCI Generic request schema, see the [GenericChatRequest API documentation](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/datatypes/GenericChatRequest).

For legacy Cohere V1 requests, see the [CohereChatRequest API documentation](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/datatypes/CohereChatRequest).

## Streaming

`OciGenAiGenericChat` supports LangChain streaming through the standard `stream()` API.

```ts
const stream = await model.stream("Explain OCI Generative AI.");

for await (const chunk of stream) {
  process.stdout.write(String(chunk.content));
}
```

Streaming returns LangChain `AIMessageChunk` instances as data is received from OCI Generative AI.

## Tool calling

`OciGenAiGenericChat` supports LangChain tool binding and tool-call/tool-result turns.

You can bind standard LangChain tools or compatible OpenAI-style tool definitions:

```ts
const tools = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get weather information for a location.",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and country to check.",
          },
        },
        required: ["location"],
      },
    },
  },
];

const modelWithTools = model.bindTools(tools);

const response = await modelWithTools.invoke(
  "What's the weather in Bengaluru?"
);

console.log(response.tool_calls);
```

The OCI Generic integration converts standard LangChain tool definitions into the OCI Generic function-definition format.

Tool choice can also be configured:

```ts
const modelWithRequiredTool = model.bindTools(tools, {
  tool_choice: "required",
});
```

Supported tool-choice forms include `auto`, `none`, `required`, `any`, boolean
values (`true` means required; `false` means none), a tool-name string, and an
OpenAI-style named-function choice.

Tool-call results can be supplied in subsequent LangChain turns using `ToolMessage`. The tool-call ID returned by OCI is used to correlate the tool result with the original model-generated call.

> [!NOTE]
> Tool result round trips described here apply to `OciGenAiGenericChat`. The legacy Cohere V1 integration does not support tool-result round trips.

## Structured output

Because `OciGenAiGenericChat` implements the standard LangChain chat-model interface, it can participate in LangChain's structured-output flow.

For example:

```ts
import { z } from "zod";

const structuredModel = model.withStructuredOutput(
  z.object({
    name: z.string(),
    description: z.string(),
  })
);

const result = await structuredModel.invoke("Describe OCI Generative AI.");

console.log(result);
```

Use structured output when the application needs model responses that conform to a defined schema rather than free-form text.

`OciGenAiGenericChat` currently supports structured output only through
function calling. LangChain `jsonMode`, `jsonSchema`, and `strict`
structured-output options are not implemented by this adapter.

Structured output is implemented by binding one function named after the
schema (or `extract` when the schema has no name) and forcing the model to
call it. If your schema has a top-level field literally called `name`, pass an
explicit function name so smaller models do not confuse the two:

```ts
const structuredModel = model.withStructuredOutput(personSchema, {
  name: "extract_person",
});
```

## Embeddings

`OciGenAiEmbeddings` provides text embeddings through OCI's `embedText` API.

It supports:

- On-demand models
- Dedicated endpoints
- Batching of document inputs
- `inputType` for model-specific purposes such as `SEARCH_DOCUMENT` and `SEARCH_QUERY`
- `truncate`
- `outputDimensions`
- The same OCI authentication and client lifecycle options as the chat integrations

OCI accepts up to 96 text inputs per `embedText` request. `OciGenAiEmbeddings` uses a default batch size of 96 and allows the batch size to be configured.

### Document embeddings

```ts
import { OciGenAiEmbeddings } from "@oracle/langchain-oci";
import { models } from "oci-generativeaiinference";

const embeddings = new OciGenAiEmbeddings({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "cohere.embed-v4.0",
  inputType: models.EmbedTextDetails.InputType.SearchDocument,
});

const documentVectors = await embeddings.embedDocuments([
  "OCI Generative AI provides embedding models.",
  "LangChain uses vectors for retrieval.",
]);

console.log(documentVectors);

await embeddings.close();
```

### Query embeddings

For models that distinguish between document and query inputs, use the appropriate `inputType` for query embeddings:

```ts
import { OciGenAiEmbeddings } from "@oracle/langchain-oci";
import { models } from "oci-generativeaiinference";

const embeddings = new OciGenAiEmbeddings({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "cohere.embed-v4.0",
  inputType: models.EmbedTextDetails.InputType.SearchQuery,
});

const queryVector = await embeddings.embedQuery("What does OCI provide?");

console.log(queryVector);

await embeddings.close();
```

`inputType` is configured per `OciGenAiEmbeddings` instance and applies to both
`embedDocuments()` and `embedQuery()`. For asymmetric retrieval, use a
`SEARCH_DOCUMENT`-configured instance while indexing documents and a separate
`SEARCH_QUERY`-configured instance while embedding search queries.

### Dedicated embedding endpoint

Embeddings can also be configured to use a dedicated endpoint:

```ts
const embeddings = new OciGenAiEmbeddings({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  dedicatedEndpointId: process.env.OCI_DEDICATED_ENDPOINT_ID!,
});
```

Specify either `onDemandModelId` or `dedicatedEndpointId`.

### Advanced embedding options

The embedding integration supports additional options such as `batchSize`, `truncate`, `inputType`, and `outputDimensions`.

For example:

```ts
const embeddings = new OciGenAiEmbeddings({
  compartmentId: process.env.OCI_COMPARTMENT_ID!,
  onDemandModelId: "cohere.embed-v4.0",
  inputType: models.EmbedTextDetails.InputType.SearchDocument,
  batchSize: 96,
  outputDimensions: 1024,
});
```

Use only options supported by the selected OCI embedding model.

## Development and testing

From `libs/js/oci`, install development dependencies and run the isolated unit test suite:

```bash
pnpm install
pnpm test
```

### Integration tests

The integration tests make real OCI Generative AI calls for generic and legacy Cohere V1 models.

Configure OCI API-key authentication in `~/.oci/config`. The `DEFAULT` profile is used by default.

Set the compartment and model IDs before running integration tests:

```bash
export OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID='<compartment-ocid>'

# OciGenAiCohereChat uses the legacy COHERE (V1) API format.
# Set this only when testing a legacy Cohere V1 model or
# compatible dedicated endpoint.
export OCI_GENAI_INTEGRATION_TESTS_COHERE_ON_DEMAND_MODEL_ID='<legacy-cohere-v1-model-id>'

export OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID='meta.llama-3.3-70b-instruct'

# Optional: prefer a specific embedding model.
# export OCI_GENAI_INTEGRATION_TESTS_EMBEDDING_ON_DEMAND_MODEL_ID='cohere.embed-english-v3.0'

# Optional: choose a non-default OCI config file or profile.
export OCI_CONFIG_FILE="$HOME/.oci/config"
export OCI_CONFIG_PROFILE='DEFAULT'

# Optional: use a non-Phoenix OCI GenAI endpoint.
# Override the region and endpoint together.
# export OCI_REGION='us-chicago-1'
# export OCI_ENDPOINT='https://inference.generativeai.us-chicago-1.oci.oraclecloud.com'

# Optional: run only one chat model family.
# This is useful when the tenancy does not offer a legacy Cohere V1 model.
# export OCI_GENAI_INTEGRATION_TESTS_CHAT_MODELS='generic'

pnpm test:int
```

Use model IDs available to your tenancy and region.

### Generic LangGraph tool-round-trip integration test

To run the real Generic LangGraph tool-round-trip test in Phoenix with xAI Grok:

```bash
export OCI_COMPARTMENT_ID='<compartment-ocid>'
export OCI_MODEL_ID='xai.grok-3'
export OCI_ENDPOINT='https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com'

pnpm test:langgraph:int
```

You can override the model and endpoint when testing another model or region.

## Resource lifecycle

Both chat models and embeddings expose `close()` to release an OCI SDK client created internally by the integration.

```ts
await model.close();
await embeddings.close();
```

A caller-provided OCI SDK client remains owned by the caller and is not closed by the LangChain integration.

## Limitations

The current integration is intentionally focused on text-based OCI Generative AI chat and text embeddings.

In particular:

- `OciGenAiCohereChat` uses the legacy Cohere V1 API format.
- Current Cohere chat models requiring Cohere V2 are not supported by `OciGenAiCohereChat`.
- Tool-result round trips are supported for the Generic integration, not the legacy Cohere integration.
- Chat message content is currently text-only. Multimodal message blocks such as images, audio, video, or documents are not supported by the current chat adapter.
- `OciGenAiEmbeddings` currently uses OCI's `embedText` API for text embeddings.

## Additional information

For additional information, see the [OCI Generative AI service documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm).

For OCI Generative AI API details, see the [Generative AI API documentation](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/).

If you are interested in the Python version of this integration, see the [LangChain OCI Generative AI documentation](https://python.langchain.com/docs/integrations/llms/oci_generative_ai/).

## Related

- [LangChain chat model documentation](https://docs.langchain.com/oss/javascript/langchain/models)
- [LangChain structured output documentation](https://docs.langchain.com/oss/javascript/langchain/structured-output)
