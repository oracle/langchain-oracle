# @oracle/langchain-oci

Oracle Cloud Infrastructure (OCI) Generative AI is a fully managed
service that provides a set of state-of-the-art, customizable large
language models (LLMs) that cover a wide range of use cases, and which
is available through a single API. Using the OCI Generative AI service
you can access ready-to-use pretrained models, or create and host your
own fine-tuned custom models based on your own data on dedicated AI
clusters. Detailed documentation of the service and API is available
[here](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
and
[here](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/).

This package enables you to use OCI Generative AI in your LangChainJS applications.

`OciGenAiGenericChat` supports text chat, streaming, token usage and finish
metadata, LangChain tool binding, and tool-message turns. This enables the
standard LangChain structured-output flow for OCI Generic models. Cohere support
uses OCI's legacy V1 API format; its tool-result round trip is not supported.

## Prerequisites

In order to use this integration you will need the following:

1. An OCI
   tenancy. If you do not already have and account, please create one
   [here](https://signup.cloud.oracle.com?sourceType=:ex:of:::::LangChainJSIntegration&SC=:ex:of:::::LangChainJSIntegration&pcode=). 2. Setup an [authentication
   method](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm)
   (Using a [configuration
   file](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm)
   with [API Key
   authentication](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm#apisigningkey_topic_How_to_Generate_an_API_Signing_Key_Console)
   is the simplest to start with). 3. Please make sure that your OCI
   tenancy is registered in one of the [supported
   regions](https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm#regions). 4. You will need the ID (aka OCID) of a compartment in which your OCI
   user has [access to use the Generative AI
   service](https://docs.oracle.com/en-us/iaas/Content/generative-ai/iam-policies.htm).
   You can either use the `root` compartment or [create your
   own](https://docs.oracle.com/en-us/iaas/Content/Identity/compartments/To_create_a_compartment.htm). 5. Retrieve the desired model name from the [available
   models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)
   list (please make sure not to select a deprecated model).

## Installation

The integration makes use of the [OCI TypeScript
SDK](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/typescriptsdk.htm).
To install the integration dependencies, execute the following:

```bash npm2yarn
npm install oci-common oci-generativeaiinference @langchain/core @oracle/langchain-oci
```

This package, along with the main LangChain package, depends on [`@langchain/core`](https://npmjs.com/package/@langchain/core/).
If you are using this package with other LangChain packages, you should make sure that all of the packages depend on the same instance of @langchain/core.
You can do so by adding appropriate field to your project's `package.json` like this:

```json
{
  "name": "your-project",
  "version": "0.0.0",
  "dependencies": {
    "@langchain/core": "^0.3.0",
    "@oracle/langchain-oci": "^0.0.1"
  },
  "resolutions": {
    "@langchain/core": "^0.3.0"
  },
  "overrides": {
    "@langchain/core": "^0.3.0"
  },
  "pnpm": {
    "overrides": {
      "@langchain/core": "^0.3.0"
    }
  }
}
```

The field you need depends on the package manager you're using, but we recommend adding a field for the common `yarn`, `npm`, and `pnpm` to maximize compatibility.

## Development and testing

From `libs/js/oci`, install development dependencies and run the isolated test suite:

```bash
pnpm install
pnpm test
```

The integration tests make real OCI Generative AI calls for both Cohere and
generic models. Configure OCI API-key authentication in `~/.oci/config` (the
`DEFAULT` profile is used by default), then set the compartment and the two
model IDs before running them. They target Phoenix by default; override the
region and endpoint together when testing another region:

```bash
export OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID='<compartment-ocid>'
# OciGenAiCohereChat uses the legacy COHERE (V1) API format. Set this only
# when testing a legacy Cohere V1 model or compatible dedicated endpoint.
export OCI_GENAI_INTEGRATION_TESTS_COHERE_ON_DEMAND_MODEL_ID='<legacy-cohere-v1-model-id>'
export OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID='meta.llama-3.3-70b-instruct'
# Optional: prefer a specific embedding model. In Phoenix, use a Cohere Embed
# V3 model; the test falls back to the documented V3 public model IDs on 404.
# export OCI_GENAI_INTEGRATION_TESTS_EMBEDDING_ON_DEMAND_MODEL_ID='cohere.embed-english-v3.0'
# Optional: choose a non-default OCI config file or profile.
export OCI_CONFIG_FILE="$HOME/.oci/config"
export OCI_CONFIG_PROFILE='DEFAULT'
# Optional: use a non-Phoenix OCI GenAI endpoint.
# export OCI_REGION='us-chicago-1'
# export OCI_ENDPOINT='https://inference.generativeai.us-chicago-1.oci.oraclecloud.com'
# Optional: run only one model family. This is useful when the tenancy does not
# offer a legacy Cohere V1 model.
# export OCI_GENAI_INTEGRATION_TESTS_CHAT_MODELS='generic'
pnpm test:int
```

To run the real Generic LangGraph tool-round-trip test in Phoenix with xAI Grok,
set the compartment ID (and optionally override the model or endpoint), then run
the focused command:

```bash
export OCI_COMPARTMENT_ID='<compartment-ocid>'
export OCI_MODEL_ID='xai.grok-3'
export OCI_ENDPOINT='https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com'
pnpm test:langgraph:int
```

Use model IDs available to your tenancy and region. To use a non-default OCI
profile or authentication method, pass `newClientParams` when constructing the
chat model, as shown below.

`OciGenAiNewClientAuthType` supports configuration-file, instance-principal,
resource-principal, and session authentication, as well as a caller-provided
OCI authentication provider. Resource Principal authentication is intended for
OCI Functions and Data Science environments, where the SDK reads credentials
from the runtime environment.

## Instantiation

The OCI Generative AI service supports two groups of LLMs: 1. Cohere
family of LLMs. 2. Generic family of LLMs which include model such as
Llama.

The following code demonstrates how to create an instance for the generic
family. `OciGenAiCohereChat` uses OCI's legacy COHERE (V1) API format; current
Cohere models require V2 support and are not yet supported by this package. The
only mandatory two parameters are: 1.
`compartmentId` - A compartment OCID in which the user you are using for
authentication was granted permissions to access the Generative AI
service. 2. `onDemandModelId` or `dedicatedEndpointId` - Either a
[pre-trained
model](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)
name/OCID or a dedicated endpoint OCID for an endpoint configured on a
[dedicated AI cluster
(DAC)](https://docs.oracle.com/en-us/iaas/Content/generative-ai/ai-cluster.htm).
Either `onDemandModelId` or `dedicatedEndpointId` must be provided but
not both.

In this example, since no other parameters are specified, a default SDK
client will be created with the following configuration: 1.
Authentication will be attempted using a [configuration
file](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm)
which should be already setup and available under `~/.oci/config`. The
`config` file is expected to contain a `DEFAULT` profile with the
correct information. Please see the prerequisites for more information. 2. The retry strategy will be set to a single attempt. If the first API
call was not successful, the request will fail. 3. The region will be
set to `us-chicago-1`. Please make sure that your tenancy is registered
this region.

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const genericLlm = new OciGenAiGenericChat({
  compartmentId: "oci.compartment...",
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  // dedicatedEndpointId: "oci.dedicatedendpoint..."
});
```

## SDK client options

The above example used default values to create the SDK client behind
the scenes. If you need more control in the creation of the client, here
are additional options for `OciGenAiGenericChat`.

The first example will create an SDK client with the following
configuration: 1. [Instance Principal
authentication](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm#sdk_authentication_methods_instance_principaldita).
Please note that this authentication method requires
[configuration](https://docs.oracle.com/en-us/iaas/Content/Identity/Tasks/callingservicesfrominstances.htm). 2. Using the Sao Paulo region. 3. Up to 3 attempts will be made in case
API calls fail.

```ts
import { MaxAttemptsTerminationStrategy, Region } from "oci-common";
import {
  OciGenAiGenericChat,
  OciGenAiNewClientAuthType,
} from "@oracle/langchain-oci";

const genericLlm = new OciGenAiGenericChat({
  compartmentId: "oci.compartment...",
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

The second example will create an SDK client with the following
configuration: 1. Config file authentication. 1. Use the config file:
`/my/path/config`. 1. Use the details under the
`MY_PROFILE_IN_CONFIG_FILE` profile in the specified config file. 1. The
retry strategy will be set to a single attempt. If the first API call
was not successful, the request will fail. 1. The region will be set to
`us-chicago-1`. Please make sure that your tenancy is registered this
region.

```ts
import { OciGenAiGenericChat, OciGenAiNewClientAuthType } from "@oracle/langchain-oci";

const genericLlm = new OciGenAiGenericChat({
  compartmentId: "oci.compartment...",
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

The third example will create an SDK client with the following
configuration: 1. Use [Resource
Principal](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm#sdk_authentication_methods_resource_principal)
authentication. 1. The retry strategy will be set to a single attempt.
If the first API call was not successful, the request will fail. 1. The
region will be set to `us-chicago-1`. Please make sure that your tenancy
is registered this region.

```ts
import {
  OciGenAiGenericChat,
  OciGenAiNewClientAuthType,
} from "@oracle/langchain-oci";

const genericLlm = new OciGenAiGenericChat({
  compartmentId: "oci.compartment...",
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  newClientParams: {
    authType: OciGenAiNewClientAuthType.ResourcePrincipal,
  },
});
```

You can also instantiate the OCI Generative AI chat classes using
`GenerativeAiInferenceClient` that you create on your own. This way you
control the creation and configuration of the client to suit your
specific needs:

```ts
import { ConfigFileAuthenticationDetailsProvider } from "oci-common";
import { GenerativeAiInferenceClient } from "oci-generativeaiinference";
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

const client = new GenerativeAiInferenceClient({
  authenticationDetailsProvider: new ConfigFileAuthenticationDetailsProvider(),
});

const genericLlm = new OciGenAiGenericChat({
  compartmentId: "oci.compartment...",
  onDemandModelId: "meta.llama-3.3-70b-instruct",
  client,
});
```

## Invocation

In this example, we make a simple call to the OCI Generative AI service using a
generic model.
Please note that you can pass additional request parameters under the
`requestParams` key as shown in the `invoke` call below. For more
information please see the [Cohere request
parameters](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/datatypes/CohereChatRequest)
(the `apiFormat`, `chatHistory`, `isStream`, `message` & `stopSequences`
parameters are automatically generated or inferred from the call
context) and the [Generic request
parameters](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/datatypes/GenericChatRequest)
(the `apiFormat`, `isStream`, `messages` & `stop` parameters are
automatically generated or inferred from the call context).

```ts
import { OciGenAiGenericChat } from "@oracle/langchain-oci";

(async () => {
  const llm = new OciGenAiGenericChat({
    compartmentId: "oci.compartment...",
    onDemandModelId: "meta.llama-3.3-70b-instruct",
  });

  const result = await llm.invoke("Tell me a joke about beagles", {
    requestParams: {
      temperature: 1,
      maxTokens: 300,
    },
  });

  console.log(result);
})();
```

AIMessage { “content”: “Why did the beagle cross the road?he was tied to
the chicken!hope you enjoyed the joke! Would you like to hear another
one?”, “additional_kwargs”: {}, “response_metadata”: {}, “tool_calls”:
\[\], “invalid_tool_calls”: \[\] }

## Embeddings

`OciGenAiEmbeddings` provides text embeddings through OCI's `embedText` API.
It supports on-demand models and dedicated endpoints, batches documents (up to
96 strings per request by default), and uses the same authentication and client
lifecycle options as the chat integrations. Set `inputType` when the selected
model requires a purpose such as `SEARCH_DOCUMENT` or `SEARCH_QUERY`.

```ts
import { OciGenAiEmbeddings } from "@oracle/langchain-oci";
import { models } from "oci-generativeaiinference";

const embeddings = new OciGenAiEmbeddings({
  compartmentId: "oci.compartment...",
  onDemandModelId: "cohere.embed-v4.0",
  inputType: models.EmbedTextDetails.InputType.SearchDocument,
});

const documentVectors = await embeddings.embedDocuments([
  "OCI Generative AI provides embedding models.",
  "LangChain uses vectors for retrieval.",
]);
const queryVector = await embeddings.embedQuery("What does OCI provide?");

await embeddings.close();
```

## Additional information

For additional information, please checkout the [OCI Generative AI
service
documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm).

If you are interested in the python version of this integration, you can
find more information
[here](https://python.langchain.com/docs/integrations/llms/oci_generative_ai/).

## Related

- Chat model [conceptual guide](/docs/concepts/#chat-models)
- Chat model [how-to guides](/docs/how_to/#chat-models)
