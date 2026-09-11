/* eslint-disable no-process-env */

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  ConfigFileAuthenticationDetailsProvider,
  type AuthenticationDetailsProvider,
} from "oci-common";
import {
  GenerativeAiClient,
  models as generativeAiModels,
} from "oci-generativeai";
import { GenerativeAiInferenceClient } from "oci-generativeaiinference";
import { expect, test } from "vitest";
import { z } from "zod";

import { createAgent, tool } from "langchain";
import { AIMessageChunk } from "@langchain/core/messages";
import { OciGenAiCohereChat } from "../cohere_chat.js";
import { OciGenAiGenericChat } from "../generic_chat.js";
import { OciGenAiNewClientAuthType } from "../types.js";
import type { OciGenAiModelBaseParams } from "../types.js";

type OciGenAiChatParameters = Partial<OciGenAiModelBaseParams>;
type OciGenAiChatConstructor = new (
  args: OciGenAiChatParameters
) => OciGenAiTestChatModel;
type OciGenAiTestChatModel = BaseChatModel & { close(): Promise<void> };
type OciGenAiChatModelFamily = "cohere" | "generic";

interface OciGenAiChatTestConfiguration {
  family: OciGenAiChatModelFamily;
  ChatClassType: OciGenAiChatConstructor;
  creationParams: OciGenAiChatParameters[];
}

/*
 *  OciGenAiChat tests
 */

// Reuse the standard OCI CLI variable when the test-specific variable is not
// present, so a developer can run the test from an already configured shell.
const compartmentId =
  process.env.OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID ??
  process.env.OCI_COMPARTMENT_ID;
const regionId = process.env.OCI_REGION ?? "us-phoenix-1";
const serviceEndpoint =
  process.env.OCI_ENDPOINT ??
  "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com";
const configFilePath = process.env.OCI_CONFIG_FILE;
const configProfile = process.env.OCI_CONFIG_PROFILE;
const newClientParams = {
  authType: OciGenAiNewClientAuthType.ConfigFile,
  regionId,
  serviceEndpoint,
  authParams:
    configFilePath || configProfile
      ? {
          clientConfigFilePath: configFilePath ?? "",
          clientProfile: configProfile ?? "DEFAULT",
        }
      : undefined,
};
let resolvedGenericModelId: string | undefined;
const chatModelConfigurations: OciGenAiChatTestConfiguration[] = [
  {
    family: "cohere",
    ChatClassType: OciGenAiCohereChat,
    creationParams: [
      {
        compartmentId,
        onDemandModelId:
          process.env.OCI_GENAI_INTEGRATION_TESTS_COHERE_ON_DEMAND_MODEL_ID,
        newClientParams,
      },
    ],
  },
  {
    family: "generic",
    ChatClassType: OciGenAiGenericChat,
    creationParams: [
      {
        compartmentId,
        onDemandModelId:
          process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
        newClientParams,
      },
    ],
  },
];
const selectedFamilies = new Set(
  (process.env.OCI_GENAI_INTEGRATION_TESTS_CHAT_MODELS ?? "cohere,generic")
    .split(",")
    .map((family) => family.trim())
);
const selectedChatModelConfigurations = chatModelConfigurations.filter(
  ({ family }) => selectedFamilies.has(family)
);

if (selectedChatModelConfigurations.length === 0) {
  throw new Error(
    "OCI_GENAI_INTEGRATION_TESTS_CHAT_MODELS must include cohere or generic"
  );
}

test("OCI GenAI chat invoke", async () => {
  await testEachChatModelType(async (chatClass: OciGenAiTestChatModel) => {
    const response = await chatClass.invoke(
      "generate a marketing slogan for a pet insurance company"
    );

    expect(response.content.length).toBeGreaterThan(0);
  });
});

test("OCI GenAI chat stream", async () => {
  await testEachChatModelType(async (chatClass: OciGenAiTestChatModel) => {
    const response = await chatClass.stream(
      "generate a story about person and their dog"
    );

    let numChunks: number = 0;

    for await (const chunk of response) {
      expect(chunk).toBeInstanceOf(AIMessageChunk);
      expect(chunk.content).toBeDefined();
      numChunks += 1;
    }

    expect(numChunks).toBeGreaterThan(0);
    console.log(`Chunks generated: ${numChunks}`);
  });
});

test.skipIf(!selectedFamilies.has("generic") || !compartmentId)(
  "OCI GenAI Generic chat accepts invocation request parameters",
  async () => {
    await testGenericChatWithModelFallback(
      OciGenAiGenericChat,
      {
        compartmentId,
        onDemandModelId:
          process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
      },
      async (chatClass) => {
        const genericChat = chatClass as OciGenAiGenericChat;
        const response = await genericChat.invoke(
          "Tell me a joke about beagles.",
          {
            requestParams: {
              temperature: 1,
              maxTokens: 300,
            },
          }
        );

        expect(response.content.length).toBeGreaterThan(0);
        expect(response.response_metadata).toBeDefined();
      }
    );
  }
);

test.skipIf(!selectedFamilies.has("generic") || !compartmentId)(
  "OCI GenAI Generic chat supports structured output",
  async () => {
    await testGenericChatWithModelFallback(
      OciGenAiGenericChat,
      {
        compartmentId,
        onDemandModelId:
          process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
      },
      async (chatClass) => {
        const structuredModel = (
          chatClass as OciGenAiGenericChat
        ).withStructuredOutput(
          z.object({
            name: z.string(),
            description: z.string(),
          })
        );
        const result = await structuredModel.invoke(
          "Use the provided extraction tool to describe OCI Generative AI."
        );

        expect(result.name).toBeTypeOf("string");
        expect(result.description).toBeTypeOf("string");
      }
    );
  }
);

test.skipIf(!selectedFamilies.has("generic") || !compartmentId)(
  "OCI GenAI Generic chat supports structured output streaming",
  async () => {
    await testGenericChatWithModelFallback(
      OciGenAiGenericChat,
      {
        compartmentId,
        onDemandModelId:
          process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
      },
      async (chatClass) => {
        const structuredModel = (
          chatClass as OciGenAiGenericChat
        ).withStructuredOutput(
          z.object({
            name: z.string(),
            description: z.string(),
          })
        );

        const stream = await structuredModel.stream(
          "Use the provided extraction tool to describe OCI Generative AI."
        );

        const chunks = [];

        for await (const chunk of stream) {
          chunks.push(chunk);
        }

        expect(chunks.length).toBeGreaterThan(0);

        const result = chunks[chunks.length - 1];

        expect(result.name).toBeTypeOf("string");
        expect(result.name.length).toBeGreaterThan(0);
        expect(result.description).toBeTypeOf("string");
        expect(result.description.length).toBeGreaterThan(0);
      }
    );
  }
);

// Minimal Generic-chat example. In particular, no
// `newClientParams` are supplied, so the OCI SDK resolves API-key credentials
// from ~/.oci/config using its DEFAULT profile.
test.skipIf(
  !selectedFamilies.has("generic") ||
    !compartmentId ||
    !process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID
)(
  "OCI GenAI Generic chat uses the SDK default config-file credentials",
  async () => {
    const chatClass = new OciGenAiGenericChat({
      compartmentId,
      onDemandModelId:
        process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
    });

    try {
      const response = await chatClass.invoke("Reply with one short greeting.");
      expect(response.content.length).toBeGreaterThan(0);
    } finally {
      await chatClass.close();
    }
  }
);

// Non-default config-file example. Deliberately do
// not supply regionId or serviceEndpoint: the SDK must obtain the region from
// the selected OCI profile as well as use that profile's credentials.
test.skipIf(
  !selectedFamilies.has("generic") ||
    !compartmentId ||
    !configFilePath ||
    !configProfile ||
    !process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID
)("OCI GenAI Generic chat uses an explicit config-file profile", async () => {
  if (!configFilePath || !configProfile) {
    throw new Error(
      "OCI_CONFIG_FILE and OCI_CONFIG_PROFILE are required for this test"
    );
  }

  const chatClass = new OciGenAiGenericChat({
    compartmentId,
    onDemandModelId:
      process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
    newClientParams: {
      authType: OciGenAiNewClientAuthType.ConfigFile,
      authParams: {
        clientConfigFilePath: configFilePath,
        clientProfile: configProfile,
      },
    },
  });

  try {
    const response = await chatClass.invoke("Reply with one short greeting.");
    expect(response.content.length).toBeGreaterThan(0);
  } finally {
    await chatClass.close();
  }
});

// Session authentication uses an OCI CLI security-token session stored in
// the selected config profile. The session profile must contain a
// security_token_file entry and be valid when this test runs.
test.skipIf(
  !selectedFamilies.has("generic") ||
    !compartmentId ||
    !process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID ||
    !process.env.OCI_SESSION_CONFIG_PROFILE
)("OCI GenAI Generic chat uses session authentication", async () => {
  const sessionConfigFile = process.env.OCI_SESSION_CONFIG_FILE;
  const sessionConfigProfile = process.env.OCI_SESSION_CONFIG_PROFILE;

  if (!sessionConfigProfile) {
    throw new Error(
      "OCI_SESSION_CONFIG_PROFILE is required for the session-authentication test"
    );
  }

  const chatClass = new OciGenAiGenericChat({
    compartmentId,
    onDemandModelId:
      process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
    newClientParams: {
      authType: OciGenAiNewClientAuthType.Session,
      authParams: {
        clientConfigFilePath: sessionConfigFile ?? "",
        clientProfile: sessionConfigProfile,
      },
    },
  });

  try {
    const response = await chatClass.invoke("Reply with one short greeting.");
    expect(response.content.length).toBeGreaterThan(0);
  } finally {
    await chatClass.close();
  }
});

test.skipIf(
  !selectedFamilies.has("generic") ||
    !compartmentId ||
    !process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID
)(
  "OCI GenAI Generic chat supports LangChain agent tool round-trip",
  async () => {
    await testGenericChatWithModelFallback(
      OciGenAiGenericChat,
      {
        compartmentId,
        onDemandModelId:
          process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
      },
      async (chatClass) => {
        let toolCalled = false;

        const queryInfrastructure = tool(
          async ({
            resourceType,
            region,
          }: {
            resourceType: string;
            region: string;
          }) => {
            toolCalled = true;

            expect(resourceType).toBe("compute");
            expect(region).toBe("us-ashburn-1");

            return {
              status: "healthy",
              active_instances: 12,
              cpu_utilization: "45%",
              alerts: [],
            };
          },
          {
            name: "query_infrastructure",
            description: "Query OCI infrastructure status and health metrics.",
            schema: z.object({
              resourceType: z.string(),
              region: z.string(),
            }),
          }
        );

        const agent = createAgent({
          model: chatClass,
          tools: [queryInfrastructure],
          systemPrompt:
            "You are an infrastructure monitoring assistant. " +
            "Use query_infrastructure whenever the user asks about OCI infrastructure health.",
        });

        const result = await agent.invoke({
          messages: [
            {
              role: "user",
              content:
                "Check compute resource health in us-ashburn-1 and summarize the result.",
            },
          ],
        });

        expect(toolCalled).toBe(true);

        expect(result.messages.length).toBeGreaterThan(1);

        const toolMessages = result.messages.filter(
          (message) => message.getType() === "tool"
        );

        expect(toolMessages.length).toBeGreaterThan(0);

        const lastMessage = result.messages[result.messages.length - 1];

        expect(lastMessage).toBeDefined();
        expect(lastMessage.getType()).toBe("ai");
        expect(lastMessage.content).toBeTruthy();

        const finalText =
          typeof lastMessage.content === "string"
            ? lastMessage.content
            : JSON.stringify(lastMessage.content);

        expect(finalText).toContain("healthy");
        expect(finalText).toContain("12");
        expect(finalText).toContain("45%");
      }
    );
  }
);

// The application owns an
// SDK client built from the default config provider, and closes it itself.
test.skipIf(
  !selectedFamilies.has("generic") ||
    !compartmentId ||
    !process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID
)(
  "OCI GenAI Generic chat accepts an injected config-file SDK client",
  async () => {
    const client = new GenerativeAiInferenceClient({
      authenticationDetailsProvider:
        new ConfigFileAuthenticationDetailsProvider(),
    });
    const chatClass = new OciGenAiGenericChat({
      compartmentId,
      onDemandModelId:
        process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID,
      client,
    });

    try {
      const response = await chatClass.invoke("Reply with one short greeting.");
      expect(response.content.length).toBeGreaterThan(0);
    } finally {
      await chatClass.close();
      client.close();
    }
  }
);

/*
 * Utils
 */

async function testEachChatModelType(
  testFunction: (chatClass: OciGenAiTestChatModel) => Promise<void>
) {
  for (const {
    family,
    ChatClassType,
    creationParams,
  } of selectedChatModelConfigurations) {
    for (const params of creationParams) {
      if (family === "generic") {
        await testGenericChatWithModelFallback(
          ChatClassType,
          params,
          testFunction
        );
      } else {
        await testChatModel(ChatClassType, params, testFunction);
      }
    }
  }
}

async function testChatModel(
  ChatClassType: OciGenAiChatConstructor,
  params: OciGenAiChatParameters,
  testFunction: (chatClass: OciGenAiTestChatModel) => Promise<void>
) {
  const chatClass = new ChatClassType(params);
  try {
    await testFunction(chatClass);
  } finally {
    // Integration tests own these clients, including after a failed request.
    await chatClass.close();
  }
}

async function testGenericChatWithModelFallback(
  ChatClassType: OciGenAiChatConstructor,
  params: OciGenAiChatParameters,
  testFunction: (chatClass: OciGenAiTestChatModel) => Promise<void>
) {
  const modelIds = await getGenericModelCandidates();
  let unavailableModelError: unknown;

  for (const modelId of modelIds) {
    try {
      await testChatModel(
        ChatClassType,
        { ...params, onDemandModelId: modelId },
        testFunction
      );
      resolvedGenericModelId = modelId;
      return;
    } catch (error) {
      if (!isUnavailableGenericModelError(error)) {
        throw error;
      }
      unavailableModelError = error;
      console.warn(
        `OCI Generic integration model '${modelId}' is unavailable or does not support chat; trying the next listed model.`
      );
    }
  }

  throw new Error(
    `No listed Generic on-demand chat model was usable in ${regionId}. Tried: ${modelIds.join(
      ", "
    )}`,
    { cause: unavailableModelError }
  );
}

async function getGenericModelCandidates(): Promise<string[]> {
  if (!compartmentId) {
    throw new Error("OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID is required");
  }

  const modelClient = new GenerativeAiClient({
    authenticationDetailsProvider: getConfigFileAuthProvider(),
  });
  modelClient.regionId = regionId;

  try {
    const availableModels = await listActiveOnDemandChatModels(modelClient);
    const configuredModelId =
      process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID;
    const modelIds = [
      resolvedGenericModelId,
      configuredModelId,
      ...availableModels,
    ]
      .filter((modelId): modelId is string => typeof modelId === "string")
      .filter(
        (modelId, index, candidates) => candidates.indexOf(modelId) === index
      );

    if (modelIds.length === 0) {
      throw new Error(
        `No active non-Cohere on-demand chat models were listed in ${regionId}`
      );
    }

    console.info(
      `Found ${modelIds.length} OCI Generic integration candidates in ${regionId}; configured model is tried first.`
    );
    return modelIds;
  } finally {
    modelClient.close();
  }
}

function getConfigFileAuthProvider(): AuthenticationDetailsProvider {
  return new ConfigFileAuthenticationDetailsProvider(
    configFilePath,
    configProfile
  );
}

async function listActiveOnDemandChatModels(
  modelClient: GenerativeAiClient
): Promise<string[]> {
  const modelIds: string[] = [];
  let page: string | undefined;

  do {
    const response = await modelClient.listModels({
      compartmentId: compartmentId as string,
      capability: [generativeAiModels.ModelCapability.Chat],
      lifecycleState: "ACTIVE",
      page,
    });
    modelIds.push(
      ...response.modelCollection.items
        .filter(
          (model) =>
            model.type === "BASE" &&
            model.capabilities.includes(
              generativeAiModels.ModelSummary.Capabilities.Chat
            ) &&
            !model.id.startsWith("cohere.") &&
            !isOnDemandRetired(model.timeOnDemandRetired)
        )
        .map((model) => model.id)
    );
    page = response.opcNextPage || undefined;
  } while (page);

  return modelIds;
}

function isOnDemandRetired(timeOnDemandRetired: unknown): boolean {
  if (timeOnDemandRetired === undefined || timeOnDemandRetired === null) {
    return false;
  }

  // OCI SDK type declarations specify Date, but ListModels can deserialize
  // this timestamp as its RFC3339 string representation.
  let retiredAt = Number.NaN;
  if (
    typeof timeOnDemandRetired === "object" &&
    "getTime" in timeOnDemandRetired &&
    typeof timeOnDemandRetired.getTime === "function"
  ) {
    retiredAt = timeOnDemandRetired.getTime();
  } else if (typeof timeOnDemandRetired === "string") {
    retiredAt = Date.parse(timeOnDemandRetired);
  }
  return Number.isFinite(retiredAt) && retiredAt <= Date.now();
}

function isUnavailableGenericModelError(error: unknown): boolean {
  let currentError = error;
  while (currentError !== null && typeof currentError === "object") {
    if (
      "message" in currentError &&
      typeof currentError.message === "string" &&
      (/Entity with key .+ not found/.test(currentError.message) ||
        /does not support Chat/.test(currentError.message))
    ) {
      return true;
    }
    currentError = "cause" in currentError ? currentError.cause : undefined;
  }
  return false;
}
