/* eslint-disable no-process-env */

import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { expect, test } from "vitest";

import { AIMessageChunk } from "@langchain/core/messages";
import { OciGenAiCohereChat } from "../cohere_chat.js";
import { OciGenAiGenericChat } from "../generic_chat.js";
import { OciGenAiNewClientAuthType } from "../types.js";
import type { OciGenAiModelBaseParams } from "../types.js";

type OciGenAiChatParameters = Partial<OciGenAiModelBaseParams>;
type OciGenAiChatConstructor = new (
  args: OciGenAiChatParameters
) => BaseChatModel;
type OciGenAiChatModelFamily = "cohere" | "generic";

interface OciGenAiChatTestConfiguration {
  family: OciGenAiChatModelFamily;
  ChatClassType: OciGenAiChatConstructor;
  creationParams: OciGenAiChatParameters[];
}

/*
 *  OciGenAiChat tests
 */

const compartmentId = process.env.OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID;
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
  await testEachChatModelType(
    async (
      ChatClassType: OciGenAiChatConstructor,
      creationParams: OciGenAiChatParameters[]
    ) => {
      for (const params of creationParams) {
        const chatClass = new ChatClassType(params);
        const response = await chatClass.invoke(
          "generate a marketing slogan for a pet insurance company"
        );

        expect(response.content.length).toBeGreaterThan(0);
      }
    }
  );
});

test("OCI GenAI chat stream", async () => {
  await testEachChatModelType(
    async (
      ChatClassType: OciGenAiChatConstructor,
      creationParams: OciGenAiChatParameters[]
    ) => {
      for (const params of creationParams) {
        const chatClass = new ChatClassType(params);
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
      }
    }
  );
});

/*
 * Utils
 */

async function testEachChatModelType(
  testFunction: (
    ChatClassType: OciGenAiChatConstructor,
    creationParams: OciGenAiChatParameters[]
  ) => Promise<void>
) {
  for (const {
    ChatClassType,
    creationParams,
  } of selectedChatModelConfigurations) {
    await testFunction(ChatClassType, creationParams);
  }
}
