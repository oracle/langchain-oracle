/* eslint-disable no-process-env */

import { expect, test } from "vitest";

import { OciGenAiEmbeddings } from "../embeddings.js";
import { OciGenAiNewClientAuthType } from "../types.js";

// This test is opt-in because it makes billable OCI requests. It uses the
// standard OCI CLI variables so it can run from an already configured shell.
const compartmentId =
  process.env.OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID ??
  process.env.OCI_COMPARTMENT_ID;
const configuredModelId =
  process.env.OCI_GENAI_INTEGRATION_TESTS_EMBEDDING_ON_DEMAND_MODEL_ID;
const regionId = process.env.OCI_REGION ?? "us-phoenix-1";
const configFilePath = process.env.OCI_CONFIG_FILE;
const configProfile = process.env.OCI_CONFIG_PROFILE;

// The management ListModels API identifies base models with OCI resource OCIDs,
// but on-demand embedText expects these public inference model names. Embed 4
// is not currently on demand in Phoenix, so retain the broadly available V3
// models as portable test fallbacks.
const defaultEmbeddingModelIds = [
  "cohere.embed-english-v3.0",
  "cohere.embed-multilingual-v3.0",
  "cohere.embed-english-light-v3.0",
  "cohere.embed-multilingual-light-v3.0",
];

test.skipIf(!compartmentId)(
  "OCI GenAI text embeddings",
  async () => {
    const modelIds = getEmbeddingModelCandidates();
    let unavailableModelError: unknown;

    for (const modelId of modelIds) {
      const embeddings = new OciGenAiEmbeddings({
        compartmentId: compartmentId!,
        onDemandModelId: modelId,
        newClientParams: {
          authType: OciGenAiNewClientAuthType.ConfigFile,
          regionId,
          serviceEndpoint:
            process.env.OCI_ENDPOINT ??
            "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com",
          authParams:
            configFilePath || configProfile
              ? {
                  clientConfigFilePath: configFilePath ?? "",
                  clientProfile: configProfile ?? "DEFAULT",
                }
              : undefined,
        },
      });

      try {
        const documentVectors = await embeddings.embedDocuments([
          "OCI Generative AI supports text embedding models.",
          "LangChain retrieval uses document and query vectors.",
        ]);
        const queryVector = await embeddings.embedQuery(
          "What does OCI Generative AI support?"
        );

        expect(documentVectors).toHaveLength(2);
        expect(documentVectors.every((vector) => vector.length > 0)).toBe(true);
        expect(queryVector.length).toBeGreaterThan(0);
        return;
      } catch (error) {
        if (!isUnavailableEmbeddingModelError(error)) {
          throw error;
        }
        unavailableModelError = error;
        console.warn(
          `OCI embedding integration model '${modelId}' is unavailable; trying the next listed model.`
        );
      } finally {
        await embeddings.close();
      }
    }

    throw new Error(
      `No listed on-demand embedding model was usable in ${regionId}. Tried: ${modelIds.join(
        ", "
      )}`,
      { cause: unavailableModelError }
    );
  },
  100_000
);

function getEmbeddingModelCandidates(): string[] {
  const modelIds = [configuredModelId, ...defaultEmbeddingModelIds]
    .filter((modelId): modelId is string => typeof modelId === "string")
    .filter(
      (modelId, index, candidates) => candidates.indexOf(modelId) === index
    );

  console.info(
    `Trying ${modelIds.length} OCI embedding integration candidates in ${regionId}; configured model is tried first.`
  );
  return modelIds;
}

function isUnavailableEmbeddingModelError(error: unknown): boolean {
  let currentError = error;
  while (currentError !== null && typeof currentError === "object") {
    if (
      "message" in currentError &&
      typeof currentError.message === "string" &&
      /Entity with key .+ not found/.test(currentError.message)
    ) {
      return true;
    }
    currentError = "cause" in currentError ? currentError.cause : undefined;
  }
  return false;
}
