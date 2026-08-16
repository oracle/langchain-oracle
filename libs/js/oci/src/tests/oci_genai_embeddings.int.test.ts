/* eslint-disable no-process-env */

import { expect, test } from "vitest";

import { OciGenAiEmbeddings } from "../embeddings.js";
import { OciGenAiNewClientAuthType } from "../types.js";

// This test is opt-in because it makes billable OCI requests. It uses the
// standard OCI CLI variables so it can run from an already configured shell.
const compartmentId =
  process.env.OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID ??
  process.env.OCI_COMPARTMENT_ID;
const modelId =
  process.env.OCI_GENAI_INTEGRATION_TESTS_EMBEDDING_ON_DEMAND_MODEL_ID;

test.skipIf(!compartmentId || !modelId)(
  "OCI GenAI text embeddings",
  async () => {
    const embeddings = new OciGenAiEmbeddings({
      compartmentId: compartmentId!,
      onDemandModelId: modelId!,
      newClientParams: {
        authType: OciGenAiNewClientAuthType.ConfigFile,
        regionId: process.env.OCI_REGION ?? "us-phoenix-1",
        serviceEndpoint:
          process.env.OCI_ENDPOINT ??
          "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com",
        authParams:
          process.env.OCI_CONFIG_FILE || process.env.OCI_CONFIG_PROFILE
            ? {
                clientConfigFilePath: process.env.OCI_CONFIG_FILE ?? "",
                clientProfile: process.env.OCI_CONFIG_PROFILE ?? "DEFAULT",
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
    } finally {
      await embeddings.close();
    }
  },
  100_000
);
