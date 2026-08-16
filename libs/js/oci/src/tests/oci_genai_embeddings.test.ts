import { expect, test, vi } from "vitest";

import {
  models,
  type GenerativeAiInferenceClient,
} from "oci-generativeaiinference";

import { OciGenAiEmbeddings } from "../embeddings.js";

function createClient(
  embeddings: number[][] = [[1, 2]]
): GenerativeAiInferenceClient & { embedText: ReturnType<typeof vi.fn> } {
  return {
    embedText: vi.fn().mockResolvedValue({
      embedTextResult: { embeddings },
    }),
    close: vi.fn(),
  } as unknown as GenerativeAiInferenceClient & {
    embedText: ReturnType<typeof vi.fn>;
  };
}

function createEmbeddings(client = createClient()): OciGenAiEmbeddings {
  return new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
  });
}

test("OciGenAiEmbeddings batches documents and preserves their order", async () => {
  const client = createClient();
  client.embedText
    .mockResolvedValueOnce({ embedTextResult: { embeddings: [[1], [2]] } })
    .mockResolvedValueOnce({ embedTextResult: { embeddings: [[3]] } });
  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
    batchSize: 2,
    truncate: models.EmbedTextDetails.Truncate.End,
    inputType: models.EmbedTextDetails.InputType.SearchDocument,
    outputDimensions: 1024,
  });

  await expect(
    embeddings.embedDocuments(["one", "two", "three"])
  ).resolves.toEqual([[1], [2], [3]]);
  expect(client.embedText).toHaveBeenCalledTimes(2);
  expect(client.embedText).toHaveBeenNthCalledWith(1, {
    embedTextDetails: expect.objectContaining({
      inputs: ["one", "two"],
      compartmentId: "ocid1.compartment.oc1..example",
      inputType: "SEARCH_DOCUMENT",
      outputDimensions: 1024,
      truncate: "END",
      servingMode: expect.objectContaining({
        modelId: "cohere.embed-v4.0",
        servingType: "ON_DEMAND",
      }),
    }),
  });
  expect(client.embedText).toHaveBeenNthCalledWith(2, {
    embedTextDetails: expect.objectContaining({ inputs: ["three"] }),
  });
});

test("OciGenAiEmbeddings embeds one query and supports dedicated serving", async () => {
  const client = createClient([[0.1, 0.2, 0.3]]);
  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    dedicatedEndpointId: "ocid1.generativeaiendpoint.oc1..example",
  });

  await expect(
    embeddings.embedQuery("where is the document?")
  ).resolves.toEqual([0.1, 0.2, 0.3]);
  expect(client.embedText).toHaveBeenCalledWith({
    embedTextDetails: expect.objectContaining({
      inputs: ["where is the document?"],
      servingMode: expect.objectContaining({
        endpointId: "ocid1.generativeaiendpoint.oc1..example",
        servingType: "DEDICATED",
      }),
    }),
  });
});

test("OciGenAiEmbeddings does not call OCI for an empty document array", async () => {
  const client = createClient();

  await expect(createEmbeddings(client).embedDocuments([])).resolves.toEqual(
    []
  );
  expect(client.embedText).not.toHaveBeenCalled();
});

test("OciGenAiEmbeddings does not close a caller-owned SDK client", async () => {
  const client = createClient([[1]]);
  const embeddings = createEmbeddings(client);

  await embeddings.embedQuery("test");
  await embeddings.close();

  expect(client.close).not.toHaveBeenCalled();
});

test("OciGenAiEmbeddings validates serving target and batch size", () => {
  const common = { compartmentId: "ocid1.compartment.oc1..example" };

  expect(() => new OciGenAiEmbeddings(common)).toThrow(
    "Exactly one of onDemandModelId or dedicatedEndpointId must be provided"
  );
  expect(
    () =>
      new OciGenAiEmbeddings({
        ...common,
        onDemandModelId: "model",
        dedicatedEndpointId: "endpoint",
      })
  ).toThrow(
    "Exactly one of onDemandModelId or dedicatedEndpointId must be provided"
  );
  expect(
    () =>
      new OciGenAiEmbeddings({
        ...common,
        onDemandModelId: "model",
        batchSize: 97,
      })
  ).toThrow("batchSize must be an integer between 1 and 96");
});

test("OciGenAiEmbeddings rejects malformed OCI responses", async () => {
  const client = createClient();
  client.embedText.mockResolvedValue({
    embedTextResult: { embeddings: [["bad"]] },
  });

  await expect(createEmbeddings(client).embedQuery("test")).rejects.toThrow(
    "OCI embedding response did not contain numeric embeddings"
  );
});

test("OciGenAiEmbeddings rejects a response with the wrong number of vectors", async () => {
  const client = createClient([[1], [2]]);

  await expect(createEmbeddings(client).embedQuery("test")).rejects.toThrow(
    "OCI embedding response contained 2 vectors for 1 inputs"
  );
});

test("OciGenAiEmbeddings preserves the cause of an OCI error", async () => {
  const client = createClient();
  const cause = new Error("OCI unavailable");
  client.embedText.mockRejectedValue(cause);

  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
    maxRetries: 0,
  });

  await expect(embeddings.embedQuery("test")).rejects.toThrow(
    "Error executing embedding API, error: OCI unavailable"
  );
});
