import { expect, test, vi } from "vitest";

import {
  models,
  type GenerativeAiInferenceClient,
} from "oci-generativeaiinference";

import { OciGenAiEmbeddings } from "../embeddings.js";
import { OciGenAiSdkClient } from "../oci_genai_sdk_client.js";

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

function createOciError(statusCode: number, message: string): Error {
  return Object.assign(new Error(message), { statusCode });
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

test("OciGenAiEmbeddings splits 97 documents into 96-input OCI batches", async () => {
  const client = createClient();
  client.embedText.mockImplementation((request) => {
    const inputs: string[] = request.embedTextDetails.inputs ?? [];
    return Promise.resolve({
      embedTextResult: {
        embeddings: inputs.map((input) => [Number(input.substring(4))]),
      },
    });
  });
  const documents = Array.from({ length: 97 }, (_, index) => `doc-${index}`);

  await expect(
    createEmbeddings(client).embedDocuments(documents)
  ).resolves.toEqual(documents.map((_, index) => [index]));
  expect(client.embedText).toHaveBeenCalledTimes(2);
  expect(
    client.embedText.mock.calls[0]?.[0].embedTextDetails.inputs
  ).toHaveLength(96);
  expect(
    client.embedText.mock.calls[1]?.[0].embedTextDetails.inputs
  ).toHaveLength(1);
});

test("OciGenAiEmbeddings preserves order when batches complete out of order", async () => {
  const client = createClient();
  const pending: Array<
    (response: { embedTextResult: { embeddings: number[][] } }) => void
  > = [];
  client.embedText.mockImplementation(
    () =>
      new Promise((resolve) => {
        pending.push(resolve);
      })
  );
  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
    batchSize: 2,
    maxConcurrency: 2,
  });

  const result = embeddings.embedDocuments(["one", "two", "three"]);
  await vi.waitFor(() => expect(pending).toHaveLength(2));
  pending[1]!({ embedTextResult: { embeddings: [[3]] } });
  pending[0]!({ embedTextResult: { embeddings: [[1], [2]] } });

  await expect(result).resolves.toEqual([[1], [2], [3]]);
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
  expect(
    () =>
      new OciGenAiEmbeddings({
        ...common,
        onDemandModelId: "model",
        maxConcurrency: 0,
      })
  ).toThrow("maxConcurrency must be a positive integer");
});

test("OciGenAiEmbeddings rejects malformed OCI responses", async () => {
  const client = createClient();
  client.embedText.mockResolvedValue({
    embedTextResult: { embeddings: [["bad"]] },
  });

  await expect(createEmbeddings(client).embedQuery("test")).rejects.toThrow(
    "OCI embedding response contained invalid embeddings"
  );
});

test("OciGenAiEmbeddings rejects empty and non-finite OCI vectors", async () => {
  const client = createClient();
  client.embedText.mockResolvedValueOnce({
    embedTextResult: { embeddings: [[]] },
  });
  client.embedText.mockResolvedValueOnce({
    embedTextResult: { embeddings: [[Number.NaN]] },
  });
  client.embedText.mockResolvedValueOnce({
    embedTextResult: { embeddings: [[Number.POSITIVE_INFINITY]] },
  });
  const embeddings = createEmbeddings(client);

  await expect(embeddings.embedQuery("empty vector")).rejects.toThrow(
    "OCI embedding response contained invalid embeddings"
  );
  await expect(embeddings.embedQuery("non-finite vector")).rejects.toThrow(
    "OCI embedding response contained invalid embeddings"
  );
  await expect(embeddings.embedQuery("infinite vector")).rejects.toThrow(
    "OCI embedding response contained invalid embeddings"
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
    "OCI embedding request failed: OCI unavailable"
  );
});

test("OciGenAiEmbeddings retries retryable OCI failures up to maxRetries", async () => {
  vi.useFakeTimers();
  try {
    const client = createClient([[1]]);
    client.embedText
      .mockRejectedValueOnce(createOciError(500, "OCI service unavailable"))
      .mockResolvedValueOnce({ embedTextResult: { embeddings: [[1]] } });
    const embeddings = new OciGenAiEmbeddings({
      client,
      compartmentId: "ocid1.compartment.oc1..example",
      onDemandModelId: "cohere.embed-v4.0",
      maxRetries: 1,
    });

    const result = embeddings.embedQuery("test");
    await vi.runAllTimersAsync();

    await expect(result).resolves.toEqual([1]);
    expect(client.embedText).toHaveBeenCalledTimes(2);
  } finally {
    vi.useRealTimers();
  }
});

test("OciGenAiEmbeddings does not retry non-retryable OCI failures", async () => {
  const client = createClient();
  client.embedText.mockRejectedValue(
    createOciError(401, "OCI authentication failed")
  );
  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
    maxRetries: 3,
  });

  await expect(embeddings.embedQuery("test")).rejects.toThrow(
    "OCI embedding request failed: OCI authentication failed"
  );
  expect(client.embedText).toHaveBeenCalledTimes(1);
});

test("OciGenAiEmbeddings bounds concurrent document batches", async () => {
  const client = createClient();
  const pending: Array<() => void> = [];
  let activeRequests = 0;
  let maximumActiveRequests = 0;
  client.embedText.mockImplementation((request) => {
    activeRequests += 1;
    maximumActiveRequests = Math.max(maximumActiveRequests, activeRequests);
    const inputs: string[] = request.embedTextDetails.inputs ?? [];
    return new Promise((resolve) => {
      pending.push(() => {
        activeRequests -= 1;
        resolve({
          embedTextResult: { embeddings: inputs.map((_, index) => [index]) },
        });
      });
    });
  });
  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
    batchSize: 1,
    maxConcurrency: 2,
  });

  const result = embeddings.embedDocuments(["one", "two", "three", "four"]);
  await vi.waitFor(() => expect(pending).toHaveLength(2));
  pending.splice(0).forEach((resolve) => resolve());
  await vi.waitFor(() => expect(pending).toHaveLength(2));
  pending.splice(0).forEach((resolve) => resolve());

  await expect(result).resolves.toEqual([[0], [0], [0], [0]]);
  expect(maximumActiveRequests).toBe(2);
});

test("OciGenAiEmbeddings stops scheduling batches after a worker fails", async () => {
  const client = createClient();
  const failure = new Error("OCI unavailable");
  let resolveSecondRequest:
    | ((response: { embedTextResult: { embeddings: number[][] } }) => void)
    | undefined;
  client.embedText.mockRejectedValueOnce(failure).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        resolveSecondRequest = resolve;
      })
  );
  const embeddings = new OciGenAiEmbeddings({
    client,
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
    batchSize: 1,
    maxConcurrency: 2,
    maxRetries: 0,
  });

  const result = embeddings.embedDocuments(["one", "two", "three"]);
  const rejectedResult = expect(result).rejects.toThrow(
    "OCI embedding request failed: OCI unavailable"
  );
  await vi.waitFor(() => expect(resolveSecondRequest).toBeDefined());

  await rejectedResult;
  resolveSecondRequest!({ embedTextResult: { embeddings: [[2]] } });
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

  // The second OCI request was already in flight, but its worker must not
  // claim the third batch after the first worker has failed.
  expect(client.embedText).toHaveBeenCalledTimes(2);
});

test("OciGenAiEmbeddings initializes one owned SDK client for concurrent calls", async () => {
  const client = createClient([[1]]);
  const sdkClient = { client, close: vi.fn() } as unknown as OciGenAiSdkClient;
  const createClientSpy = vi
    .spyOn(OciGenAiSdkClient, "create")
    .mockResolvedValue(sdkClient);
  const embeddings = new OciGenAiEmbeddings({
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
  });

  await expect(
    Promise.all(
      Array.from({ length: 20 }, (_, index) =>
        embeddings.embedQuery(`query-${index}`)
      )
    )
  ).resolves.toEqual(Array.from({ length: 20 }, () => [1]));
  expect(createClientSpy).toHaveBeenCalledTimes(1);
  await embeddings.close();
  expect(sdkClient.close).toHaveBeenCalledTimes(1);
});

test("OciGenAiEmbeddings rejects operations after close", async () => {
  const client = createClient([[1]]);
  const embeddings = createEmbeddings(client);

  await embeddings.close();

  await expect(embeddings.embedQuery("test")).rejects.toThrow(
    "OciGenAiEmbeddings is closed"
  );
  expect(client.embedText).not.toHaveBeenCalled();
});

test("OciGenAiEmbeddings closes an owned client created after close", async () => {
  const client = createClient([[1]]);
  const sdkClient = { client, close: vi.fn() } as unknown as OciGenAiSdkClient;
  let resolveClient: ((client: OciGenAiSdkClient) => void) | undefined;
  vi.spyOn(OciGenAiSdkClient, "create").mockReturnValue(
    new Promise((resolve) => {
      resolveClient = resolve;
    })
  );
  const embeddings = new OciGenAiEmbeddings({
    compartmentId: "ocid1.compartment.oc1..example",
    onDemandModelId: "cohere.embed-v4.0",
  });

  const request = embeddings.embedQuery("test");
  await vi.waitFor(() => expect(resolveClient).toBeDefined());
  const closePromise = embeddings.close();
  const concurrentClosePromise = embeddings.close();
  resolveClient!(sdkClient);

  await Promise.all([closePromise, concurrentClosePromise]);
  await expect(request).rejects.toThrow("OciGenAiEmbeddings is closed");
  expect(sdkClient.close).toHaveBeenCalledTimes(1);
  expect(client.embedText).not.toHaveBeenCalled();
});
