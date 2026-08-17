import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";
import { models, type responses } from "oci-generativeaiinference";

import { OciGenAiSdkClient } from "./oci_genai_sdk_client.js";
import type { OciGenAiClientParams, OciGenAiServingParams } from "./types.js";

const { DedicatedServingMode, OnDemandServingMode } = models;

/** Parameters for the OCI Generative AI text embeddings integration. */
export interface OciGenAiEmbeddingsParams
  extends EmbeddingsParams,
    OciGenAiClientParams,
    OciGenAiServingParams {
  /** OCID of the compartment authorized to use OCI Generative AI. */
  compartmentId: string;
  /** Maximum number of input strings sent in a single OCI request (1–96). */
  batchSize?: number;
  /** OCI behavior for an input longer than the model's token limit. */
  truncate?: models.EmbedTextDetails.Truncate;
  /** Optional OCI embedding purpose, such as SEARCH_DOCUMENT or SEARCH_QUERY. */
  inputType?: models.EmbedTextDetails.InputType;
  /** Optional output-vector dimension, supported by compatible OCI models. */
  outputDimensions?: number;
}

/**
 * LangChain text embeddings backed by OCI Generative AI's `embedText` API.
 *
 * This integration currently exposes text-only embeddings. OCI Embed 4 also
 * supports multimodal `embedContents`, which can be added in a later
 * provider-specific extension. It shares the chat integration's authenticated
 * SDK-client lifecycle.
 */
export class OciGenAiEmbeddings extends Embeddings {
  // OCI EmbedText supports at most 96 text inputs per request.
  static readonly _DEFAULT_BATCH_SIZE = 96;

  // Conservative default for OCI request concurrency. Callers can increase it
  // through EmbeddingsParams.maxConcurrency when service capacity permits.
  static readonly _DEFAULT_MAX_CONCURRENCY = 2;

  private _sdkClient: OciGenAiSdkClient | undefined;

  // Single-flight lazy initialization prevents concurrent calls from creating
  // multiple SDK clients, one of which would otherwise be unreachable to close.
  private _sdkClientPromise: Promise<OciGenAiSdkClient> | undefined;

  // A caller-injected SDK client may be shared and is never closed here.
  private _ownsSdkClient = false;

  private _closed = false;

  // Concurrent close() callers share one cleanup operation and therefore all
  // observe completion of an in-flight client initialization cleanup.
  private _closePromise: Promise<void> | undefined;

  private readonly _params: OciGenAiEmbeddingsParams;

  private readonly _batchSize: number;

  private readonly _maxConcurrency: number;

  constructor(params: OciGenAiEmbeddingsParams) {
    OciGenAiEmbeddings._validateParams(params);
    const maxConcurrency =
      params.maxConcurrency ?? OciGenAiEmbeddings._DEFAULT_MAX_CONCURRENCY;
    const batchSize =
      params.batchSize ?? OciGenAiEmbeddings._DEFAULT_BATCH_SIZE;
    super({ ...params, maxConcurrency });
    this._batchSize = batchSize;
    this._maxConcurrency = maxConcurrency;
    // Retain a top-level copy so later caller mutation cannot alter serving or
    // lifecycle behavior after the embeddings instance is constructed.
    this._params = { ...params, batchSize, maxConcurrency };
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    if (documents.length === 0) {
      return [];
    }

    const results = new Array<number[]>(documents.length);
    let nextStartIndex = 0;
    let workerFailed = false;
    const workerCount = Math.min(
      this._maxConcurrency,
      Math.ceil(documents.length / this._batchSize)
    );

    // The worker window bounds promises created by this invocation. The shared
    // LangChain AsyncCaller additionally enforces maxConcurrency across all
    // concurrent embedDocuments() and embedQuery() calls on this instance.
    // Each worker writes directly to the final result array by document index,
    // preserving order without retaining a batches array or flattening results.
    await Promise.all(
      Array.from({ length: workerCount }, async () => {
        while (nextStartIndex < documents.length && !workerFailed) {
          const startIndex = nextStartIndex;
          nextStartIndex += this._batchSize;
          const batch = documents.slice(
            startIndex,
            startIndex + this._batchSize
          );
          let batchEmbeddings: number[][];
          try {
            batchEmbeddings = await this._embedInputs(batch);
          } catch (error) {
            // OCI requests already in flight cannot be cancelled by the SDK,
            // but other workers must not start additional batches after this
            // invocation has failed.
            workerFailed = true;
            throw error;
          }

          if (workerFailed) {
            return;
          }

          for (let index = 0; index < batchEmbeddings.length; index += 1) {
            const embedding = batchEmbeddings[index];
            if (embedding === undefined) {
              throw new Error(
                `Missing embedding for document at index ${startIndex + index}`
              );
            }
            results[startIndex + index] = embedding;
          }
        }
      })
    );

    return results;
  }

  async embedQuery(text: string): Promise<number[]> {
    const embeddings = await this._embedInputs([text]);
    const embedding = embeddings[0];

    if (!embedding) {
      throw new Error("OCI embedding response did not contain a query vector");
    }

    return embedding;
  }

  /**
   * Shuts down an SDK client only when this integration created it.
   * Call close() after active embedding operations complete: closing the OCI
   * SDK client may interrupt requests that are already in flight.
   */
  async close(): Promise<void> {
    if (this._closePromise) {
      return this._closePromise;
    }

    this._closed = true;
    this._closePromise = this._closeClient();
    return this._closePromise;
  }

  private async _closeClient(): Promise<void> {
    const clientPromise = this._sdkClientPromise;

    // Await construction so close() does not resolve before a client created
    // in the background has either failed or been closed by _setupClient().
    if (clientPromise) {
      try {
        await clientPromise;
      } catch {
        // A closed in-flight client intentionally rejects setup after cleanup.
      }
    }

    const client = this._sdkClient;
    const ownsClient = this._ownsSdkClient;
    this._sdkClient = undefined;
    this._ownsSdkClient = false;

    if (client && ownsClient) {
      client.close();
    }
  }

  private async _embedInputs(inputs: string[]): Promise<number[][]> {
    const sdkClient = await this._setupClient();

    try {
      const response = await this.caller.call(() =>
        sdkClient.client.embedText({
          embedTextDetails: {
            inputs,
            compartmentId: this._params.compartmentId,
            servingMode: this._getServingMode(),
            truncate: this._params.truncate,
            inputType: this._params.inputType,
            outputDimensions: this._params.outputDimensions,
          },
        })
      );
      const embeddings = OciGenAiEmbeddings._parseResponse(response);

      if (embeddings.length !== inputs.length) {
        throw new Error(
          `OCI embedding response contained ${embeddings.length} vectors for ${inputs.length} inputs`
        );
      }

      return embeddings;
    } catch (error) {
      // Use a structural check because errors can originate from another JS
      // realm, and the package's lint rules intentionally prohibit instanceof.
      const message =
        error !== null &&
        typeof error === "object" &&
        "message" in error &&
        typeof error.message === "string"
          ? error.message
          : String(error);
      throw new Error(`OCI embedding request failed: ${message}`, {
        cause: error,
      });
    }
  }

  private async _setupClient(): Promise<OciGenAiSdkClient> {
    if (this._closed) {
      throw new Error("OciGenAiEmbeddings is closed");
    }

    if (this._sdkClient) {
      return this._sdkClient;
    }

    if (!this._sdkClientPromise) {
      this._sdkClientPromise = OciGenAiSdkClient.create(this._params)
        .then((client) => {
          // close() may run while the asynchronous client construction is in
          // flight. An owned client must be closed instead of resurrected.
          if (this._closed) {
            if (!this._params.client) {
              client.close();
            }
            throw new Error("OciGenAiEmbeddings is closed");
          }

          this._sdkClient = client;
          this._ownsSdkClient = !this._params.client;
          return client;
        })
        .finally(() => {
          this._sdkClientPromise = undefined;
        });
    }

    const clientPromise = this._sdkClientPromise;
    if (clientPromise === undefined) {
      throw new Error("OCI SDK client initialization was not started");
    }
    return clientPromise;
  }

  private _getServingMode():
    | models.DedicatedServingMode
    | models.OnDemandServingMode {
    if (this._params.dedicatedEndpointId) {
      return {
        servingType: DedicatedServingMode.servingType,
        endpointId: this._params.dedicatedEndpointId,
      };
    }

    const modelId = this._params.onDemandModelId;
    if (typeof modelId !== "string" || modelId.trim().length === 0) {
      throw new Error("Invalid onDemandModelId");
    }

    return {
      servingType: OnDemandServingMode.servingType,
      modelId,
    };
  }

  private static _parseResponse(
    response: responses.EmbedTextResponse
  ): number[][] {
    const embeddings = response.embedTextResult?.embeddings;

    if (
      !Array.isArray(embeddings) ||
      !embeddings.every(
        (embedding) =>
          Array.isArray(embedding) &&
          embedding.length > 0 &&
          embedding.every(
            (value) => typeof value === "number" && Number.isFinite(value)
          )
      )
    ) {
      throw new Error("OCI embedding response contained invalid embeddings");
    }

    return embeddings;
  }

  private static _validateParams(params: OciGenAiEmbeddingsParams): void {
    if (
      typeof params.compartmentId !== "string" ||
      !params.compartmentId.trim()
    ) {
      throw new Error("compartmentId must be a non-empty string");
    }

    const hasModelId =
      typeof params.onDemandModelId === "string" &&
      params.onDemandModelId.trim().length > 0;
    const hasEndpointId =
      typeof params.dedicatedEndpointId === "string" &&
      params.dedicatedEndpointId.trim().length > 0;
    if (hasModelId === hasEndpointId) {
      throw new Error(
        "Exactly one of onDemandModelId or dedicatedEndpointId must be provided"
      );
    }

    if (
      params.batchSize !== undefined &&
      (!Number.isInteger(params.batchSize) ||
        params.batchSize < 1 ||
        params.batchSize > OciGenAiEmbeddings._DEFAULT_BATCH_SIZE)
    ) {
      throw new Error("batchSize must be an integer between 1 and 96");
    }

    if (
      params.maxConcurrency !== undefined &&
      (!Number.isInteger(params.maxConcurrency) || params.maxConcurrency < 1)
    ) {
      throw new Error("maxConcurrency must be a positive integer");
    }
  }
}
