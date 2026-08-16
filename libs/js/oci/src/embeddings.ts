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
 * This first phase deliberately accepts strings only. It shares the chat
 * integration's authenticated SDK-client lifecycle while leaving OCI Embed v4
 * multimodal `embedContents` support for a later provider-specific extension.
 */
export class OciGenAiEmbeddings extends Embeddings {
  static readonly _DEFAULT_BATCH_SIZE = 96;

  _sdkClient: OciGenAiSdkClient | undefined;

  // A caller-injected SDK client may be shared and is never closed here.
  _ownsSdkClient = false;

  private _params: OciGenAiEmbeddingsParams;

  constructor(params: OciGenAiEmbeddingsParams) {
    super(params);
    OciGenAiEmbeddings._validateParams(params);
    this._params = params;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    if (documents.length === 0) {
      return [];
    }

    const embeddings: number[][] = [];
    const batchSize =
      this._params.batchSize ?? OciGenAiEmbeddings._DEFAULT_BATCH_SIZE;

    // Keep batches ordered so the returned vector index always matches the
    // corresponding LangChain document index.
    for (let start = 0; start < documents.length; start += batchSize) {
      const inputs = documents.slice(start, start + batchSize);
      embeddings.push(...(await this._embedInputs(inputs)));
    }

    return embeddings;
  }

  async embedQuery(document: string): Promise<number[]> {
    const embeddings = await this._embedInputs([document]);
    const embedding = embeddings[0];

    if (!embedding) {
      throw new Error("OCI embedding response did not contain a query vector");
    }

    return embedding;
  }

  /** Closes an SDK client only when this integration created it. */
  async close(): Promise<void> {
    if (this._sdkClient && this._ownsSdkClient) {
      this._sdkClient.close();
      this._sdkClient = undefined;
      this._ownsSdkClient = false;
    }
  }

  private async _embedInputs(inputs: string[]): Promise<number[][]> {
    await this._setupClient();

    try {
      const response = await this.caller.call(() =>
        this._sdkClient!.client.embedText({
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
      throw new Error(`Error executing embedding API, error: ${message}`, {
        cause: error,
      });
    }
  }

  private async _setupClient(): Promise<void> {
    if (this._sdkClient) {
      return;
    }

    this._sdkClient = await OciGenAiSdkClient.create(this._params);
    this._ownsSdkClient = !this._params.client;
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

    return {
      servingType: OnDemandServingMode.servingType,
      modelId: this._params.onDemandModelId!,
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
          embedding.every((value) => typeof value === "number")
      )
    ) {
      throw new Error(
        "OCI embedding response did not contain numeric embeddings"
      );
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
    const servingModeCount = Number(hasModelId) + Number(hasEndpointId);
    if (servingModeCount !== 1) {
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
  }
}
