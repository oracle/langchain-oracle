import { AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import { SimpleChatModel } from "@langchain/core/language_models/chat_models";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

import {
  models,
  type requests,
  type responses,
} from "oci-generativeaiinference";
import type {
  OciGenAiChatCallResponseType,
  OciGenAiModelBaseParams,
  OciGenAiModelCallOptions,
  OciGenAiSupportedRequestType,
  OciGenAiSupportedResponseType,
} from "./types.js";

import { OciGenAiSdkClient } from "./oci_genai_sdk_client.js";
import { JsonServerEventsIterator } from "./server_events_iterator.js";

const { DedicatedServingMode, OnDemandServingMode } = models;
type DedicatedServingMode = models.DedicatedServingMode;
type OnDemandServingMode = models.OnDemandServingMode;

export interface OciGenAiStreamChunk {
  text?: string;
  finishReason?: string;
}

export abstract class OciGenAiBaseChat<RequestType> extends SimpleChatModel<
  OciGenAiModelCallOptions<RequestType>
> {
  _sdkClient: OciGenAiSdkClient | undefined;

  // A caller-injected SDK client remains caller-owned and must not be closed.
  _ownsSdkClient = false;

  _params: Partial<OciGenAiModelBaseParams>;

  constructor(params?: Partial<OciGenAiModelBaseParams>) {
    super(params ?? {});
    this._params = params ?? {};
  }

  abstract _createRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): OciGenAiSupportedRequestType;

  abstract _parseResponse(
    response: OciGenAiSupportedResponseType | undefined
  ): string;

  abstract _parseStreamedResponseChunk(
    chunk: unknown
  ): OciGenAiStreamChunk | undefined;

  async _call(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<string> {
    const response: responses.ChatResponse = await this._makeRequest(
      messages,
      options
    );
    // The OCI SDK's ChatResult union includes Cohere V2 responses, but this
    // integration only sends the V1 Cohere request format or the generic
    // format, whose response types are represented by this base class.
    return this._parseResponse(
      response?.chatResult?.chatResponse as OciGenAiSupportedResponseType
    );
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const response: ReadableStream<Uint8Array> = await this._makeRequest(
      messages,
      options,
      true
    );
    const responseChunkIterator = new JsonServerEventsIterator(response);

    for await (const responseChunk of responseChunkIterator) {
      yield* this._streamResponseChunk(responseChunk, runManager);
    }
  }

  async *_streamResponseChunk(
    responseChunkData: unknown,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const parsedChunk = this._parseStreamedResponseChunk(responseChunkData);

    if (parsedChunk === undefined) {
      return;
    }

    const text = parsedChunk.text ?? "";
    // Preserve OCI terminal state even when its final SSE event has no text.
    yield this._createStreamResponse(text, parsedChunk.finishReason);
    if (text) {
      await runManager?.handleLLMNewToken(text);
    }
  }

  async _makeRequest<ResponseType>(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): Promise<ResponseType> {
    const request: OciGenAiSupportedRequestType = this._prepareRequest(
      messages,
      options,
      stream
    );
    await this._setupClient();
    return (await this._chat(request)) as ResponseType;
  }

  async _setupClient() {
    if (this._sdkClient) {
      return;
    }

    this._sdkClient = await OciGenAiSdkClient.create(this._params);
    this._ownsSdkClient = !this._params.client;
  }

  async close(): Promise<void> {
    if (this._sdkClient && this._ownsSdkClient) {
      this._sdkClient.close();
      this._sdkClient = undefined;
      this._ownsSdkClient = false;
    }
  }

  _createStreamResponse(text: string, finishReason?: string) {
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        content: text,
        response_metadata: finishReason
          ? { finish_reason: finishReason }
          : undefined,
      }),
      text,
    });
  }

  _prepareRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): OciGenAiSupportedRequestType {
    this._assertMessages(messages);
    return this._createRequest(messages, options, stream);
  }

  _assertMessages(messages: BaseMessage[]) {
    if (messages.length === 0) {
      throw new Error("No messages provided");
    }

    for (const message of messages) {
      OciGenAiBaseChat._contentToText(message.content);
    }
  }

  static _contentToText(content: BaseMessage["content"]): string {
    if (typeof content === "string") {
      return content;
    }

    if (Array.isArray(content)) {
      // LangChain v1 messages may represent text as content blocks. Ignore
      // non-text blocks rather than serializing provider-specific objects.
      const textBlocks = content.filter(
        (block): block is { type: "text"; text: string } =>
          typeof block === "object" &&
          block !== null &&
          "type" in block &&
          block.type === "text" &&
          "text" in block &&
          typeof block.text === "string"
      );

      if (textBlocks.length > 0) {
        return textBlocks.map((block) => block.text).join("");
      }
    }

    throw new Error("Unsupported message content");
  }

  async _chat(
    chatRequest: OciGenAiSupportedRequestType
  ): Promise<OciGenAiChatCallResponseType> {
    try {
      return await this._callChat(chatRequest);
    } catch (error) {
      throw new Error(
        `Error executing chat API, error: ${(<Error>error)?.message}`
      );
    }
  }

  async _callChat(
    chatRequest: OciGenAiSupportedRequestType
  ): Promise<OciGenAiChatCallResponseType> {
    if (!OciGenAiBaseChat._isSdkClient(this._sdkClient)) {
      throw new Error("OCI SDK client not initialized");
    }

    const fullChatRequest: requests.ChatRequest =
      this._composeFullRequest(chatRequest);
    return await this._sdkClient.client.chat(fullChatRequest);
  }

  _composeFullRequest(
    chatRequest: OciGenAiSupportedRequestType
  ): requests.ChatRequest {
    return {
      chatDetails: {
        chatRequest,
        compartmentId: this._getCompartmentId(),
        servingMode: this._getServingMode(),
      },
    };
  }

  static _isSdkClient(sdkClient: unknown): sdkClient is OciGenAiSdkClient {
    return (
      sdkClient !== null &&
      typeof sdkClient === "object" &&
      typeof (<OciGenAiSdkClient>sdkClient).client === "object"
    );
  }

  _getServingMode(): OnDemandServingMode | DedicatedServingMode {
    this._assertServingMode();

    if (typeof this._params?.onDemandModelId === "string") {
      return <OnDemandServingMode>{
        servingType: OnDemandServingMode.servingType,
        modelId: this._params.onDemandModelId,
      };
    }

    return <DedicatedServingMode>{
      servingType: DedicatedServingMode.servingType,
      endpointId: this._params.dedicatedEndpointId,
    };
  }

  _getCompartmentId(): string {
    if (!OciGenAiBaseChat._isValidString(this._params.compartmentId)) {
      throw new Error("Invalid compartmentId");
    }

    return this._params.compartmentId;
  }

  _assertServingMode() {
    const hasModelId = OciGenAiBaseChat._isValidString(
      this._params.onDemandModelId
    );
    const hasEndpointId = OciGenAiBaseChat._isValidString(
      this._params.dedicatedEndpointId
    );

    // OCI accepts one serving target per request; choosing one when both are
    // supplied would silently send a request to the wrong target.
    if (hasModelId === hasEndpointId) {
      throw new Error(
        "Exactly one of onDemandModelId or dedicatedEndpointId must be supplied"
      );
    }
  }

  static _isValidString(value: unknown): value is string {
    return typeof value === "string" && value.length > 0;
  }

  _llmType() {
    return "oci_genai";
  }
}
