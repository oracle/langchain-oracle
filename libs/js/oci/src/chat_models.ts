import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  type ToolCall,
  type ToolCallChunk,
  type UsageMetadata,
} from "@langchain/core/messages";
import {
  type ChatGeneration,
  ChatGenerationChunk,
  type ChatResult,
} from "@langchain/core/outputs";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
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

/** Provider-neutral information extracted from one OCI streaming event. */
export interface OciGenAiStreamChunk {
  text?: string;
  finishReason?: string;
  toolCallChunks?: ToolCallChunk[];
  usageMetadata?: UsageMetadata;
}

/** Provider-neutral information extracted from a completed OCI chat response. */
export interface OciGenAiParsedResponse {
  content: string;
  toolCalls?: ToolCall[];
  usageMetadata?: UsageMetadata;
  responseMetadata?: Record<string, unknown>;
}

/**
 * Shared LangChain chat-model lifecycle for OCI chat APIs. Subclasses translate
 * between LangChain messages and an OCI-specific request/response format.
 */
export abstract class OciGenAiBaseChat<RequestType> extends BaseChatModel<
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
  ): OciGenAiParsedResponse;

  abstract _parseStreamedResponseChunk(
    chunk: unknown
  ): OciGenAiStreamChunk | undefined;

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    const response: responses.ChatResponse = await this._makeRequest(
      messages,
      options
    );
    const parsed = this._parseResponse(
      response?.chatResult?.chatResponse as OciGenAiSupportedResponseType
    );
    const message = new AIMessage({
      content: parsed.content,
      tool_calls: parsed.toolCalls ?? [],
      usage_metadata: parsed.usageMetadata,
      response_metadata: parsed.responseMetadata ?? {},
    });
    const generation: ChatGeneration = {
      message,
      text: parsed.content,
      generationInfo: parsed.responseMetadata ?? {},
    };

    return {
      generations: [generation],
      llmOutput: {
        ...(parsed.responseMetadata ?? {}),
        usage: parsed.usageMetadata,
      },
    };
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
    yield this._createStreamResponse(
      text,
      parsedChunk.finishReason,
      parsedChunk.toolCallChunks,
      parsedChunk.usageMetadata
    );
    if (text || parsedChunk.toolCallChunks?.length) {
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
    // Only close a client this model created; an injected SDK client may be
    // shared by the application and remains the caller's responsibility.
    if (this._sdkClient && this._ownsSdkClient) {
      this._sdkClient.close();
      this._sdkClient = undefined;
      this._ownsSdkClient = false;
    }
  }

  _createStreamResponse(
    text: string,
    finishReason?: string,
    toolCallChunks?: ToolCallChunk[],
    usageMetadata?: UsageMetadata
  ) {
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        content: text,
        tool_call_chunks: toolCallChunks ?? [],
        usage_metadata: usageMetadata,
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
      // This integration is intentionally text-only until OCI multimodal
      // conversion is implemented. Reject a mixed array instead of silently
      // dropping image, document, audio, video, or reasoning blocks.
      const textBlocks = content.filter(
        (block): block is { type: "text"; text: string } =>
          typeof block === "object" &&
          block !== null &&
          "type" in block &&
          block.type === "text" &&
          "text" in block &&
          typeof block.text === "string"
      );

      if (textBlocks.length === content.length && textBlocks.length > 0) {
        return textBlocks.map((block) => block.text).join("");
      }
    }

    throw new Error("Unsupported message content");
  }

  static _toUsageMetadata(
    usage: models.Usage | undefined
  ): UsageMetadata | undefined {
    if (!usage) {
      return undefined;
    }

    const inputTokens = usage.promptTokens ?? 0;
    const outputTokens = usage.completionTokens ?? 0;
    return {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: usage.totalTokens ?? inputTokens + outputTokens,
    };
  }

  static _toolCall(name: string, args: unknown, id: string): ToolCall {
    let parsedArgs: unknown = args ?? {};
    if (typeof parsedArgs === "string") {
      try {
        parsedArgs = JSON.parse(parsedArgs);
      } catch {
        // OCI can return malformed tool arguments. Preserve the tool call so
        // an agent can still surface it, using an empty object as safe args.
        parsedArgs = {};
      }
    }

    return {
      type: "tool_call",
      name,
      args:
        parsedArgs !== null && typeof parsedArgs === "object" ? parsedArgs : {},
      id,
    };
  }

  static _toolCallChunk(
    name: string | undefined,
    args: string | undefined,
    id: string | undefined,
    index: number
  ): ToolCallChunk {
    return { type: "tool_call_chunk", name, args: args ?? "", id, index };
  }

  async _chat(
    chatRequest: OciGenAiSupportedRequestType
  ): Promise<OciGenAiChatCallResponseType> {
    try {
      return await this._callChat(chatRequest);
    } catch (error) {
      // Use a structural check because this package's lint rules prohibit
      // instanceof, and errors can originate from a separate JS realm.
      const message =
        error !== null &&
        typeof error === "object" &&
        "message" in error &&
        typeof error.message === "string"
          ? error.message
          : String(error);
      throw new Error(`Error executing chat API, error: ${message}`, {
        cause: error,
      });
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
      "client" in sdkClient &&
      typeof (sdkClient as OciGenAiSdkClient).client?.chat === "function"
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
