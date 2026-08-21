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

  // Single-flight lazy initialization prevents concurrent invocations from
  // constructing multiple SDK clients, one of which could become orphaned.
  _sdkClientPromise: Promise<OciGenAiSdkClient> | undefined;

  // A caller-injected SDK client remains caller-owned and must not be closed.
  _ownsSdkClient = false;

  _closed = false;

  // Concurrent close() callers share the same cleanup, including any client
  // construction that was already in progress when shutdown began.
  _closePromise: Promise<void> | undefined;

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

  /**
   * Streams chat generation chunks incrementally from the OCI Generative AI service.
   *
   * @param messages - Array of LangChain chat history messages.
   * @param options - Provider call options (e.g., temperature, maxTokens, stop).
   * @param runManager - Optional callback manager to trigger stream events (e.g., handleLLMNewToken).
   * @returns An async generator yielding standardized LangChain `ChatGenerationChunk` instances.
   */
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    // Sends the HTTP POST request to the OCI GenAI service with stream: true.
    // response has ReadableStream<Uint8Array> yielding incoming
    // binary network packets over HTTP.
    const response: ReadableStream<Uint8Array> = await this._makeRequest(
      messages,
      options,
      true
    );

    // Initialize the Server-Sent Events (SSE) framing iterator.
    const responseChunkIterator = new JsonServerEventsIterator(response);

    // Iterate through incoming parsed SSE JSON events as they arrive over the wire.
    for await (const responseChunk of responseChunkIterator) {
      // Normalize provider-specific delta JSON into LangChain `ChatGenerationChunk`s,
      // invoke active tracer callbacks (e.g., LangSmith / token callbacks), and
      // delegate-yield the standardized chunks directly to the consumer.
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

  async _setupClient(): Promise<void> {
    if (this._closed) {
      throw new Error("OciGenAiBaseChat is closed");
    }

    if (this._sdkClient) {
      return;
    }

    if (!this._sdkClientPromise) {
      this._sdkClientPromise = OciGenAiSdkClient.create(this._params)
        .then((client) => {
          // close() can begin while asynchronous client construction is in
          // flight. Close an owned client rather than installing it afterward.
          if (this._closed) {
            if (!this._params.client) {
              client.close();
            }
            throw new Error("OciGenAiBaseChat is closed");
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
    await clientPromise;
  }

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
    if (clientPromise) {
      try {
        await clientPromise;
      } catch {
        // A closed in-flight client intentionally rejects after cleanup.
      }
    }

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

      // Only accept arrays consisting entirely of text blocks. Reject mixed or
      // non-text payloads (e.g. image_url, audio) early to prevent partial or
      // silent content drops until OCI multimodal support is explicitly implemented.
      if (textBlocks.length === content.length && textBlocks.length > 0) {
        return textBlocks.map((block) => block.text).join("");
      }
    }

    // Fail fast if content is empty or contains unsupported/multimodal block types.
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
    const sdkClient = this._sdkClient;
    if (!OciGenAiBaseChat._isSdkClient(sdkClient)) {
      throw new Error("OCI SDK client not initialized");
    }

    const fullChatRequest: requests.ChatRequest =
      this._composeFullRequest(chatRequest);
    // Delegate retries and instance-wide concurrency to LangChain's
    // AsyncCaller rather than duplicating that policy around the OCI SDK.
    return this.caller.call(() => sdkClient.client.chat(fullChatRequest));
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
