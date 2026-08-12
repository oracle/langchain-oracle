import { models } from "oci-generativeaiinference";

import { BaseMessage } from "@langchain/core/messages";
import { LangSmithParams } from "@langchain/core/language_models/chat_models";
import { OciGenAiBaseChat, type OciGenAiStreamChunk } from "./chat_models.js";

const {
  CohereChatBotMessage,
  CohereChatRequest,
  CohereSystemMessage,
  CohereUserMessage,
} = models;
type CohereChatRequest = models.CohereChatRequest;
type CohereChatResponse = models.CohereChatResponse;
type CohereMessage = models.CohereMessage;
type CohereChatBotMessage = models.CohereChatBotMessage;
type CohereSystemMessage = models.CohereSystemMessage;
type CohereUserMessage = models.CohereUserMessage;

interface HistoryMessageInfo {
  chatHistory: CohereMessage[];
  message: string;
}

interface CohereStreamedResponseChunkData {
  apiFormat: string;
  text: string;
}

export type CohereCallOptions = Omit<
  CohereChatRequest,
  "apiFormat" | "message" | "chatHistory" | "isStream" | "stopSequences"
>;

export class OciGenAiCohereChat extends OciGenAiBaseChat<CohereCallOptions> {
  override _createRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): CohereChatRequest {
    const historyMessage: HistoryMessageInfo =
      OciGenAiCohereChat._splitMessageAndHistory(messages);

    return <CohereChatRequest>{
      apiFormat: CohereChatRequest.apiFormat,
      message: historyMessage.message,
      chatHistory: historyMessage.chatHistory,
      ...options.requestParams,
      isStream: !!stream,
      stopSequences: options.stop,
    };
  }

  override _parseResponse(response: CohereChatResponse | undefined): string {
    if (!OciGenAiCohereChat._isCohereResponse(response)) {
      throw new Error("Invalid CohereResponse object");
    }

    return response.text;
  }

  override _parseStreamedResponseChunk(chunk: unknown): OciGenAiStreamChunk {
    if (OciGenAiCohereChat._isCohereChunkData(chunk)) {
      return { text: chunk.text };
    }

    throw new Error("Invalid streamed response chunk data");
  }

  static _splitMessageAndHistory(messages: BaseMessage[]): HistoryMessageInfo {
    const currentMessage = messages.at(-1);
    // Cohere V1 separates the current user input from ordered chat history.
    // Promoting an earlier user turn would reorder later assistant/system turns.
    if (!currentMessage || currentMessage.getType() !== "human") {
      throw new Error(
        "Cohere chat requires the final message to be a human message"
      );
    }

    const cohereCurrentMessage = this._convertBaseMessageToCohereMessage(
      currentMessage
    ) as CohereUserMessage;

    return {
      chatHistory: messages
        .slice(0, -1)
        .map(this._convertBaseMessageToCohereMessage),
      message: cohereCurrentMessage.message,
    };
  }

  static _convertBaseMessageToCohereMessage(
    baseMessage: BaseMessage
  ): CohereMessage {
    const messageType: string = baseMessage.getType();
    const message = OciGenAiBaseChat._contentToText(baseMessage.content);

    switch (messageType) {
      case "ai":
        return <CohereChatBotMessage>{
          role: CohereChatBotMessage.role,
          message,
        };

      case "system":
        return <CohereSystemMessage>{
          role: CohereSystemMessage.role,
          message,
        };

      case "human":
        return <CohereUserMessage>{
          role: CohereUserMessage.role,
          message,
        };

      default:
        throw new Error(`Message type '${messageType}' is not supported`);
    }
  }

  static _isCohereResponse(response: unknown): response is CohereChatResponse {
    return (
      response !== null &&
      typeof response === "object" &&
      typeof (<CohereChatResponse>response).text === "string"
    );
  }

  static _isCohereChunkData(
    chunkData: unknown
  ): chunkData is CohereStreamedResponseChunkData {
    return (
      chunkData !== null &&
      typeof chunkData === "object" &&
      typeof (<CohereStreamedResponseChunkData>chunkData).text === "string" &&
      (<CohereStreamedResponseChunkData>chunkData).apiFormat ===
        CohereChatRequest.apiFormat
    );
  }

  override getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    return {
      ls_provider: "oci_genai_cohere",
      ls_model_name:
        this._params.onDemandModelId || this._params.dedicatedEndpointId || "",
      ls_model_type: "chat",
      ls_temperature: options.requestParams?.temperature || 0,
      ls_max_tokens: options.requestParams?.maxTokens || 0,
      ls_stop: options.stop || [],
    };
  }
}
