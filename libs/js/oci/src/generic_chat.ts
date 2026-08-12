import { BaseMessage } from "@langchain/core/messages";
import { LangSmithParams } from "@langchain/core/language_models/chat_models";

import { models } from "oci-generativeaiinference";

import { OciGenAiBaseChat, type OciGenAiStreamChunk } from "./chat_models.js";

const {
  AssistantMessage,
  GenericChatRequest,
  SystemMessage,
  TextContent,
  UserMessage,
} = models;
type GenericChatRequest = models.GenericChatRequest;
type GenericChatResponse = models.GenericChatResponse;
type Message = models.Message;
type TextContent = models.TextContent;
type ChatChoice = models.ChatChoice;
type ChatContent = models.ChatContent;

export type GenericCallOptions = Omit<
  GenericChatRequest,
  "apiFormat" | "messages" | "isStream" | "stop"
>;

export class OciGenAiGenericChat extends OciGenAiBaseChat<GenericCallOptions> {
  override _createRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): GenericChatRequest {
    return <GenericChatRequest>{
      apiFormat: GenericChatRequest.apiFormat,
      messages:
        OciGenAiGenericChat._convertBaseMessagesToGenericMessages(messages),
      ...options.requestParams,
      isStream: !!stream,
      stop: options.stop,
    };
  }

  override _parseResponse(response: GenericChatResponse): string {
    if (!OciGenAiGenericChat._isGenericResponse(response)) {
      throw new Error("Invalid GenericChatResponse object");
    }

    return response.choices
      ?.map((choice: ChatChoice) =>
        choice.message.content
          ?.map((content: ChatContent) => (<TextContent>content).text)
          .join("")
      )
      .join("");
  }

  override _parseStreamedResponseChunk(
    chunk: unknown
  ): OciGenAiStreamChunk | undefined {
    if (!OciGenAiGenericChat._isValidChatChoice(chunk)) {
      throw new Error("Invalid streamed response chunk data");
    }

    if (OciGenAiGenericChat._isFinalChunk(chunk)) {
      return { finishReason: chunk.finishReason };
    }

    return { text: OciGenAiGenericChat._getChunkDataText(chunk) };
  }

  static _convertBaseMessagesToGenericMessages(
    messages: BaseMessage[]
  ): Message[] {
    return messages.map(this._convertBaseMessageToGenericMessage);
  }

  static _convertBaseMessageToGenericMessage(
    baseMessage: BaseMessage
  ): Message {
    const messageType: string = baseMessage.getType();
    const text = OciGenAiBaseChat._contentToText(baseMessage.content);
    const messageRole: string =
      OciGenAiGenericChat._convertBaseMessageTypeToRole(messageType);

    return OciGenAiGenericChat._createMessage(messageRole, text);
  }

  static _convertBaseMessageTypeToRole(baseMessageType: string): string {
    switch (baseMessageType) {
      case "ai":
        return AssistantMessage.role;

      case "system":
        return SystemMessage.role;

      case "human":
        return UserMessage.role;

      default:
        throw new Error(`Message type '${baseMessageType}' is not supported`);
    }
  }

  static _createMessage(role: string, text: string): Message {
    return {
      role,
      content: OciGenAiGenericChat._createTextContent(text),
    };
  }

  static _createTextContent(text: string): TextContent[] {
    return [
      {
        type: TextContent.type,
        text,
      },
    ];
  }

  static _isGenericResponse(
    response: unknown
  ): response is GenericChatResponse {
    return (
      response !== null &&
      typeof response === "object" &&
      this._isValidChoicesArray((<GenericChatResponse>response).choices)
    );
  }

  static _isValidChoicesArray(choices: unknown): choices is ChatChoice[] {
    return (
      Array.isArray(choices) &&
      choices.every(OciGenAiGenericChat._isValidChatChoice)
    );
  }

  static _isValidChatChoice(choice: unknown): choice is ChatChoice {
    return (
      choice !== null &&
      typeof choice === "object" &&
      (OciGenAiGenericChat._isValidMessage((<ChatChoice>choice).message) ||
        OciGenAiGenericChat._isFinalChunk(choice))
    );
  }

  static _isValidMessage(message: unknown): message is Message {
    return (
      message !== null &&
      typeof message === "object" &&
      OciGenAiGenericChat._isValidContentArray((<Message>message).content)
    );
  }

  static _isValidContentArray(content: TextContent[] | undefined): boolean {
    return (
      Array.isArray(content) &&
      content.every(OciGenAiGenericChat._isValidTextContent)
    );
  }

  static _isValidTextContent(content: unknown): content is TextContent {
    return (
      content !== null &&
      typeof content === "object" &&
      (<TextContent>content).type === TextContent.type &&
      typeof (<TextContent>content).text === "string"
    );
  }

  static _getChunkDataText(chunkData: ChatChoice): string | undefined {
    // Match non-streaming response parsing: OCI content parts are contiguous.
    return chunkData.message?.content
      ?.map((message: TextContent) => message.text)
      .join("");
  }

  static _isFinalChunk(chunkData: unknown) {
    return (
      chunkData !== null &&
      typeof chunkData === "object" &&
      typeof (<ChatChoice>chunkData).finishReason === "string"
    );
  }

  override getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    return {
      ls_provider: "oci_genai_generic",
      ls_model_name:
        this._params.onDemandModelId || this._params.dedicatedEndpointId || "",
      ls_model_type: "chat",
      ls_temperature: options.requestParams?.temperature || 0,
      ls_max_tokens: options.requestParams?.maxTokens || 0,
      ls_stop: options.stop || [],
    };
  }
}
