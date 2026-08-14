import {
  AIMessageChunk,
  BaseMessage,
  ToolMessage as LangChainToolMessage,
} from "@langchain/core/messages";
import {
  LangSmithParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { RunnableBinding, type Runnable } from "@langchain/core/runnables";

import { models } from "oci-generativeaiinference";

import {
  OciGenAiBaseChat,
  type OciGenAiParsedResponse,
  type OciGenAiStreamChunk,
} from "./chat_models.js";
import type { OciGenAiModelCallOptions } from "./types.js";

const {
  AssistantMessage,
  GenericChatRequest,
  SystemMessage,
  TextContent,
  ToolMessage,
  UserMessage,
} = models;
type GenericChatRequest = models.GenericChatRequest;
type GenericChatResponse = models.GenericChatResponse;
type Message = models.Message;
type TextContent = models.TextContent;
type ChatChoice = models.ChatChoice;
type ToolMessage = models.ToolMessage;

export type GenericCallOptions = Omit<
  GenericChatRequest,
  "apiFormat" | "messages" | "isStream" | "stop"
>;

/** OCI Generic chat model, including LangChain tool-call and tool-result turns. */
export class OciGenAiGenericChat extends OciGenAiBaseChat<GenericCallOptions> {
  override _createRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): GenericChatRequest {
    const requestParams = options.requestParams ?? {};
    return <GenericChatRequest>{
      apiFormat: GenericChatRequest.apiFormat,
      messages:
        OciGenAiGenericChat._convertBaseMessagesToGenericMessages(messages),
      ...requestParams,
      isStream: !!stream,
      stop: options.stop,
    };
  }

  override _parseResponse(
    response: GenericChatResponse
  ): OciGenAiParsedResponse {
    if (!OciGenAiGenericChat._isGenericResponse(response)) {
      throw new Error("Invalid GenericChatResponse object");
    }

    const choice = response.choices[0];
    const content = OciGenAiGenericChat._getChunkDataText(choice) ?? "";
    const toolCalls = OciGenAiGenericChat._getToolCalls(choice);

    return {
      content,
      toolCalls,
      usageMetadata: OciGenAiBaseChat._toUsageMetadata(
        choice.usage ?? response.usage
      ),
      responseMetadata: {
        finish_reason: choice.finishReason,
        service_tier: response.serviceTier,
      },
    };
  }

  override _parseStreamedResponseChunk(
    chunk: unknown
  ): OciGenAiStreamChunk | undefined {
    if (!OciGenAiGenericChat._isValidStreamChoice(chunk)) {
      throw new Error("Invalid streamed response chunk data");
    }

    const choice = chunk as ChatChoice;
    const toolCallChunks = choice.message
      ? OciGenAiGenericChat._getToolCallChunks(choice)
      : [];

    return {
      text: OciGenAiGenericChat._getChunkDataText(choice),
      ...(typeof choice.finishReason === "string"
        ? { finishReason: choice.finishReason }
        : {}),
      ...(toolCallChunks.length > 0 ? { toolCallChunks } : {}),
      ...(choice.usage
        ? { usageMetadata: OciGenAiBaseChat._toUsageMetadata(choice.usage) }
        : {}),
    };
  }

  static _convertBaseMessagesToGenericMessages(
    messages: BaseMessage[]
  ): Message[] {
    // OCI requires every tool result to refer to an earlier assistant tool call.
    // Tracking IDs here prevents malformed agent histories reaching the service.
    const outstandingToolCallIds = new Set<string>();

    return messages.map((message) => {
      if (message.getType() === "ai") {
        for (const toolCall of (
          message as { tool_calls?: Array<{ id?: string }> }
        ).tool_calls ?? []) {
          if (toolCall.id) {
            outstandingToolCallIds.add(toolCall.id);
          }
        }
      }

      if (message.getType() === "tool") {
        const toolCallId = (message as LangChainToolMessage).tool_call_id;
        if (!toolCallId || !outstandingToolCallIds.has(toolCallId)) {
          throw new Error(
            `ToolMessage references unknown tool call '${toolCallId ?? ""}'`
          );
        }
      }

      return this._convertBaseMessageToGenericMessage(message);
    });
  }

  static _convertBaseMessageToGenericMessage(
    baseMessage: BaseMessage
  ): Message {
    const messageType: string = baseMessage.getType();
    const text = OciGenAiBaseChat._contentToText(baseMessage.content);

    switch (messageType) {
      case "ai":
        return OciGenAiGenericChat._createAssistantMessage(baseMessage, text);

      case "tool": {
        const toolMessage = baseMessage as LangChainToolMessage;
        return <ToolMessage>{
          role: ToolMessage.role,
          toolCallId: toolMessage.tool_call_id,
          content: OciGenAiGenericChat._createTextContent(text),
        };
      }

      case "system":
        return OciGenAiGenericChat._createMessage(SystemMessage.role, text);

      case "human":
        return OciGenAiGenericChat._createMessage(UserMessage.role, text);

      default:
        throw new Error(`Message type '${messageType}' is not supported`);
    }
  }

  static _createAssistantMessage(
    baseMessage: BaseMessage,
    text: string
  ): Message {
    const toolCalls =
      (
        baseMessage as {
          tool_calls?: Array<{ id?: string; name: string; args: unknown }>;
        }
      ).tool_calls ?? [];
    return {
      role: AssistantMessage.role,
      ...(toolCalls.length > 0
        ? {
            toolCalls: toolCalls.map((toolCall, index) => ({
              id: toolCall.id ?? `langchain-tool-call-${index}`,
              type: "FUNCTION",
              name: toolCall.name,
              arguments: JSON.stringify(toolCall.args ?? {}),
            })),
          }
        : { content: OciGenAiGenericChat._createTextContent(text) }),
    } as Message;
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
      (OciGenAiGenericChat._isValidContentArray((<Message>message).content) ||
        Array.isArray((message as { toolCalls?: unknown }).toolCalls))
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

  static _getToolCalls(chunkData: ChatChoice) {
    const toolCalls =
      (
        chunkData.message as
          | {
              toolCalls?: Array<{
                id?: string;
                name?: string;
                arguments?: string;
              }>;
            }
          | undefined
      )?.toolCalls ?? [];
    return toolCalls
      .filter((toolCall) => typeof toolCall.name === "string")
      .map((toolCall, index) =>
        OciGenAiBaseChat._toolCall(
          toolCall.name as string,
          toolCall.arguments,
          toolCall.id ?? `oci-tool-call-${index}`
        )
      );
  }

  static _getToolCallChunks(chunkData: ChatChoice) {
    const toolCalls =
      (
        chunkData.message as
          | {
              toolCalls?: Array<{
                id?: string;
                name?: string;
                arguments?: string;
              }>;
            }
          | undefined
      )?.toolCalls ?? [];
    return toolCalls.map((toolCall, index) =>
      OciGenAiBaseChat._toolCallChunk(
        toolCall.name,
        toolCall.arguments,
        toolCall.id,
        index
      )
    );
  }

  bindTools(
    tools: BindToolsInput[],
    kwargs: Partial<this["ParsedCallOptions"]> = {}
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk,
    OciGenAiModelCallOptions<GenericCallOptions>
  > {
    // LangChain tools use the OpenAI-compatible schema; OCI Generic function
    // definitions use the same JSON Schema payload with provider field names.
    return new RunnableBinding({
      bound: this,
      kwargs: {
        ...kwargs,
        requestParams: {
          ...(kwargs.requestParams ?? {}),
          tools: OciGenAiGenericChat._convertTools(
            tools.map(convertToOpenAITool)
          ),
        },
      },
      config: {},
    });
  }

  static _convertTools(
    tools: ReturnType<typeof convertToOpenAITool>[]
  ): models.FunctionDefinition[] {
    return tools.map((tool) => ({
      type: models.FunctionDefinition.type,
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
    }));
  }

  static _isFinalChunk(chunkData: unknown) {
    return (
      chunkData !== null &&
      typeof chunkData === "object" &&
      typeof (<ChatChoice>chunkData).finishReason === "string"
    );
  }

  static _isValidStreamChoice(chunk: unknown): boolean {
    if (chunk === null || typeof chunk !== "object") {
      return false;
    }

    const candidate = chunk as Partial<ChatChoice>;
    return (
      (candidate.message !== undefined &&
        OciGenAiGenericChat._isValidMessage(candidate.message)) ||
      candidate.finishReason !== undefined ||
      candidate.usage !== undefined
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
