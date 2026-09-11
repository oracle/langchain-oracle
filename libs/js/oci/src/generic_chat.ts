import {
  AIMessageChunk,
  BaseMessage,
  ToolMessage as LangChainToolMessage,
} from "@langchain/core/messages";
import {
  LangSmithParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type {
  BaseLanguageModelInput,
  StructuredOutputMethodOptions,
} from "@langchain/core/language_models/base";
import {
  assembleStructuredOutputPipeline,
  createFunctionCallingParser,
} from "@langchain/core/language_models/structured_output";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { toJsonSchema } from "@langchain/core/utils/json_schema";
import {
  isSerializableSchema,
  type SerializableSchema,
} from "@langchain/core/utils/standard_schema";
import {
  getSchemaDescription,
  isInteropZodSchema,
  type InteropZodType,
} from "@langchain/core/utils/types";
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
  ToolChoiceAuto,
  ToolChoiceFunction,
  ToolChoiceNone,
  ToolChoiceRequired,
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

type OciGenAiNamedToolChoice = string & Record<never, never>;

/** Standard LangChain tool-choice forms accepted by Generic chat bindings. */
export type OciGenAiGenericToolChoice =
  | "auto"
  | "none"
  | "required"
  | "any"
  // Preserve autocomplete for the standard literals while allowing a bound
  // function name such as `tool_choice: "get_weather"`.
  | OciGenAiNamedToolChoice
  | boolean
  | { type: "function"; function: { name: string } };

type OciGenAiGenericBindToolsOptions = Partial<
  OciGenAiModelCallOptions<GenericCallOptions>
> & {
  tool_choice?: OciGenAiGenericToolChoice;
};

/** OCI Generic chat model, including LangChain tool-call and tool-result turns. */
export class OciGenAiGenericChat extends OciGenAiBaseChat<GenericCallOptions> {
  withStructuredOutput<
    RunOutput extends Record<string, unknown> = Record<string, unknown>
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      | SerializableSchema<RunOutput>
      | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<false>
  ): Runnable<BaseLanguageModelInput, RunOutput>;

  withStructuredOutput<
    RunOutput extends Record<string, unknown> = Record<string, unknown>
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      | SerializableSchema<RunOutput>
      | Record<string, unknown>,
    config: StructuredOutputMethodOptions<true>
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: RunOutput }>;

  override withStructuredOutput<
    RunOutput extends Record<string, unknown> = Record<string, unknown>
  >(
    outputSchema:
      | InteropZodType<RunOutput>
      | SerializableSchema<RunOutput>
      | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<boolean>
  ):
    | Runnable<BaseLanguageModelInput, RunOutput>
    | Runnable<
        BaseLanguageModelInput,
        { raw: BaseMessage; parsed: RunOutput }
      > {
    if (config?.strict) {
      throw new Error(
        '"strict" mode is not implemented by the OCI Generic chat adapter.'
      );
    }
    if (config?.method !== undefined && config.method !== "functionCalling") {
      throw new Error(
        `"${config.method}" is not implemented by the OCI Generic chat adapter; structured output currently uses function calling.`
      );
    }

    const functionName =
      config?.name ??
      (!isInteropZodSchema(outputSchema) &&
      !isSerializableSchema(outputSchema) &&
      typeof outputSchema.name === "string"
        ? outputSchema.name
        : "extract");
    const parameters =
      isInteropZodSchema(outputSchema) || isSerializableSchema(outputSchema)
        ? toJsonSchema(outputSchema)
        : outputSchema;
    const tools = [
      {
        type: "function" as const,
        function: {
          name: functionName,
          description:
            getSchemaDescription(outputSchema) ??
            "A function available to call.",
          parameters,
        },
      },
    ];
    // Force the generated extraction function: binding tools alone permits a
    // normal text response, which cannot satisfy a structured-output request.
    // The Core parser also validates Zod and Standard Schema results at runtime.
    const outputParser = createFunctionCallingParser(
      outputSchema,
      functionName
    );

    return assembleStructuredOutputPipeline(
      this.bindTools(tools, {
        tool_choice: {
          type: "function",
          function: { name: functionName },
        },
      }),
      outputParser,
      config?.includeRaw,
      config?.includeRaw ? "StructuredOutputRunnable" : "StructuredOutput"
    ) as
      | Runnable<BaseLanguageModelInput, RunOutput>
      | Runnable<
          BaseLanguageModelInput,
          { raw: BaseMessage; parsed: RunOutput }
        >;
  }

  override _createRequest(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream?: boolean
  ): GenericChatRequest {
    const requestParams = options.requestParams ?? {};
    return <GenericChatRequest>{
      // Keep provider tuning options, but do not allow untyped JS callers to
      // override the adapter-owned API format or converted message history.
      ...requestParams,
      apiFormat: GenericChatRequest.apiFormat,
      messages:
        OciGenAiGenericChat._convertBaseMessagesToGenericMessages(messages),
      isStream: !!stream,
      stop: options.stop,
    };
  }

  override _parseResponse(
    response: GenericChatResponse
  ): OciGenAiParsedResponse {
    // This JS adapter fails at the OCI boundary rather than converting an
    // unexpected response into an empty completion, as Python's defensive
    // provider path can do. A shape mismatch is actionable integration drift.
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
    // Keep the stream contract equally strict for unknown payloads; the
    // explicitly supported reasoning/role-only delta is handled below.
    if (!OciGenAiGenericChat._isValidStreamChoice(chunk)) {
      throw new Error("Invalid streamed response chunk data");
    }

    const choice = chunk as ChatChoice;
    const toolCallChunks = choice.message
      ? OciGenAiGenericChat._getToolCallChunks(choice)
      : [];

    const text = OciGenAiGenericChat._getChunkDataText(choice);
    const finishReason =
      typeof choice.finishReason === "string" ? choice.finishReason : undefined;
    const usageMetadata = choice.usage
      ? OciGenAiBaseChat._toUsageMetadata(choice.usage)
      : undefined;

    // Reasoning-only and role-only deltas carry no public ChatModel output.
    // OCI can emit them before visible text for reasoning-capable models.
    if (
      text === undefined &&
      toolCallChunks.length === 0 &&
      finishReason === undefined &&
      usageMetadata === undefined
    ) {
      return undefined;
    }

    return {
      text,
      ...(finishReason !== undefined ? { finishReason } : {}),
      ...(toolCallChunks.length > 0 ? { toolCallChunks } : {}),
      ...(usageMetadata !== undefined ? { usageMetadata } : {}),
    };
  }

  /**
   * Converts LangChain messages into OCI Generic message objects for the outgoing
   * model request, validating tool-call/tool-result relationships along the way.
   *
   * A ToolMessage must reference exactly one earlier model-generated tool-call ID.
   */
  static _convertBaseMessagesToGenericMessages(
    messages: BaseMessage[]
  ): Message[] {
    // OCI requires every tool result to refer to an earlier assistant tool call.
    // Unlike Python's best-effort history conversion, do not drop malformed
    // calls: their ID is the only safe LangChain agent-loop correlation key.
    const outstandingToolCallIds = new Set<string>();

    return messages.map((message) => {
      if (message.getType() === "ai") {
        for (const toolCall of (
          message as { tool_calls?: Array<{ id?: string }> }
        ).tool_calls ?? []) {
          if (toolCall.id) {
            if (outstandingToolCallIds.has(toolCall.id)) {
              throw new Error(`Duplicate tool call id '${toolCall.id}'`);
            }
            outstandingToolCallIds.add(toolCall.id);
          }
        }
      }

      if (message.getType() === "tool") {
        const toolCallId = (message as LangChainToolMessage).tool_call_id;
        // A tool-result turn has a one-to-one relationship with the unique
        // model-generated call ID. delete() validates and consumes it, so a
        // duplicate ToolMessage cannot silently reuse an earlier call.
        if (!toolCallId || !outstandingToolCallIds.delete(toolCallId)) {
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
    // Fails before the request because silently removing a call can orphan a later ToolMessage.
    for (const toolCall of toolCalls) {
      if (!toolCall.id) {
        throw new Error(
          `LangChain tool call '${toolCall.name}' did not contain a tool call id`
        );
      }
    }
    const content =
      text || toolCalls.length === 0
        ? { content: OciGenAiGenericChat._createTextContent(text) }
        : {};

    return {
      role: AssistantMessage.role,
      // OCI Generic supports assistant content alongside tool calls. Retain
      // non-empty text so an agent history round trip does not lose it.
      ...content,
      ...(toolCalls.length > 0
        ? {
            toolCalls: toolCalls.map((toolCall) => ({
              id: toolCall.id,
              type: "FUNCTION",
              name: toolCall.name,
              arguments: JSON.stringify(toolCall.args ?? {}),
            })),
          }
        : {}),
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
      this._isValidChoicesArray((<GenericChatResponse>response).choices) &&
      OciGenAiGenericChat._isValidOptionalUsage(
        (response as { usage?: unknown }).usage
      )
    );
  }

  static _isValidChoicesArray(choices: unknown): choices is ChatChoice[] {
    return (
      Array.isArray(choices) &&
      choices.length > 0 &&
      choices.every(OciGenAiGenericChat._isValidChatChoice)
    );
  }

  static _isValidChatChoice(choice: unknown): choice is ChatChoice {
    return (
      choice !== null &&
      typeof choice === "object" &&
      OciGenAiGenericChat._isValidOptionalFinishReason(
        (choice as { finishReason?: unknown }).finishReason
      ) &&
      OciGenAiGenericChat._isValidOptionalUsage(
        (choice as { usage?: unknown }).usage
      ) &&
      (OciGenAiGenericChat._isValidMessage((<ChatChoice>choice).message) ||
        OciGenAiGenericChat._isFinalChunk(choice))
    );
  }

  static _isValidMessage(message: unknown): message is Message {
    return (
      message !== null &&
      typeof message === "object" &&
      (OciGenAiGenericChat._isValidContentArray((<Message>message).content) ||
        OciGenAiGenericChat._isValidToolCalls(
          (message as { toolCalls?: unknown }).toolCalls
        ))
    );
  }

  static _isValidToolCalls(toolCalls: unknown): boolean {
    return (
      Array.isArray(toolCalls) &&
      toolCalls.length > 0 &&
      toolCalls.every(
        (toolCall) =>
          toolCall !== null &&
          typeof toolCall === "object" &&
          typeof (toolCall as { id?: unknown }).id === "string" &&
          (toolCall as { id: string }).id.length > 0 &&
          typeof (toolCall as { name?: unknown }).name === "string" &&
          (toolCall as { name: string }).name.length > 0 &&
          ((toolCall as { arguments?: unknown }).arguments === undefined ||
            typeof (toolCall as { arguments?: unknown }).arguments === "string")
      )
    );
  }

  static _isValidOptionalFinishReason(value: unknown): boolean {
    return value === undefined || typeof value === "string";
  }

  static _isValidOptionalUsage(value: unknown): boolean {
    if (value === undefined) {
      return true;
    }
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return false;
    }

    // OCI Usage token counters are optional, but every supplied counter must
    // be a finite number before it is surfaced as LangChain usage metadata.
    return ["promptTokens", "completionTokens", "totalTokens"].every(
      (field) => {
        const tokenCount = (value as Record<string, unknown>)[field];
        return (
          tokenCount === undefined ||
          (typeof tokenCount === "number" && Number.isFinite(tokenCount))
        );
      }
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
      .map((toolCall) => {
        // Completed OCI calls need a service-provided ID. Never synthesize one:
        // it must match the ToolMessage.tool_call_id sent in the next turn.
        if (typeof toolCall.id !== "string" || !toolCall.id) {
          throw new Error(
            `OCI tool call '${toolCall.name}' did not contain a tool call id`
          );
        }
        return OciGenAiBaseChat._toolCall(
          toolCall.name as string,
          toolCall.arguments,
          toolCall.id
        );
      });
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
    // Streaming deltas may omit the id and name after their first occurrence.
    // LangChain merges the fragments by index, reconstructing the completed
    // call (including the initial id) without inventing a correlation key.
    // This intentionally stays simpler than Python's provider-specific index
    // remapping for non-standard parallel-streaming implementations.
    return toolCalls.map((toolCall, index) =>
      OciGenAiBaseChat._toolCallChunk(
        toolCall.name,
        toolCall.arguments,
        toolCall.id,
        index
      )
    );
  }

  /**
   * Binds tool definitions (Zod schemas, structured tools, or raw JSON schemas)
   * to this chat model instance for function calling.
   *
   * @param tools - Array of LangChain tools, OpenAI-format tool definitions, or schemas to bind.
   * @param kwargs - Additional call options to attach (e.g., tool_choice, custom request parameters).
   * @returns A `RunnableBinding` wrapping this model with pre-configured OCI tool schemas.
   */
  bindTools(
    tools: BindToolsInput[],
    kwargs: OciGenAiGenericBindToolsOptions = {}
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk,
    OciGenAiModelCallOptions<GenericCallOptions>
  > {
    const { tool_choice: toolChoice, requestParams, ...callOptions } = kwargs;

    // Convert once to OCI-native representations. Besides avoiding duplicate
    // conversion below, this lets us validate a named tool choice against the
    // exact set of functions that will be sent to OCI.
    const ociTools = OciGenAiGenericChat._convertTools(
      tools.map(convertToOpenAITool)
    );
    const ociToolChoice =
      toolChoice === undefined
        ? undefined
        : OciGenAiGenericChat._convertToolChoice(toolChoice);

    // A function-specific tool choice must name one of the tools in this binding.
    // Fail locally with a clear error instead of sending inconsistent `tools` and
    // `toolChoice` values to OCI and relying on service-side validation.
    const functionName =
      ociToolChoice?.type === models.ToolChoiceFunction.type
        ? (ociToolChoice as models.ToolChoiceFunction).name
        : undefined;
    if (
      functionName !== undefined &&
      !ociTools.some((tool) => tool.name === functionName)
    ) {
      throw new Error(
        `tool_choice references unbound function '${functionName}'`
      );
    }

    // LangChain tools use the OpenAI-compatible schema; OCI Generic function
    // definitions use the same JSON Schema payload with provider field names.
    // Normalize standard tool_choice forms into OCI's requestParams.toolChoice
    // field rather than leaking an unsupported snake_case option downstream.
    return new RunnableBinding({
      bound: this,
      kwargs: {
        ...callOptions,
        requestParams: {
          ...(requestParams ?? {}),
          // Explicit LangChain tool_choice takes precedence over the raw OCI option.
          ...(ociToolChoice !== undefined ? { toolChoice: ociToolChoice } : {}),
          // bindTools owns the tool definitions sent with this binding.
          tools: ociTools,
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

  static _convertToolChoice(
    toolChoice: OciGenAiGenericToolChoice
  ):
    | models.ToolChoiceFunction
    | models.ToolChoiceNone
    | models.ToolChoiceAuto
    | models.ToolChoiceRequired {
    if (toolChoice === "auto") {
      return { type: ToolChoiceAuto.type };
    }
    if (toolChoice === "none" || toolChoice === false) {
      return { type: ToolChoiceNone.type };
    }
    if (
      toolChoice === "required" ||
      toolChoice === "any" ||
      toolChoice === true
    ) {
      return { type: ToolChoiceRequired.type };
    }
    if (typeof toolChoice === "string") {
      // Match Python's Generic provider: an otherwise unreserved string is a
      // request to call the named function.
      return { type: ToolChoiceFunction.type, name: toolChoice };
    }
    if (
      toolChoice.type === "function" &&
      typeof toolChoice.function?.name === "string" &&
      toolChoice.function.name.length > 0
    ) {
      return { type: ToolChoiceFunction.type, name: toolChoice.function.name };
    }

    throw new Error("Invalid tool_choice for OCI Generic chat");
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
      ((candidate.message !== undefined &&
        OciGenAiGenericChat._isValidStreamMessage(candidate.message)) ||
        candidate.finishReason !== undefined ||
        candidate.usage !== undefined) &&
      OciGenAiGenericChat._isValidOptionalFinishReason(
        candidate.finishReason
      ) &&
      OciGenAiGenericChat._isValidOptionalUsage(candidate.usage)
    );
  }

  static _isValidStreamMessage(message: unknown): message is Message {
    if (
      message !== null &&
      typeof message === "object" &&
      (OciGenAiGenericChat._isValidContentArray((message as Message).content) ||
        OciGenAiGenericChat._isValidStreamToolCalls(
          (message as { toolCalls?: unknown }).toolCalls
        ))
    ) {
      return true;
    }

    // Reasoning-capable OCI models can send a delta containing only a role
    // and/or reasoningContent before visible text or tool-call content.
    return (
      message !== null &&
      typeof message === "object" &&
      (typeof (message as { role?: unknown }).role === "string" ||
        typeof (message as { reasoningContent?: unknown }).reasoningContent ===
          "string")
    );
  }

  static _isValidStreamToolCalls(toolCalls: unknown): boolean {
    return (
      Array.isArray(toolCalls) &&
      toolCalls.length > 0 &&
      toolCalls.every(
        (toolCall) =>
          toolCall !== null &&
          typeof toolCall === "object" &&
          ((toolCall as { id?: unknown }).id === undefined ||
            typeof (toolCall as { id?: unknown }).id === "string") &&
          ((toolCall as { name?: unknown }).name === undefined ||
            typeof (toolCall as { name?: unknown }).name === "string") &&
          ((toolCall as { arguments?: unknown }).arguments === undefined ||
            typeof (toolCall as { arguments?: unknown }).arguments === "string")
      )
    );
  }

  override getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    return {
      ls_provider: "oci_genai_generic",
      ls_model_name:
        this._params.onDemandModelId || this._params.dedicatedEndpointId || "",
      ls_model_type: "chat",
      ls_temperature: options.requestParams?.temperature ?? 0,
      ls_max_tokens: options.requestParams?.maxTokens ?? 0,
      ls_stop: options.stop ?? [],
    };
  }
}
