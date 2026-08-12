import {
  BaseChatModelCallOptions,
  BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import { AuthParams, ClientConfiguration } from "oci-common";
import type {
  GenerativeAiInferenceClient,
  models,
  responses,
} from "oci-generativeaiinference";

type ChatDetails = models.ChatDetails;
type CohereChatRequest = models.CohereChatRequest;
type CohereChatResponse = models.CohereChatResponse;
type GenericChatRequest = models.GenericChatRequest;
type GenericChatResponse = models.GenericChatResponse;

export enum OciGenAiNewClientAuthType {
  ConfigFile,
  InstancePrincipal,
  ResourcePrincipal,
  Session,
  Other,
}

export interface ConfigFileAuthParams {
  clientConfigFilePath: string;
  clientProfile: string;
}

export interface OciGenAiNewClientParams {
  authType: OciGenAiNewClientAuthType;
  regionId?: string;
  authParams?: ConfigFileAuthParams | AuthParams;
  clientConfiguration?: ClientConfiguration;
}

export interface OciGenAiClientParams {
  client?: GenerativeAiInferenceClient;
  newClientParams?: OciGenAiNewClientParams;
}

export interface OciGenAiServingParams {
  onDemandModelId?: string;
  dedicatedEndpointId?: string;
}

export type OciGenAiSupportedRequestType =
  | GenericChatRequest
  | CohereChatRequest;
export type OciGenAiModelBaseParams = BaseChatModelParams &
  OciGenAiClientParams &
  Omit<ChatDetails, "chatRequest" | "servingMode"> &
  OciGenAiServingParams;

export interface OciGenAiModelCallOptions<RequestType>
  extends BaseChatModelCallOptions {
  requestParams?: RequestType;
}

export type OciGenAiSupportedResponseType =
  | GenericChatResponse
  | CohereChatResponse;
export type OciGenAiChatCallResponseType =
  | responses.ChatResponse
  | ReadableStream<Uint8Array>
  | null;
