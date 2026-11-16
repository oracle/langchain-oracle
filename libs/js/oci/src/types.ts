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

/** OCI configuration-file location and profile used for API-key or session auth. */
export interface ConfigFileAuthParams {
  clientConfigFilePath: string;
  clientProfile: string;
}

/**
 * Controls construction of the OCI SDK client when a caller does not inject one.
 * `serviceEndpoint` is primarily useful for private or region-specific endpoints.
 */
export interface OciGenAiNewClientParams {
  authType: OciGenAiNewClientAuthType;
  regionId?: string;
  serviceEndpoint?: string;
  authParams?: ConfigFileAuthParams | AuthParams;
  clientConfiguration?: ClientConfiguration;
}

/** Provide either a caller-owned SDK client or the parameters used to create one. */
export interface OciGenAiClientParams {
  client?: GenerativeAiInferenceClient;
  newClientParams?: OciGenAiNewClientParams;
}

/** Exactly one OCI serving target must be provided for each chat request. */
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
