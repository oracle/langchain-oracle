import {
  AuthenticationDetailsProvider,
  AuthParams,
  ClientConfiguration,
  ConfigFileAuthenticationDetailsProvider,
  InstancePrincipalsAuthenticationDetailsProviderBuilder,
  SessionAuthDetailProvider,
  MaxAttemptsTerminationStrategy,
  ResourcePrincipalAuthenticationDetailsProvider,
} from "oci-common";

import { GenerativeAiInferenceClient } from "oci-generativeaiinference";

import {
  ConfigFileAuthParams,
  OciGenAiClientParams,
  OciGenAiNewClientAuthType,
} from "./types.js";

/** Owns the OCI SDK client created from the integration's auth configuration. */
export class OciGenAiSdkClient {
  private _client: GenerativeAiInferenceClient;

  private constructor(client: GenerativeAiInferenceClient) {
    this._client = client;
  }

  get client(): GenerativeAiInferenceClient {
    return this._client;
  }

  close(): void {
    this._client.close();
  }

  static async create(
    params: OciGenAiClientParams
  ): Promise<OciGenAiSdkClient> {
    const client: GenerativeAiInferenceClient = await this._getClient(params);
    return new OciGenAiSdkClient(client);
  }

  static async _getClient(
    params: OciGenAiClientParams
  ): Promise<GenerativeAiInferenceClient> {
    if (params.client) {
      return params.client;
    }

    return await this._createAndSetupNewClient(params);
  }

  static async _createAndSetupNewClient(
    params: OciGenAiClientParams
  ): Promise<GenerativeAiInferenceClient> {
    const client: GenerativeAiInferenceClient = await this._createNewClient(
      params
    );

    if (params.newClientParams?.regionId) {
      // Without an explicit override, the OCI SDK preserves the region exposed
      // by a region-aware auth provider, such as the default config profile.
      client.regionId = params.newClientParams.regionId;
    }

    if (params.newClientParams?.serviceEndpoint) {
      // Set the region first: the SDK region setter derives its default endpoint.
      // An explicit endpoint must be applied afterwards so it remains authoritative.
      client.endpoint = params.newClientParams.serviceEndpoint;
    }

    return client;
  }

  static async _createNewClient(
    params: OciGenAiClientParams
  ): Promise<GenerativeAiInferenceClient> {
    const authParams: AuthParams = await this._getClientAuthParams(params);
    const clientConfiguration: ClientConfiguration =
      this._getClientConfiguration(params.newClientParams?.clientConfiguration);
    return new GenerativeAiInferenceClient(authParams, clientConfiguration);
  }

  static async _getClientAuthParams(
    params: OciGenAiClientParams
  ): Promise<AuthParams> {
    if (params.newClientParams?.authType === OciGenAiNewClientAuthType.Other) {
      return <AuthParams>params.newClientParams.authParams;
    }

    const authenticationDetailsProvider: AuthenticationDetailsProvider =
      await this._getAuthProvider(params);
    return { authenticationDetailsProvider };
  }

  static async _getAuthProvider(
    params: OciGenAiClientParams
  ): Promise<AuthenticationDetailsProvider> {
    switch (params.newClientParams?.authType) {
      case undefined:
      case OciGenAiNewClientAuthType.ConfigFile:
        return this._getConfigFileAuthProvider(params);

      case OciGenAiNewClientAuthType.InstancePrincipal:
        return await this._getInstancePrincipalAuthProvider();

      case OciGenAiNewClientAuthType.ResourcePrincipal:
        return this._getResourcePrincipalAuthProvider();

      case OciGenAiNewClientAuthType.Session:
        return this._getSessionAuthProvider(params);

      default:
        throw new Error("Invalid authentication type");
    }
  }

  static _getConfigFileAuthProvider(
    params: OciGenAiClientParams
  ): AuthenticationDetailsProvider {
    const configFileAuthParams: ConfigFileAuthParams = <ConfigFileAuthParams>(
      params.newClientParams?.authParams
    );
    return new ConfigFileAuthenticationDetailsProvider(
      configFileAuthParams?.clientConfigFilePath,
      configFileAuthParams?.clientProfile
    );
  }

  static async _getInstancePrincipalAuthProvider(): Promise<AuthenticationDetailsProvider> {
    const instancePrincipalAuthenticationBuilder =
      new InstancePrincipalsAuthenticationDetailsProviderBuilder();
    return await instancePrincipalAuthenticationBuilder.build();
  }

  static _getResourcePrincipalAuthProvider(): AuthenticationDetailsProvider {
    // The OCI SDK obtains Resource Principal credentials from the runtime
    // environment used by OCI Functions and Data Science.
    return ResourcePrincipalAuthenticationDetailsProvider.builder();
  }

  static _getSessionAuthProvider(
    params: OciGenAiClientParams
  ): AuthenticationDetailsProvider {
    const configFileAuthParams: ConfigFileAuthParams = <ConfigFileAuthParams>(
      params.newClientParams?.authParams
    );
    return new SessionAuthDetailProvider(
      configFileAuthParams?.clientConfigFilePath,
      configFileAuthParams?.clientProfile
    );
  }

  static _getClientConfiguration(
    clientConfiguration: ClientConfiguration | undefined
  ): ClientConfiguration {
    if (clientConfiguration) {
      return clientConfiguration;
    }

    return {
      retryConfiguration: {
        terminationStrategy: new MaxAttemptsTerminationStrategy(1),
      },
    };
  }
}
