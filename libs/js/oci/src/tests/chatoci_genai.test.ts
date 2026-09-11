/* eslint-disable @typescript-eslint/no-explicit-any */
import { expect, test, vi } from "vitest";

import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  HumanMessage as LangChainHumanMessage,
  SystemMessage as LangChainSystemMessage,
  ToolMessage as LangChainToolMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

import { GenerativeAiInferenceClient, models } from "oci-generativeaiinference";
import {
  InstancePrincipalsAuthenticationDetailsProviderBuilder,
  MaxAttemptsTerminationStrategy,
  Region,
  ResourcePrincipalAuthenticationDetailsProvider,
} from "oci-common";
import { z } from "zod";

import { OciGenAiBaseChat } from "../index.js";
import { OciGenAiCohereChat } from "../cohere_chat.js";
import { OciGenAiGenericChat } from "../generic_chat.js";
import { JsonServerEventsIterator } from "../server_events_iterator.js";
import { OciGenAiSdkClient } from "../oci_genai_sdk_client.js";
import { OciGenAiClientParams, OciGenAiNewClientAuthType } from "../types.js";

const {
  CohereChatRequest,
  CohereSystemMessage: OciGenAiCohereSystemMessage,
  CohereUserMessage: OciGenAiCohereUserMessage,
  GenericChatRequest,
  TextContent,
  CohereChatBotMessage,
  CohereSystemMessage,
  CohereUserMessage,
  AssistantMessage: GenericAssistantMessage,
  UserMessage: GenericUserMessage,
  SystemMessage: GenericSystemMessage,
} = models;
type Message = models.Message;
type CohereMessage = models.CohereMessage;
type CohereChatRequest = models.CohereChatRequest;
type GenericChatRequest = models.GenericChatRequest;
type GenericChatResponse = models.GenericChatResponse;
type TextContent = models.TextContent;
type CohereChatBotMessage = models.CohereChatBotMessage;
type CohereSystemMessage = models.CohereSystemMessage;
type CohereUserMessage = models.CohereUserMessage;

type OciGenAiChatConstructor = new (args: any) =>
  | OciGenAiCohereChat
  | OciGenAiGenericChat;

/*
 *  JsonServerEventsIterator tests
 */

const invalidServerEvents: string[][] = [
  [{} as string],
  ['{"prop":"val"}\n\n'],
  [""],
  [" "],
  [' ata: {"final": true}\n\n'],
  ['data  {"prop":"val"}\n\n'],
  ['data: {"prop":"val"\n\n'],
  ["data:\n\n"],
  ["data: \n\n"],
  ["data: 5\n\n"],
  ["data: fail\n\n"],
  ['data: "testing 1, 2, 3"\n'],
  ["data: null\n\n"],
  ["data: -345.345345\n\n"],
  ["\u{1F600}e\u0301\n\n"],
];

const invalidEventDataErrors = new RegExp(
  "Event text is empty, too short or malformed|" +
    "Incomplete server-sent event at end of stream|" +
    "Stream ended with an incomplete server-sent event|" +
    "Event data is empty or too short to be valid|" +
    "Could not parse event data as JSON|" +
    "Event data could not be parsed into an object"
);

const validServerEvents: string[] = [
  'data: {"test":5}\n\n',
  'data: {"message":"this is a message"}\n\n',
  'data: {"finalReason":"i j`us`t felt like stopping", "terminate": true}\n\n',
  "data: {}\n\n",
  'data: {"message":"this is a message","ignore":{"yes":"no"}}\n\n',
  'data: {"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":" I"}]},"pad":"aaaaa"}\n\n',
  'data: {"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":" discover"}]},"pad":"aaaaaaaaaaa"}\n\n',
];

interface ValidServerEventProps {
  finalReason: string;
  terminate: boolean;
}

const validServerEventsProps: string[] = [
  `data: ${JSON.stringify(<ValidServerEventProps>{
    finalReason: "reason 1",
    terminate: true,
  })}\n\n`,
  `data: ${JSON.stringify(<ValidServerEventProps>{
    finalReason: "this is a message",
    terminate: true,
  })}\n\n`,
  `data: ${JSON.stringify(<ValidServerEventProps>{
    finalReason: "i just felt like stopping",
    terminate: true,
  })}\n\n`,
];

test("JsonServerEventsIterator invalid events", async () => {
  for (const values of invalidServerEvents) {
    const stream: ReadableStream<Uint8Array> =
      createStreamFromStringArray(values);
    const streamIterator = new JsonServerEventsIterator(stream);
    await testInvalidValues(streamIterator);
  }
});

test("JsonServerEventsIterator empty events", async () => {
  await testNumExpectedServerEvents([], 0);
});

test("JsonServerEventsIterator valid events", async () => {
  let numExpectedEvents: number = 0;

  for (const event of validServerEvents) {
    if (event.startsWith("data:")) {
      numExpectedEvents += 1;
    }
  }

  await testNumExpectedServerEvents(validServerEvents, numExpectedEvents);
});

test("JsonServerEventsIterator valid events check properties", async () => {
  const stream: ReadableStream<Uint8Array> = createStreamFromStringArray(
    validServerEventsProps
  );
  const streamIterator = new JsonServerEventsIterator(stream);

  for await (const event of streamIterator) {
    expect(typeof (<ValidServerEventProps>event).finalReason).toBe("string");
    expect((<ValidServerEventProps>event).terminate).toBe(true);
  }
});

test("JsonServerEventsIterator parses multiple events in one chunk", async () => {
  const events = await collectServerEvents([
    'data: {"text":"one"}\n\ndata: {"text":"two"}\n\n',
  ]);

  expect(events).toEqual([{ text: "one" }, { text: "two" }]);
});

test("JsonServerEventsIterator parses an event split across chunks", async () => {
  const events = await collectServerEvents([
    'data: {"text":"he',
    'llo"}\n',
    "\n",
  ]);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator preserves UTF-8 characters split across chunks", async () => {
  const text = 'data: {"text":"😊"}\n\n';
  const encoded = new TextEncoder().encode(text);
  const emojiStart = new TextEncoder().encode('data: {"text":"').length;
  const events = await collectServerEventBytes([
    encoded.slice(0, emojiStart + 2),
    encoded.slice(emojiStart + 2),
  ]);

  expect(events).toEqual([{ text: "😊" }]);
});

test("JsonServerEventsIterator parses CRLF framing", async () => {
  const events = await collectServerEvents(['data: {"text":"hello"}\r\n\r\n']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator parses a CRLF delimiter split across chunks", async () => {
  const events = await collectServerEvents([
    'data: {"text":"hello"}\r\n\r',
    "\n",
  ]);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator handles JSON and delimiter fragmentation together", async () => {
  const events = await collectServerEvents([
    'data: {"text":"hello"}\n\ndata: {"text":"wor',
    'ld"}\n',
    "\n",
  ]);

  expect(events).toEqual([{ text: "hello" }, { text: "world" }]);
});

test("JsonServerEventsIterator parses mixed newline framing", async () => {
  const events = await collectServerEvents([
    'data: {"text":"one"}\r\n\ndata: {"text":"two"}\n\r\n',
  ]);

  expect(events).toEqual([{ text: "one" }, { text: "two" }]);
});

test("JsonServerEventsIterator accepts data fields without a space", async () => {
  const events = await collectServerEvents(['data:{"text":"hello"}\n\n']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator joins multiple data fields in one event", async () => {
  const events = await collectServerEvents([
    'event: message\nid: 123\ndata: {"text":"hello",\ndata: "done":true}\n\n',
  ]);

  expect(events).toEqual([{ text: "hello", done: true }]);
});

test("JsonServerEventsIterator ignores comments and control-only events", async () => {
  const events = await collectServerEvents([
    ": keepalive\n\n",
    "event: message\nid: 123\nretry: 1000\n\n",
    'data: {"text":"hello"}\n\n',
  ]);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator ignores the standard done sentinel", async () => {
  const events = await collectServerEvents([
    'data: {"text":"hello"}\n\ndata: [DONE]\n\n',
  ]);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator dispatches a final event at end of stream", async () => {
  const events = await collectServerEvents(['data: {"text":"hello"}']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator identifies incomplete data at end of stream", async () => {
  await expect(collectServerEvents(['data: {"text":"partial'])).rejects.toThrow(
    "Stream ended with an incomplete server-sent event"
  );
});

test("JsonServerEventsIterator parses CR-only framing", async () => {
  const events = await collectServerEvents(['data: {"text":"hello"}\r\r']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator bounds incomplete events", async () => {
  await expect(
    collectServerEvents([
      `data: {"text":"${"x".repeat(
        JsonServerEventsIterator._MAX_BUFFERED_TEXT_LENGTH
      )}`,
    ])
  ).rejects.toThrow("Server-sent event exceeds maximum buffered text length");
});

/*
 *  OciGenAiSdkClient tests
 */

const authenticationDetailsProvider = {
  getRegion() {
    return Region.US_PHOENIX_1;
  },
  getPassphrase() {
    return "";
  },
  async getKeyId(): Promise<string> {
    return "";
  },
  getPrivateKey() {
    return `-----BEGIN RSA PRIVATE KEY-----
MIICXQIBAAKBgQDTkUM7vYZSUYtm2bY/OmcvF9dQ37I3HMyKIKmFPck7Q4u5LqPB
qTuDNnd0tHBFfRaGpVsgcT46g1sIJwvfCnB5VFkAsheMHc8uUOBUD0DqBbkOLFGU
KI45rD0BUzOzjRW/NI5YFWUJJZGuD7tUP1gEwmr0wIvqTdpPI/CyN0pUTQIDAQAB
AoGAJzg1g3yVyurM8csIKt5zxFoiEx701ZykGjMF2epjRHY4D6MivkLWAnP1XxAY
A/m1VE6Q/wmfJI+3L2K1o6o2wSDUqbU+qW3xHVxc3U63JpUBa2MFQaupriEaA8ky
4iq5Zhs2OlRL02+A9KHvfus6MFhWWPLnkNrSx8cIaJycGgECQQDyFIuB9z76OUCU
B63TbqeRhzbBsVUc/6hErWacb4JCUtGk6s141l5V5pDNO2+w3mQ6HxqWLSct+19t
5BormrDNAkEA37uQj+OkjYBoeGEuB00PJBnlUIaQ/qHv7863aLlKcFdnFvmrzztA
A06QhjNCFBwJHwdSLz95ztDTpccmLIAxgQJBAO/Q4pOR+FWyugLryIwYpvBIXzpr
DsJ3kp7WmTyISyahHQafhYYb98BpdTGbm/4/klLx1UjI2nN2/wbCXhqsWFECQAu/
PGLhr/UiBdo0OAd4G1Bo76pftmM4O3Ha57Re7jKh1C7Xoxa5ZK4HxPzW2iRWKIBx
kPYcHhgmzMYKg82YWYECQQCejFaH73vZO3qUn+2pdHg3mUYYYQA7r/ms7MQ7mckg
1wPuzmfsEfsAzOaMvs8SsyG5sOdBLWfsGRabFaleBntX
-----END RSA PRIVATE KEY-----`;
  },
};

const defaultClient = {
  newClientParams: {
    authType: OciGenAiNewClientAuthType.Other,
    authParams: { authenticationDetailsProvider },
  },
};

test("OciGenAiSdkClient create default client", async () => {
  const sdkClient = await OciGenAiSdkClient.create(defaultClient);
  testSdkClient(sdkClient, Region.US_PHOENIX_1.regionId, 0);
});

test("OciGenAiSdkClient create client based on parameters", async () => {
  const newClientParams: OciGenAiClientParams = {
    newClientParams: {
      authType: OciGenAiNewClientAuthType.Other,
      regionId: "mars",
      authParams: { authenticationDetailsProvider },
      clientConfiguration: {
        retryConfiguration: {
          terminationStrategy: new MaxAttemptsTerminationStrategy(5),
        },
      },
    },
  };

  const sdkClient = await OciGenAiSdkClient.create(newClientParams);
  testSdkClient(sdkClient, "mars", 4);
});

test("OciGenAiSdkClient create client based on some parameters #2", async () => {
  const sdkClient = await OciGenAiSdkClient.create(defaultClient);
  testSdkClient(sdkClient, Region.US_PHOENIX_1.regionId, 0);
});

test("OciGenAiSdkClient creates an Instance Principal client", async () => {
  const build = vi
    .spyOn(
      InstancePrincipalsAuthenticationDetailsProviderBuilder.prototype,
      "build"
    )
    .mockResolvedValue(authenticationDetailsProvider as any);

  try {
    const sdkClient = await OciGenAiSdkClient.create({
      newClientParams: {
        authType: OciGenAiNewClientAuthType.InstancePrincipal,
        regionId: Region.SA_SAOPAULO_1.regionId,
        clientConfiguration: {
          retryConfiguration: {
            terminationStrategy: new MaxAttemptsTerminationStrategy(3),
          },
        },
      },
    });

    expect(build).toHaveBeenCalledOnce();
    expect((<any>sdkClient.client)._authProvider).toBe(
      authenticationDetailsProvider
    );
    testSdkClient(sdkClient, Region.SA_SAOPAULO_1.regionId, 2);
  } finally {
    build.mockRestore();
  }
});

test("OciGenAiSdkClient creates a Resource Principal client", async () => {
  const builder = vi
    .spyOn(ResourcePrincipalAuthenticationDetailsProvider, "builder")
    .mockReturnValue(authenticationDetailsProvider as any);

  try {
    const sdkClient = await OciGenAiSdkClient.create({
      newClientParams: {
        authType: OciGenAiNewClientAuthType.ResourcePrincipal,
      },
    });

    expect(builder).toHaveBeenCalledOnce();
    expect((<any>sdkClient.client)._authProvider).toBe(
      authenticationDetailsProvider
    );
    testSdkClient(sdkClient, Region.US_PHOENIX_1.regionId, 0);
  } finally {
    builder.mockRestore();
  }
});

test("OciGenAiSdkClient pre-configured client", async () => {
  const client = new GenerativeAiInferenceClient(
    { authenticationDetailsProvider },
    {
      retryConfiguration: {
        terminationStrategy: new MaxAttemptsTerminationStrategy(10),
      },
    }
  );

  client.regionId = "venus";
  const sdkClient = await OciGenAiSdkClient.create({ client });
  testSdkClient(sdkClient, "venus", 9);
});

test("OCI GenAI Generic chat uses a caller-owned SDK client", async () => {
  const client = new GenerativeAiInferenceClient({
    authenticationDetailsProvider,
  });
  const callChat = vi.spyOn(client, "chat").mockResolvedValue({
    chatResult: {
      chatResponse: {
        choices: [
          {
            message: {
              content: [{ type: TextContent.type, text: "hello" }],
              role: "ASSISTANT",
            },
          },
        ],
      },
    },
  } as any);
  const closeClient = vi.spyOn(client, "close");
  const chat = new OciGenAiGenericChat({
    compartmentId: "oci.compartment.ocid",
    onDemandModelId: "oci.model.ocid",
    maxRetries: 0,
    client,
  });

  try {
    const response = await chat.invoke("Say hello");
    expect(response.content).toBe("hello");
    expect(callChat).toHaveBeenCalledOnce();

    await chat.close();
    expect(closeClient).not.toHaveBeenCalled();
  } finally {
    client.close();
    callChat.mockRestore();
    closeClient.mockRestore();
  }
});

/*
 *  Chat models tests
 */

const compartmentId = "oci.compartment.ocid";
const onDemandModelId = "oci.model.ocid";
const dedicatedEndpointId = "oci.dedicated.oci";
const createParams = {
  compartmentId,
  onDemandModelId,
  // Unit-test service failures must not spend time in production retry backoff.
  maxRetries: 0,
};

const DummyClient = {
  async chat() {
    return undefined;
  },
};

test("OCI GenAI chat models creation", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor) => {
      let instance = new ChatClassType({ client: DummyClient });
      await expect(instance.invoke("prompt")).rejects.toThrow(
        "Invalid compartmentId"
      );

      instance = new ChatClassType({
        compartmentId,
        client: DummyClient,
      });

      await expect(instance.invoke("prompt")).rejects.toThrow(
        "Exactly one of onDemandModelId or dedicatedEndpointId must be supplied"
      );

      instance = new ChatClassType({
        compartmentId,
        onDemandModelId: "",
        client: DummyClient,
      });

      await expect(instance.invoke("prompt")).rejects.toThrow(
        "Exactly one of onDemandModelId or dedicatedEndpointId must be supplied"
      );

      instance = new ChatClassType({
        compartmentId,
        onDemandModelId,
        client: DummyClient,
      });

      await expect(instance.invoke("prompt")).rejects.toThrow(
        /Invalid CohereResponse object|Invalid GenericChatResponse object/
      );

      expect(instance._params.compartmentId).toBe(compartmentId);
      expect(instance._params.onDemandModelId).toBe(onDemandModelId);
    }
  );
});

test("OCI GenAI chat rejects both serving modes", async () => {
  const chat = new OciGenAiGenericChat({
    compartmentId,
    onDemandModelId,
    dedicatedEndpointId,
    client: DummyClient as unknown as GenerativeAiInferenceClient,
  });

  await expect(chat.invoke("prompt")).rejects.toThrow(
    "Exactly one of onDemandModelId or dedicatedEndpointId must be supplied"
  );
});

test("OCI GenAI chat accepts text-only content blocks", () => {
  const message = new LangChainHumanMessage({
    content: [{ type: "text", text: "hello" }],
  } as any);

  expect(OciGenAiBaseChat._contentToText(message.content)).toBe("hello");
  expect(
    new OciGenAiGenericChat(createParams)._prepareRequest([message], {}, false)
  ).toMatchObject({
    messages: [
      {
        content: [{ type: TextContent.type, text: "hello" }],
      },
    ],
  });
});

test("OCI GenAI chat rejects mixed text and multimodal content blocks", () => {
  const message = new LangChainHumanMessage({
    content: [
      { type: "text", text: "describe this image" },
      { type: "image_url", image_url: "data:image/png;base64,blah" },
    ],
  } as any);

  expect(() => OciGenAiBaseChat._contentToText(message.content)).toThrow(
    "Unsupported message content"
  );
  expect(() =>
    new OciGenAiGenericChat(createParams)._prepareRequest([message], {}, false)
  ).toThrow("Unsupported message content");
});

test("OCI GenAI chat identifies itself for tracing", () => {
  expect(new OciGenAiGenericChat(createParams)._llmType()).toBe("oci_genai");
});

test("OCI GenAI chat closes only clients it owns", async () => {
  const closeOwned = vi.fn();
  const closeExternal = vi.fn();
  const ownedChat = new OciGenAiGenericChat(createParams);
  const externalChat = new OciGenAiGenericChat(createParams);

  ownedChat._sdkClient = { close: closeOwned } as unknown as OciGenAiSdkClient;
  ownedChat._ownsSdkClient = true;
  externalChat._sdkClient = {
    close: closeExternal,
  } as unknown as OciGenAiSdkClient;

  await ownedChat.close();
  await externalChat.close();

  expect(closeOwned).toHaveBeenCalledOnce();
  expect(closeExternal).not.toHaveBeenCalled();
  expect(ownedChat._sdkClient).toBeUndefined();
});

test("OCI GenAI chat initializes one owned SDK client for concurrent calls", async () => {
  const sdkClient = {
    client: { chat: vi.fn() },
    close: vi.fn(),
  } as unknown as OciGenAiSdkClient;
  const createClientSpy = vi
    .spyOn(OciGenAiSdkClient, "create")
    .mockResolvedValue(sdkClient);
  const chat = new OciGenAiGenericChat(createParams);

  try {
    await Promise.all([chat._setupClient(), chat._setupClient()]);

    expect(createClientSpy).toHaveBeenCalledOnce();
    expect(chat._sdkClient).toBe(sdkClient);
    await chat.close();
    expect(sdkClient.close).toHaveBeenCalledOnce();
  } finally {
    createClientSpy.mockRestore();
  }
});

test("OCI GenAI chat closes an owned client created after close", async () => {
  const sdkClient = {
    client: { chat: vi.fn() },
    close: vi.fn(),
  } as unknown as OciGenAiSdkClient;
  let resolveClient: ((client: OciGenAiSdkClient) => void) | undefined;
  const createClientSpy = vi.spyOn(OciGenAiSdkClient, "create").mockReturnValue(
    new Promise((resolve) => {
      resolveClient = resolve;
    })
  );
  const chat = new OciGenAiGenericChat(createParams);

  try {
    const setup = chat._setupClient();
    const rejectedSetup = expect(setup).rejects.toThrow(
      "OciGenAiBaseChat is closed"
    );
    await vi.waitFor(() => expect(resolveClient).toBeDefined());
    const close = chat.close();
    resolveClient!(sdkClient);

    await Promise.all([close, rejectedSetup]);
    expect(sdkClient.close).toHaveBeenCalledOnce();
  } finally {
    createClientSpy.mockRestore();
  }
});

test("OCI GenAI chat delegates OCI invocations through AsyncCaller", async () => {
  const response = { chatResult: { chatResponse: { choices: [] } } };
  const client = {
    chat: vi.fn().mockResolvedValue(response),
  };
  const chat = new OciGenAiGenericChat(createParams);
  chat._sdkClient = { client } as unknown as OciGenAiSdkClient;
  const callerCallSpy = vi.spyOn(chat.caller, "call");

  await expect(
    chat._callChat(chat._createRequest([new HumanMessage("hello")], {}, false))
  ).resolves.toBe(response);
  expect(callerCallSpy).toHaveBeenCalledOnce();
  expect(client.chat).toHaveBeenCalledOnce();
});

const chatClassReturnValues = [
  {
    chatResult: {
      chatResponse: {
        text: "response text",
      },
    },
  },
  {
    chatResult: {
      chatResponse: {
        choices: [
          {
            message: {
              content: [
                {
                  type: TextContent.type,
                  text: "response text",
                },
              ],
            },
          },
        ],
      },
    },
  },
];

test("OCI GenAI Cohere chat rejects tool messages", async () => {
  const chatClass = new OciGenAiCohereChat(createParams);

  await expect(
    chatClass.invoke([
      new LangChainToolMessage({
        content: "tools message",
        tool_call_id: "tool_id",
      }),
      new LangChainHumanMessage("Human message"),
    ])
  ).rejects.toThrow("Message type 'tool' is not supported");
});

const lastHumanMessage = "Last human message";
const messages = [
  new LangChainHumanMessage("Human message"),
  new LangChainSystemMessage("System message"),
  new LangChainSystemMessage("System message"),
  new LangChainHumanMessage(lastHumanMessage),
];

const callOptions = {
  stop: ["\n", "."],
  requestParams: {
    temperature: 0.32,
    maxTokens: 1,
  },
};

const createRequestParams = [
  {
    test: (cohereRequest: CohereChatRequest, params: any) => {
      expect(cohereRequest.apiFormat).toBe(CohereChatRequest.apiFormat);
      expect(cohereRequest.message).toBe(lastHumanMessage);
      expect(cohereRequest.chatHistory).toStrictEqual(
        removeElements(params.convertMessages(messages), [3])
      );
      expect(cohereRequest.isStream).toBe(true);
      expect(cohereRequest.stopSequences).toStrictEqual(callOptions.stop);
      expect(cohereRequest.temperature).toBe(
        callOptions.requestParams.temperature
      );
      expect(cohereRequest.maxTokens).toBe(callOptions.requestParams.maxTokens);
    },
    convertMessages: (messages: BaseMessage[]): Message[] =>
      messages.map(OciGenAiCohereChat._convertBaseMessageToCohereMessage),
  },
  {
    test: (genericRequest: GenericChatRequest, params: any) => {
      expect(genericRequest.apiFormat).toBe(GenericChatRequest.apiFormat);
      expect(genericRequest.messages).toStrictEqual(
        params.convertMessages(messages)
      );
      expect(genericRequest.isStream).toBe(true);
      expect(genericRequest.stop).toStrictEqual(callOptions.stop);
      expect(genericRequest.temperature).toBe(
        callOptions.requestParams.temperature
      );
      expect(genericRequest.maxTokens).toBe(
        callOptions.requestParams.maxTokens
      );
    },
    convertMessages: (messages: BaseMessage[]): Message[] =>
      messages.map(OciGenAiGenericChat._convertBaseMessageToGenericMessage),
  },
];

const invalidMessages = [
  [],
  [
    new LangChainToolMessage("Human message", "tool"),
    new LangChainSystemMessage("System message"),
    new LangChainHumanMessage(lastHumanMessage),
  ],
  [
    new LangChainSystemMessage({
      content: [
        {
          type: "image_url",
          image_url: "data:image/pgn;base64,blah",
        },
      ],
    }),
  ],
];

test("OCI GenAI chat create request", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor, params) => {
      const chatClass = new ChatClassType(createParams);
      const request = chatClass._prepareRequest(messages, callOptions, true);
      params.test(request, params);
    },
    createRequestParams
  );
});

test("OCI GenAI Generic streaming usage is opt-in", () => {
  const chat = new OciGenAiGenericChat(createParams);
  const withoutUsage = chat._createRequest(messages, {}, true);
  const withUsage = chat._createRequest(
    messages,
    { requestParams: { streamOptions: { isIncludeUsage: true } } },
    true
  );

  expect(withoutUsage.streamOptions).toBeUndefined();
  expect(withUsage.streamOptions).toEqual({ isIncludeUsage: true });
});

test("OCI GenAI Generic request params cannot override adapter invariants", () => {
  const chat = new OciGenAiGenericChat(createParams);
  const request = chat._createRequest(
    [new HumanMessage("hello")],
    {
      requestParams: {
        apiFormat: "COHERE",
        messages: [],
      } as unknown as GenericChatRequest,
    },
    false
  );

  expect(request.apiFormat).toBe(GenericChatRequest.apiFormat);
  expect(request.messages).toEqual([
    {
      role: GenericUserMessage.role,
      content: [{ type: TextContent.type, text: "hello" }],
    },
  ]);
});

test("only Generic chat exposes LangChain tool binding", () => {
  expect(typeof new OciGenAiGenericChat(createParams).bindTools).toBe(
    "function"
  );
  expect(typeof new OciGenAiCohereChat(createParams).bindTools).toBe(
    "undefined"
  );
});

test("OCI GenAI chat create invalid request messages", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor) => {
      const chatClass = new ChatClassType(createParams);
      expect(() =>
        chatClass._prepareRequest(invalidMessages[0], callOptions, true)
      ).toThrow("No messages provided");
      const prepareToolMessage = () =>
        chatClass._prepareRequest(invalidMessages[1], callOptions, true);
      if (ChatClassType === OciGenAiCohereChat) {
        expect(prepareToolMessage).toThrow(
          "Message type 'tool' is not supported"
        );
      } else {
        expect(prepareToolMessage).toThrow(
          "ToolMessage references unknown tool call 'tool'"
        );
      }
      expect(() =>
        chatClass._prepareRequest(invalidMessages[2], callOptions, true)
      ).toThrow("Unsupported message content");
    }
  );
});

const invalidCohereResponseValues = [
  undefined,
  null,
  {},
  { props: true },
  { text: 5505 },
  { text: ["hello "] },
  [],
];

test("OCI GenAI chat Cohere parse invalid response", async () => {
  const cohereChat = new OciGenAiCohereChat(createParams);

  for (const invalidValue of invalidCohereResponseValues) {
    expect(() => cohereChat._parseResponse(<any>invalidValue)).toThrow(
      "Invalid CohereResponse object"
    );
  }
});

const validCohereResponseValues = [
  {
    apiFormat: CohereChatRequest.apiFormat,
    value: undefined,
    text: "This is the response text",
  },
  {
    text: "This is the response text",
  },
];

test("OCI GenAI Cohere parse valid response", async () => {
  const cohereChat = new OciGenAiCohereChat(createParams);

  for (const validValue of validCohereResponseValues) {
    expect(cohereChat._parseResponse(<any>validValue).content).toBe(
      "This is the response text"
    );
  }
});

const invalidCGenericResponseValues = [
  { choices: [] },
  undefined,
  null,
  {},
  [],
  { props: true },
  { choices: 5505 },
  { choices: ["hello "] },
  { choices: null },
  { choices: {} },
  {
    choices: [
      {
        content: undefined,
      },
    ],
  },
  {
    message: {
      content: {},
    },
  },
  {
    message: {
      content: [],
    },
  },
  { finishReason: {} },
  { finishReason: false },
  {
    choices: [5],
  },
  {
    choices: [
      {
        message: "bad value",
      },
    ],
  },
  {
    choices: [
      {
        message: {},
      },
    ],
  },
  {
    choices: [
      {
        message: null,
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: null,
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          toolCalls: [],
        },
      },
    ],
  },
  {
    choices: [
      {
        finishReason: 123,
        message: {
          content: [{ type: TextContent.type, text: "some text" }],
        },
      },
    ],
  },
  {
    choices: [
      {
        usage: "garbage",
        message: {
          content: [{ type: TextContent.type, text: "some text" }],
        },
      },
    ],
  },
  {
    usage: "garbage",
    choices: [
      {
        message: {
          content: [{ type: TextContent.type, text: "some text" }],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          toolCalls: [42],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [{}],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [null],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              text: "some text",
            },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              type: "IMAGE",
              text: "some text",
            },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: [1, 2, 3, 4],
            },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: null,
            },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: "This is ",
            },
          ],
        },
      },
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: false,
            },
          ],
        },
      },
    ],
  },
];

test("OCI GenAI Generic parse invalid response", async () => {
  const genericChat = new OciGenAiGenericChat(createParams);

  for (const invalidValue of invalidCGenericResponseValues) {
    expect(() => genericChat._parseResponse(<any>invalidValue)).toThrow(
      "Invalid GenericChatResponse object"
    );
  }
});

const validGenericResponseValues = [
  {
    choices: [
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: "This is the response text",
            },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: "This is ",
            },
            {
              type: TextContent.type,
              text: "the ",
            },
            {
              type: TextContent.type,
              text: "response text",
            },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: "This is ",
            },
          ],
        },
      },
      {
        message: {
          content: [
            {
              type: TextContent.type,
              text: "the response text",
            },
          ],
        },
      },
    ],
  },
];

test("OCI GenAI Generic parse valid response", async () => {
  const genericChat = new OciGenAiGenericChat(createParams);

  for (const validValue of validGenericResponseValues) {
    expect(["This is the response text", "This is ", ""]).toContain(
      genericChat._parseResponse(<any>validValue).content
    );
  }
});

test("OCI GenAI Generic invoke preserves tool calls and usage metadata", async () => {
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: {
      chat: async () => ({
        chatResult: {
          chatResponse: {
            choices: [
              {
                finishReason: "TOOL_CALLS",
                message: {
                  toolCalls: [
                    {
                      id: "call-1",
                      type: "FUNCTION",
                      name: "get_weather",
                      arguments: '{"city":"London"}',
                    },
                  ],
                },
              },
            ],
            usage: {
              promptTokens: 10,
              completionTokens: 5,
              totalTokens: 15,
            },
          },
        },
      }),
    } as any,
  });

  const message = await chat.invoke("weather?");

  expect(message.content).toBe("");
  expect(message.tool_calls).toEqual([
    {
      type: "tool_call",
      id: "call-1",
      name: "get_weather",
      args: { city: "London" },
    },
  ]);
  expect(message.usage_metadata).toEqual({
    input_tokens: 10,
    output_tokens: 5,
    total_tokens: 15,
  });
  expect(message.response_metadata).toMatchObject({
    finish_reason: "TOOL_CALLS",
  });
});

test("OCI GenAI Generic bindTools sends OCI function definitions", async () => {
  const requests: any[] = [];
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: {
      chat: async (request: unknown) => {
        requests.push(request);
        return {
          chatResult: {
            chatResponse: {
              choices: [
                {
                  finishReason: "STOP",
                  message: { content: [] },
                },
              ],
            },
          },
        };
      },
    } as any,
  });

  const bound = chat.bindTools([
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather for a city",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    },
  ]);

  await bound.invoke("weather?");

  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    chatDetails: {
      chatRequest: {
        tools: [
          {
            type: "FUNCTION",
            name: "get_weather",
            parameters: {
              required: ["city"],
            },
          },
        ],
      },
    },
  });
});

test("OCI GenAI Generic supports LangChain structured output", async () => {
  const requests: any[] = [];
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: {
      chat: async (request: unknown) => {
        requests.push(request);
        return {
          chatResult: {
            chatResponse: {
              choices: [
                {
                  finishReason: "TOOL_CALLS",
                  message: {
                    toolCalls: [
                      {
                        id: "call-extract",
                        type: "FUNCTION",
                        name: "describe_oci",
                        arguments:
                          '{"name":"OCI Generative AI","description":"A managed AI service."}',
                      },
                    ],
                  },
                },
              ],
            },
          },
        };
      },
    } as any,
  });

  const structuredModel = chat.withStructuredOutput(
    z.object({
      name: z.string(),
      description: z.string(),
    }),
    { name: "describe_oci" }
  );

  await expect(
    structuredModel.invoke("Describe OCI Generative AI.")
  ).resolves.toEqual({
    name: "OCI Generative AI",
    description: "A managed AI service.",
  });
  expect(requests[0]).toMatchObject({
    chatDetails: {
      chatRequest: {
        tools: [{ type: "FUNCTION", name: "describe_oci" }],
        toolChoice: { type: "FUNCTION", name: "describe_oci" },
      },
    },
  });

  const rawStructuredModel = chat.withStructuredOutput(
    z.object({
      name: z.string(),
      description: z.string(),
    }),
    { name: "describe_oci", includeRaw: true }
  );
  await expect(
    rawStructuredModel.invoke("Describe OCI Generative AI.")
  ).resolves.toMatchObject({
    parsed: {
      name: "OCI Generative AI",
      description: "A managed AI service.",
    },
    raw: { tool_calls: [{ id: "call-extract", name: "describe_oci" }] },
  });
});

test("OCI GenAI Generic validates Zod structured output", async () => {
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: {
      chat: async () => ({
        chatResult: {
          chatResponse: {
            choices: [
              {
                finishReason: "TOOL_CALLS",
                message: {
                  toolCalls: [
                    {
                      id: "call-extract",
                      type: "FUNCTION",
                      name: "extract",
                      arguments: '{"name":123,"description":false}',
                    },
                  ],
                },
              },
            ],
          },
        },
      }),
    } as any,
  });

  const structuredModel = chat.withStructuredOutput(
    z.object({
      name: z.string(),
      description: z.string(),
    })
  );

  await expect(
    structuredModel.invoke("Describe OCI Generative AI.")
  ).rejects.toThrow("Failed to parse");

  const rawStructuredModel = chat.withStructuredOutput(
    z.object({
      name: z.string(),
      description: z.string(),
    }),
    { includeRaw: true }
  );
  await expect(
    rawStructuredModel.invoke("Describe OCI Generative AI.")
  ).resolves.toMatchObject({
    parsed: null,
    raw: { tool_calls: [{ id: "call-extract", name: "extract" }] },
  });
});

test("OCI GenAI Generic rejects unsupported structured-output methods", () => {
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: { chat: async () => ({}) } as any,
  });
  const schema = z.object({ name: z.string() });

  expect(() =>
    chat.withStructuredOutput(schema, { method: "jsonMode" })
  ).toThrow('"jsonMode" is not implemented by the OCI Generic chat adapter');
  expect(() =>
    chat.withStructuredOutput(schema, { method: "jsonSchema" })
  ).toThrow('"jsonSchema" is not implemented by the OCI Generic chat adapter');
});

test("OCI GenAI Generic reconstructs streamed structured-output arguments", async () => {
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: {
      chat: async () =>
        createStreamFromStringArray([
          'data: {"message":{"toolCalls":[{"id":"call-extract","type":"FUNCTION","name":"extract","arguments":"{\\"name\\":\\"OCI"}]}}\n\n',
          'data: {"message":{"toolCalls":[{"type":"FUNCTION","arguments":" Generative AI\\",\\"description\\":\\"A managed AI service.\\"}"}]}}\n\n',
        ]),
    } as any,
  });
  const structuredModel = chat.withStructuredOutput(
    z.object({
      name: z.string(),
      description: z.string(),
    })
  );
  const results = [];
  const stream = await structuredModel.stream("Describe OCI Generative AI.");

  for await (const result of stream) {
    results.push(result);
  }

  expect(results.at(-1)).toEqual({
    name: "OCI Generative AI",
    description: "A managed AI service.",
  });
});

test("OCI GenAI Generic bindTools normalizes LangChain tool_choice", async () => {
  const requests: any[] = [];
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: {
      chat: async (request: unknown) => {
        requests.push(request);
        return {
          chatResult: {
            chatResponse: {
              choices: [{ finishReason: "STOP", message: { content: [] } }],
            },
          },
        };
      },
    } as any,
  });
  const tool = {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get weather for a city",
      parameters: { type: "object" },
    },
  };

  await chat.bindTools([tool], { tool_choice: "required" }).invoke("weather?");
  await chat
    .bindTools([tool], {
      tool_choice: { type: "function", function: { name: "get_weather" } },
    })
    .invoke("weather?");
  await chat
    .bindTools([tool], { tool_choice: "get_weather" })
    .invoke("weather?");

  expect(requests[0]).toMatchObject({
    chatDetails: { chatRequest: { toolChoice: { type: "REQUIRED" } } },
  });
  expect(requests[1]).toMatchObject({
    chatDetails: {
      chatRequest: {
        toolChoice: { type: "FUNCTION", name: "get_weather" },
      },
    },
  });
  expect(requests[2]).toMatchObject({
    chatDetails: {
      chatRequest: {
        toolChoice: { type: "FUNCTION", name: "get_weather" },
      },
    },
  });
});

test("OCI GenAI Generic bindTools rejects a named choice for an unbound tool", () => {
  const chat = new OciGenAiGenericChat({
    ...createParams,
    client: { chat: async () => ({}) } as any,
  });
  const tool = {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get weather for a city",
      parameters: { type: "object" },
    },
  };

  expect(() =>
    chat.bindTools([tool], { tool_choice: "does_not_exist" })
  ).toThrow("tool_choice references unbound function 'does_not_exist'");
  expect(() =>
    chat.bindTools([tool], {
      tool_choice: {
        type: "function",
        function: { name: "does_not_exist" },
      },
    })
  ).toThrow("tool_choice references unbound function 'does_not_exist'");
});

test("OCI GenAI Generic parses usage-only stream events", () => {
  const chat = new OciGenAiGenericChat(createParams);

  expect(
    chat._parseStreamedResponseChunk({
      usage: {
        promptTokens: 10,
        completionTokens: 5,
        totalTokens: 15,
      },
    })
  ).toEqual({
    usageMetadata: {
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
    },
  });
});

test("OCI GenAI Generic ignores reasoning-only stream events", () => {
  const chat = new OciGenAiGenericChat(createParams);

  // Reasoning-capable OCI models can emit this before their visible content.
  // The text-first adapter intentionally does not expose chain-of-thought.
  expect(
    chat._parseStreamedResponseChunk({
      index: 0,
      message: {
        role: "ASSISTANT",
        reasoningContent: "Let me think through this.",
      },
      serviceTier: "DEFAULT",
    })
  ).toBeUndefined();
});

test("OCI GenAI Generic reconstructs streamed tool-call arguments", () => {
  const chat = new OciGenAiGenericChat(createParams);
  const firstDelta = chat._parseStreamedResponseChunk({
    message: {
      toolCalls: [
        {
          id: "call-weather",
          type: "FUNCTION",
          name: "get_weather",
          arguments: '{"ci',
        },
      ],
    },
  });
  const secondDelta = chat._parseStreamedResponseChunk({
    message: {
      toolCalls: [
        {
          type: "FUNCTION",
          arguments: 'ty":"London"}',
        },
      ],
    },
  });

  if (!firstDelta?.toolCallChunks || !secondDelta?.toolCallChunks) {
    throw new Error("Expected tool call chunks from Generic stream deltas");
  }

  const merged = new AIMessageChunk({
    content: "",
    tool_call_chunks: firstDelta.toolCallChunks,
  }).concat(
    new AIMessageChunk({
      content: "",
      tool_call_chunks: secondDelta.toolCallChunks,
    })
  );

  expect(merged.tool_call_chunks).toEqual([
    {
      type: "tool_call_chunk",
      id: "call-weather",
      name: "get_weather",
      args: '{"city":"London"}',
      index: 0,
    },
  ]);
  expect(merged.tool_calls).toEqual([
    {
      type: "tool_call",
      id: "call-weather",
      name: "get_weather",
      args: { city: "London" },
    },
  ]);
});

test("OCI GenAI Generic rejects completed tool calls without IDs", () => {
  const chat = new OciGenAiGenericChat(createParams);

  expect(() =>
    chat._parseResponse({
      choices: [
        {
          message: {
            toolCalls: [
              {
                type: "FUNCTION",
                name: "get_weather",
                arguments: "{}",
              },
            ],
          },
        },
      ],
    } as unknown as GenericChatResponse)
  ).toThrow("Invalid GenericChatResponse object");

  expect(() =>
    OciGenAiGenericChat._convertBaseMessagesToGenericMessages([
      new AIMessage({
        content: "",
        tool_calls: [{ name: "get_weather", args: {} }],
      }),
    ])
  ).toThrow("LangChain tool call 'get_weather' did not contain a tool call id");
});

test("OCI GenAI Generic preserves malformed tool-call arguments safely", () => {
  const chat = new OciGenAiGenericChat(createParams);

  expect(
    chat._parseResponse({
      choices: [
        {
          message: {
            toolCalls: [
              {
                id: "call-weather",
                type: "FUNCTION",
                name: "get_weather",
                arguments: '{"city":',
              },
            ],
          },
        },
      ],
    } as unknown as GenericChatResponse).toolCalls
  ).toEqual([
    {
      type: "tool_call",
      id: "call-weather",
      name: "get_weather",
      args: {},
    },
  ]);
});

const invalidCohereStreamedChunks = [
  null,
  {},
  {
    ext: "this is some text",
    prop: true,
  },
  {
    ext: "this is some text",
    message: ["hello"],
  },
  {
    apiFormat: CohereChatRequest.apiFormat,
  },
];

test("OCI GenAI Cohere parse invalid streamed chunks", async () => {
  const cohereChat = new OciGenAiCohereChat(createParams);

  for (const invalidValue of invalidCohereStreamedChunks) {
    expect(() => cohereChat._parseStreamedResponseChunk(invalidValue)).toThrow(
      "Invalid streamed response chunk data"
    );
  }
});

const validCohereStreamedChunks = [
  {
    apiFormat: CohereChatRequest.apiFormat,
    text: "this is some text",
  },
  {
    apiFormat: CohereChatRequest.apiFormat,
    text: "this is some text",
    pad: "aaaaa",
  },
];

test("OCI GenAI Cohere parse invalid streamed chunks", async () => {
  const cohereChat = new OciGenAiCohereChat(createParams);

  for (const invalidValue of validCohereStreamedChunks) {
    expect(cohereChat._parseStreamedResponseChunk(invalidValue)).toEqual({
      text: "this is some text",
    });
  }
});

const invalidGenericStreamedChunks = [
  null,
  {},
  {
    ext: "this is some text",
    prop: true,
  },
  {
    ext: "this is some text",
    message: ["hello"],
  },
  {
    apiFormat: CohereChatRequest.apiFormat,
  },
  { finishReason: 123 },
  { usage: "garbage" },
  {
    message: {
      toolCalls: [],
    },
  },
  {
    message: {
      content: [{ type: TextContent.type, text: "valid text" }],
    },
    usage: "garbage",
  },
];

test("OCI GenAI Generic parse invalid streamed chunks", async () => {
  const genericChat = new OciGenAiGenericChat(createParams);

  for (const invalidValue of invalidGenericStreamedChunks) {
    expect(() => genericChat._parseStreamedResponseChunk(invalidValue)).toThrow(
      "Invalid streamed response chunk data"
    );
  }
});

test("OCI GenAI Generic streamed content matches non-streamed concatenation", () => {
  const genericChat = new OciGenAiGenericChat(createParams);
  const chunk = {
    message: {
      content: [
        { type: TextContent.type, text: "hello" },
        { type: TextContent.type, text: " world" },
      ],
    },
  };

  expect(genericChat._parseStreamedResponseChunk(chunk)).toEqual({
    text: "hello world",
  });
});

const validGenericStreamedChunks = [
  {
    message: {
      content: [
        {
          type: TextContent.type,
          text: "this is some text",
        },
      ],
    },
  },
  {
    finishReason: "stop sequence",
  },
];

test("OCI GenAI Generic parse invalid streamed chunks", async () => {
  const genericChat = new OciGenAiGenericChat(createParams);

  for (const invalidValue of validGenericStreamedChunks) {
    expect(genericChat._parseStreamedResponseChunk(invalidValue)).toEqual(
      "finishReason" in invalidValue
        ? { finishReason: invalidValue.finishReason }
        : { text: "this is some text" }
    );
  }
});

test("OCI GenAI cohere history and message split", () => {
  const lastHumanMessage = "Last human message";

  testCohereMessageHistorySplit({
    messages: [new LangChainHumanMessage(lastHumanMessage)],
    lastHumanMessage,
    numExpectedMessagesInHistory: 0,
    numExpectedHumanMessagesInHistory: 0,
    numExpectedOtherMessagesInHistory: 0,
  });

  testCohereMessageHistorySplit({
    messages: [
      new LangChainSystemMessage("System message"),
      new LangChainHumanMessage("Human message"),
      new AIMessage("Assistant message"),
      new LangChainHumanMessage(lastHumanMessage),
    ],
    lastHumanMessage,
    numExpectedMessagesInHistory: 3,
    numExpectedHumanMessagesInHistory: 1,
    numExpectedOtherMessagesInHistory: 2,
  });
});

test("OCI GenAI cohere requires a final human message", () => {
  const invalidConversations = [
    [],
    [new LangChainSystemMessage("System message")],
    [new LangChainHumanMessage("Human message"), new AIMessage("Reply")],
  ];

  for (const messages of invalidConversations) {
    expect(() => OciGenAiCohereChat._splitMessageAndHistory(messages)).toThrow(
      "Cohere chat requires the final message to be a human message"
    );
  }
});

test("OCI GenAI chat cohere _convertBaseMessageToCohereMessage", () => {
  const messageContent = "message content";
  const testCases = [
    {
      message: new AIMessage(messageContent),
      expectedRole: CohereChatBotMessage.role,
    },
    {
      message: new SystemMessage(messageContent),
      expectedRole: CohereSystemMessage.role,
    },
    {
      message: new HumanMessage(messageContent),
      expectedRole: CohereUserMessage.role,
    },
    {
      message: new ToolMessage(messageContent, "tool id"),
      expectedError: "Message type 'tool' is not supported",
    },
  ];

  testCases.forEach((testCase) => {
    if (testCase.expectedError) {
      expect(() =>
        OciGenAiCohereChat._convertBaseMessageToCohereMessage(testCase.message)
      ).toThrowError(testCase.expectedError);
    } else {
      expect(
        OciGenAiCohereChat._convertBaseMessageToCohereMessage(testCase.message)
      ).toEqual({
        role: testCase.expectedRole,
        message: messageContent,
      });
    }
  });
});

test("OCI GenAI chat generic _convertBaseMessagesToGenericMessages", () => {
  const testCases: Array<{
    input: BaseMessage[];
    expectedOutput?: unknown;
    expectedError?: string;
  }> = [
    {
      input: [],
      expectedOutput: [],
    },
    {
      input: [new AIMessage("Hello")],
      expectedOutput: [
        {
          role: GenericAssistantMessage.role,
          content: [
            {
              text: "Hello",
              type: TextContent.type,
            },
          ],
        },
      ],
    },
    {
      input: [
        new AIMessage("Hello"),
        new HumanMessage("Hi"),
        new SystemMessage("Welcome"),
      ],
      expectedOutput: [
        {
          role: GenericAssistantMessage.role,
          content: [
            {
              text: "Hello",
              type: TextContent.type,
            },
          ],
        },
        {
          role: GenericUserMessage.role,
          content: [
            {
              text: "Hi",
              type: TextContent.type,
            },
          ],
        },
        {
          role: GenericSystemMessage.role,
          content: [
            {
              text: "Welcome",
              type: TextContent.type,
            },
          ],
        },
      ],
    },
    {
      input: [
        new AIMessage({
          content: "Hello",
          tool_calls: [
            {
              id: "id",
              name: "get_weather",
              args: { city: "London" },
            },
          ],
        }),
        new ToolMessage("Hi", "id"),
        new HumanMessage("Hi"),
      ],
      expectedOutput: [
        {
          role: GenericAssistantMessage.role,
          content: [
            {
              text: "Hello",
              type: TextContent.type,
            },
          ],
          toolCalls: [
            {
              id: "id",
              type: "FUNCTION",
              name: "get_weather",
              arguments: '{"city":"London"}',
            },
          ],
        },
        {
          role: "TOOL",
          toolCallId: "id",
          content: [
            {
              text: "Hi",
              type: TextContent.type,
            },
          ],
        },
        {
          role: GenericUserMessage.role,
          content: [
            {
              text: "Hi",
              type: TextContent.type,
            },
          ],
        },
      ],
    },
    {
      input: [
        new AIMessage({
          content: "",
          tool_calls: [
            { id: "call-1", name: "get_weather", args: { city: "London" } },
          ],
        }),
        new ToolMessage("London", "call-1"),
        new ToolMessage("London again", "call-1"),
      ],
      expectedError: "ToolMessage references unknown tool call 'call-1'",
    },
    {
      input: [
        new AIMessage({
          content: "",
          tool_calls: [
            { id: "call-1", name: "get_weather", args: {} },
            { id: "call-1", name: "get_forecast", args: {} },
          ],
        }),
      ],
      expectedError: "Duplicate tool call id 'call-1'",
    },
  ];

  testCases.forEach((testCase) => {
    if (testCase.expectedError) {
      expect(() =>
        OciGenAiGenericChat._convertBaseMessagesToGenericMessages(
          testCase.input
        )
      ).toThrow(testCase.expectedError);
    } else {
      expect(
        OciGenAiGenericChat._convertBaseMessagesToGenericMessages(
          testCase.input
        )
      ).toEqual(testCase.expectedOutput);
    }
  });
});

test("OCI GenAI chat Cohere _isCohereResponse", () => {
  const testCaseArray = [
    {
      input: {
        text: "Hello World!",
        apiFormat: "json",
      },
      expectedResult: true,
    },
    {
      input: null,
      expectedResult: false,
    },
    {
      input: "not an object",
      expectedResult: false,
    },
    {
      input: 123,
      expectedResult: false,
    },
    {
      input: undefined,
      expectedResult: false,
    },
    {
      input: {
        foo: "bar",
        apiFormat: "json",
      },
      expectedResult: false,
    },
    {
      input: {
        text: 123,
        apiFormat: "json",
      },
      expectedResult: false,
    },
  ];

  testCaseArray.forEach(({ input, expectedResult }) => {
    expect(OciGenAiCohereChat._isCohereResponse(input)).toBe(expectedResult);
  });
});

test("OCI GenAI chat generic _isGenericResponse", () => {
  const testCases = [
    {
      input: {
        timeCreated: new Date(),
        choices: [
          {
            index: 1,
            message: {
              role: "assistant",
              content: [{ type: "text", text: "Hello" }],
            },
            finishReason: "",
          },
        ],
        apiFormat: "v1",
      },
      expectedOutput: true,
    },
    {
      input: null,
      expectedOutput: false,
    },
    {
      input: "not an object",
      expectedOutput: false,
    },
    {
      input: {
        timeCreated: new Date(),
        apiFormat: "v1",
      },
      expectedOutput: false,
    },
    {
      input: {
        timeCreated: new Date(),
        choices: "not an array",
        apiFormat: "v1",
      },
      expectedOutput: false,
    },
    {
      input: {
        timeCreated: new Date(),
        choices: [],
        apiFormat: "v1",
      },
      expectedOutput: false,
    },
    {
      input: {
        timeCreated: new Date(),
        choices: [
          {
            index: 1,
            message: "not an object",
          },
        ],
        apiFormat: "v1",
      },
      expectedOutput: false,
    },
  ];

  testCases.forEach(({ input, expectedOutput }) => {
    expect(OciGenAiGenericChat._isGenericResponse(input)).toBe(expectedOutput);
  });
});

test("OCI GenAI chat models invoke + check sdkClient cache logic", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor, parameter) => {
      const chatClass = new ChatClassType({
        compartmentId,
        onDemandModelId,
        maxRetries: 0,
        client: {
          chat: async () => parameter,
        },
      });

      expect(OciGenAiBaseChat._isSdkClient(chatClass._sdkClient)).toBe(false);
      await chatClass.invoke("this is a prompt");
      await chatClass.invoke("this is a prompt");
      expect(OciGenAiBaseChat._isSdkClient(chatClass._sdkClient)).toBe(true);
    },
    chatClassReturnValues
  );
});

test("OCI GenAI chat models invoke API fail", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor) => {
      const chatClass = new ChatClassType({
        compartmentId,
        onDemandModelId,
        maxRetries: 0,
        client: {
          chat: async () => {
            throw new Error("API error");
          },
        },
      });

      expect(OciGenAiBaseChat._isSdkClient(chatClass._sdkClient)).toBe(false);
      await expect(chatClass.invoke("this is a prompt")).rejects.toThrow(
        "Error executing chat API, error: API error"
      );
      await expect(chatClass.invoke("this is a prompt")).rejects.toThrow(
        "Error executing chat API, error: API error"
      );
      expect(OciGenAiBaseChat._isSdkClient(chatClass._sdkClient)).toBe(true);
    }
  );
});

test("OCI GenAI chat preserves the original API error as its cause", async () => {
  const originalError = new Error("API error");
  const chat = new OciGenAiGenericChat({
    compartmentId,
    onDemandModelId,
    maxRetries: 0,
    client: {
      chat: async () => {
        throw originalError;
      },
    } as any,
  });

  await expect(chat.invoke("this is a prompt")).rejects.toMatchObject({
    message: "Error executing chat API, error: API error",
    cause: originalError,
  });
});

test("OCI GenAI SDK client guard requires a chat function", () => {
  expect(OciGenAiBaseChat._isSdkClient({ client: {} })).toBe(false);
  expect(OciGenAiBaseChat._isSdkClient({ client: { chat: vi.fn() } })).toBe(
    true
  );
});

test("OCI GenAI chat models invoke with with no initialized SDK client", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor) => {
      const chatClass = new ChatClassType({
        compartmentId,
        dedicatedEndpointId,
        client: {
          chat: async () => true,
        },
      });

      await expect(
        chatClass._chat(chatClass._prepareRequest(messages, callOptions, true))
      ).rejects.toThrow(
        "Error executing chat API, error: OCI SDK client not initialized"
      );
    }
  );
});

test("OCI GenAI chat models invoke with sdk client uninitialized", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor) => {
      const chatClass = new ChatClassType({
        compartmentId,
        dedicatedEndpointId,
        client: {
          chat: async () => true,
        },
      });

      await expect(
        chatClass._chat(chatClass._prepareRequest(messages, callOptions, true))
      ).rejects.toThrow(
        "Error executing chat API, error: OCI SDK client not initialized"
      );
    }
  );
});

test("OCI GenAI chat models invoke with dedicated endpoint", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor, params) => {
      const chatClass = new ChatClassType({
        compartmentId,
        dedicatedEndpointId,
        client: {
          chat: async () => params,
        },
      });

      await expect(
        chatClass.invoke("this is a message")
      ).resolves.toMatchObject({
        content: expect.anything(),
      });
    },
    chatClassReturnValues
  );
});

const chatStreamReturnValues: string[][] = [
  [
    `data: {"apiFormat":"${CohereChatRequest.apiFormat}", "text":"this is some text"}\n\n`,
    `data: {"apiFormat":"${CohereChatRequest.apiFormat}", "text":"this is some more text"}\n\n`,
  ],
  [
    `data: {"message":{"content":[{"type":"${TextContent.type}","text":"this is some text"}]}}\n\n`,
    `data: {"message":{"content":[{"type":"${TextContent.type}","text":"this is some more text"}]}}\n\n`,
    'data: {"finishReason":"stop sequence"}\n\n',
  ],
];

test("OCI GenAI chat models stream", async () => {
  await testEachChatModelType(
    async (ChatClassType: OciGenAiChatConstructor, parameter) => {
      let numApiCalls = 0;
      const chatClass = new ChatClassType({
        compartmentId,
        onDemandModelId,
        client: {
          chat: async () => {
            numApiCalls += 1;
            return createStreamFromStringArray(parameter);
          },
        },
      });

      expect(OciGenAiBaseChat._isSdkClient(chatClass._sdkClient)).toBe(false);
      const streamedMessages = [];

      for await (const message of await chatClass.stream([
        "this is a prompt",
      ])) {
        streamedMessages.push(message);
      }

      expect(streamedMessages).toHaveLength(
        ChatClassType === OciGenAiGenericChat ? 3 : 2
      );
      if (ChatClassType === OciGenAiGenericChat) {
        expect(streamedMessages.at(-1)?.response_metadata).toEqual({
          finish_reason: "stop sequence",
        });
      }
      expect(numApiCalls).toBe(1);
      expect(OciGenAiBaseChat._isSdkClient(chatClass._sdkClient)).toBe(true);
    },
    chatStreamReturnValues
  );
});

/*
 * Utils
 */

async function testInvalidValues(
  streamIterator: JsonServerEventsIterator
): Promise<void> {
  let numRuns = 0;

  try {
    for await (const _event of streamIterator) {
      numRuns += 1;
    }
  } catch (error) {
    expect((<Error>error)?.message).toMatch(invalidEventDataErrors);
  }

  expect(numRuns).toBe(0);
}

async function testNumExpectedServerEvents(
  serverEvents: string[],
  numExpectedServerEvents: number
) {
  const stream = createStreamFromStringArray(serverEvents);
  const streamIterator = new JsonServerEventsIterator(stream);
  let numEvents = 0;

  for await (const _event of streamIterator) {
    numEvents += 1;
  }

  expect(numEvents).toBe(numExpectedServerEvents);
}

async function collectServerEvents(serverEvents: string[]): Promise<unknown[]> {
  const streamIterator = new JsonServerEventsIterator(
    createStreamFromStringArray(serverEvents)
  );
  const events = [];

  for await (const event of streamIterator) {
    events.push(event);
  }

  return events;
}

async function collectServerEventBytes(
  serverEvents: Uint8Array[]
): Promise<unknown[]> {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of serverEvents) {
        controller.enqueue(event);
      }
      controller.close();
    },
  });
  const events = [];

  for await (const event of new JsonServerEventsIterator(stream)) {
    events.push(event);
  }

  return events;
}

function testSdkClient(
  sdkClient: OciGenAiSdkClient,
  regionId: string,
  maxAttempts: number
) {
  expect(OciGenAiBaseChat._isSdkClient(sdkClient)).toBe(true);
  // OCI's `region` setter stores a Region object, whereas `regionId` stores a
  // string. Consult the last setter so an explicit regionId override takes
  // precedence over the Region object retained from the auth provider.
  const client = <any>sdkClient.client;
  expect(
    client._lastSetRegionOrRegionId === "regionId"
      ? client._regionId
      : client._region?.regionId
  ).toBe(regionId);
  expect(
    (<any>sdkClient.client)._clientConfiguration?.retryConfiguration
      ?.terminationStrategy?._maxAttempts
  ).toBe(maxAttempts);
}

class StringArrayToInt8ArraySource implements UnderlyingSource {
  private valuesIndex = 0;

  private textEncoder = new TextEncoder();

  // eslint-disable-next-line no-empty-function
  constructor(private values: string[]) {}

  pull(controller: ReadableStreamDefaultController) {
    if (this.valuesIndex < this.values.length) {
      controller.enqueue(
        this.textEncoder.encode(this.values[this.valuesIndex])
      );
      this.valuesIndex += 1;
    } else {
      controller.close();
    }
  }

  cancel() {
    this.valuesIndex = this.values.length;
  }
}

function createStreamFromStringArray(
  values: string[]
): ReadableStream<Uint8Array> {
  return new ReadableStream(new StringArrayToInt8ArraySource(values));
}

async function testEachChatModelType(
  testFunction: (
    ChatClassType: OciGenAiChatConstructor,
    parameter?: any | undefined
  ) => Promise<void>,
  parameters?: any[]
) {
  const chatClassTypes: OciGenAiChatConstructor[] = [
    OciGenAiCohereChat,
    OciGenAiGenericChat,
  ];

  for (let i = 0; i < chatClassTypes.length; i += 1) {
    await testFunction(chatClassTypes[i], parameters?.[i]);
  }
}

interface TestMessageHistorySplitParams {
  messages: BaseMessage[];
  lastHumanMessage: string;
  numExpectedMessagesInHistory: number;
  numExpectedHumanMessagesInHistory: number;
  numExpectedOtherMessagesInHistory: number;
}

function testCohereMessageHistorySplit(params: TestMessageHistorySplitParams) {
  const messageAndHistory = OciGenAiCohereChat._splitMessageAndHistory(
    params.messages
  );

  expect(messageAndHistory.message).toBe(params.lastHumanMessage);
  expect(messageAndHistory.chatHistory.length).toBe(
    params.numExpectedMessagesInHistory
  );

  let numHumanMessages = params.numExpectedHumanMessagesInHistory;
  let numOtherMessages = params.numExpectedOtherMessagesInHistory;

  for (const message of messageAndHistory.chatHistory) {
    testCohereMessageHistorySplitMessage(message, params.lastHumanMessage);

    if (message.role === OciGenAiCohereUserMessage.role) {
      numHumanMessages -= 1;
    } else {
      numOtherMessages -= 1;
    }
  }

  expect(numHumanMessages).toBe(0);
  expect(numOtherMessages).toBe(0);
}

function testCohereMessageHistorySplitMessage(
  message: CohereMessage,
  lastHumanMessage: string
) {
  expect([
    CohereChatBotMessage.role,
    OciGenAiCohereSystemMessage.role,
    OciGenAiCohereUserMessage.role,
  ]).toContain(message.role);
  expect((<CohereSystemMessage>message).message).not.toBe(lastHumanMessage);
}

function removeElements(originalArray: any[], removeIndexes: number[]): any[] {
  for (const removeIndex of removeIndexes) {
    originalArray.splice(removeIndex, 1);
  }

  return originalArray;
}
