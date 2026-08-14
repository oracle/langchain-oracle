/* eslint-disable @typescript-eslint/no-explicit-any */
import { expect, test, vi } from "vitest";

import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  HumanMessage as LangChainHumanMessage,
  SystemMessage as LangChainSystemMessage,
  ToolMessage as LangChainToolMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

import { GenerativeAiInferenceClient, models } from "oci-generativeaiinference";
import { MaxAttemptsTerminationStrategy } from "oci-common";

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

test("JsonServerEventsIterator accepts data fields without a space", async () => {
  const events = await collectServerEvents(['data:{"text":"hello"}\n\n']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator ignores comments and control-only events", async () => {
  const events = await collectServerEvents([
    ": keepalive\n\n",
    "event: message\nid: 123\nretry: 1000\n\n",
    'data: {"text":"hello"}\n\n',
  ]);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator dispatches a final event at end of stream", async () => {
  const events = await collectServerEvents(['data: {"text":"hello"}']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator parses CR-only framing", async () => {
  const events = await collectServerEvents(['data: {"text":"hello"}\r\r']);

  expect(events).toEqual([{ text: "hello" }]);
});

test("JsonServerEventsIterator bounds incomplete events", async () => {
  await expect(
    collectServerEvents([
      `data: {"text":"${"x".repeat(
        JsonServerEventsIterator._MAX_BUFFER_LENGTH
      )}`,
    ])
  ).rejects.toThrow("Server-sent event exceeds maximum buffered size");
});

/*
 *  OciGenAiSdkClient tests
 */

const authenticationDetailsProvider = {
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
  testSdkClient(sdkClient, OciGenAiSdkClient._DEFAULT_REGION_ID, 0);
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
  testSdkClient(sdkClient, OciGenAiSdkClient._DEFAULT_REGION_ID, 0);
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

/*
 *  Chat models tests
 */

const compartmentId = "oci.compartment.ocid";
const onDemandModelId = "oci.model.ocid";
const dedicatedEndpointId = "oci.dedicated.oci";
const createParams = {
  compartmentId,
  onDemandModelId,
};

const DummyClient = {
  chat() {},
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

test("OCI GenAI chat accepts text content blocks", () => {
  const message = new LangChainHumanMessage({
    content: [
      { type: "text", text: "hello" },
      { type: "reasoning", reasoning: "ignored" },
    ],
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
      chat: () => ({
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
      chat: (request: unknown) => {
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
      expectedOutput: true,
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
        client: {
          chat: () => parameter,
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
        client: {
          chat: () => {
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
    client: {
      chat: () => {
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
          chat: () => true,
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
          chat: () => true,
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
          chat: () => params,
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
          chat: () => {
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
  expect((<any>sdkClient.client)._regionId).toBe(regionId);
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
