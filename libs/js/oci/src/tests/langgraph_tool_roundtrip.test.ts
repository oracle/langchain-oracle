import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import type { GenerativeAiInferenceClient } from "oci-generativeaiinference";
import { z } from "zod";
import { expect, test } from "vitest";

import { OciGenAiGenericChat } from "../generic_chat.js";

interface OciGenericChatRequest {
  tools?: unknown[];
  messages: Array<{ role?: string; toolCallId?: string }>;
}

test("OCI Generic completes a LangGraph tool round-trip", async () => {
  const weather = tool(
    async ({ city }: { city: string }) => `${city}: 22C and sunny`,
    {
      name: "get_weather",
      description: "Get current weather",
      schema: z.object({ city: z.string() }),
    }
  );
  let callCount = 0;

  const model = new OciGenAiGenericChat({
    compartmentId: "test-compartment",
    onDemandModelId: "test-model",
    client: {
      chat: (request: unknown) => {
        callCount += 1;
        const { chatRequest } = (
          request as { chatDetails: { chatRequest: OciGenericChatRequest } }
        ).chatDetails;

        if (callCount === 1) {
          expect(chatRequest.tools).toHaveLength(1);
          return {
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
              },
            },
          };
        }

        const lastMessage = chatRequest.messages.at(-1);
        expect(lastMessage).toMatchObject({
          role: "TOOL",
          toolCallId: "call-1",
        });

        return {
          chatResult: {
            chatResponse: {
              choices: [
                {
                  finishReason: "STOP",
                  message: {
                    content: [
                      { type: "TEXT", text: "London is 22C and sunny." },
                    ],
                  },
                },
              ],
            },
          },
        };
      },
    } as unknown as GenerativeAiInferenceClient,
  });
  const modelWithTools = model.bindTools([weather]);
  const toolNode = new ToolNode([weather]);

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("agent", async (state) => ({
      messages: [await modelWithTools.invoke(state.messages)],
    }))
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", (state) => {
      const lastMessage = state.messages.at(-1);
      return AIMessage.isInstance(lastMessage) && lastMessage.tool_calls?.length
        ? "tools"
        : "__end__";
    })
    .addEdge("tools", "agent")
    .compile();

  const result = await graph.invoke({
    messages: [new HumanMessage("What is the weather in London?")],
  });

  expect(callCount).toBe(2);
  expect(result.messages).toHaveLength(4);
  expect(result.messages[1]).toBeInstanceOf(AIMessage);
  expect((result.messages[1] as AIMessage).tool_calls).toEqual([
    expect.objectContaining({
      id: "call-1",
      name: "get_weather",
      args: { city: "London" },
    }),
  ]);
  expect(result.messages[2]).toBeInstanceOf(ToolMessage);
  expect((result.messages[2] as ToolMessage).tool_call_id).toBe("call-1");
  expect((result.messages[3] as AIMessage).content).toBe(
    "London is 22C and sunny."
  );
});

test("OCI Generic rejects a ToolMessage with an unknown tool call ID", () => {
  expect(() =>
    OciGenAiGenericChat._convertBaseMessagesToGenericMessages([
      new HumanMessage("weather"),
      new AIMessage({
        content: "",
        tool_calls: [
          {
            id: "call-1",
            name: "get_weather",
            args: { city: "London" },
          },
        ],
      }),
      new ToolMessage({ content: "22C", tool_call_id: "wrong-id" }),
    ])
  ).toThrow("ToolMessage references unknown tool call 'wrong-id'");
});
