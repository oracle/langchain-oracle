/* eslint-disable no-process-env */

import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { models } from "oci-generativeaiinference";
import { expect, test } from "vitest";
import { z } from "zod";

import { OciGenAiGenericChat } from "../generic_chat.js";
import { OciGenAiNewClientAuthType } from "../types.js";

const compartmentId =
  process.env.OCI_COMPARTMENT_ID ??
  process.env.OCI_GENAI_INTEGRATION_TESTS_COMPARTMENT_ID;
const modelId =
  process.env.OCI_MODEL_ID ??
  process.env.OCI_GENAI_INTEGRATION_TESTS_GENERIC_ON_DEMAND_MODEL_ID ??
  "xai.grok-3";
const serviceEndpoint =
  process.env.OCI_ENDPOINT ??
  "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com";

test.skipIf(!compartmentId)(
  "OCI Generic completes a real LangGraph tool round-trip",
  async () => {
    const weather = tool(
      async ({ city }: { city: string }) => `${city}: 22C and sunny`,
      {
        name: "get_weather",
        description:
          "Get the weather for a city. Always use this tool when asked for weather.",
        schema: z.object({ city: z.string() }),
      }
    );
    const model = new OciGenAiGenericChat({
      compartmentId,
      onDemandModelId: modelId,
      newClientParams: {
        authType: OciGenAiNewClientAuthType.ConfigFile,
        regionId: "us-phoenix-1",
        serviceEndpoint,
      },
    });
    const modelWithRequiredTool = model.bindTools([weather], {
      requestParams: {
        toolChoice: { type: models.ToolChoiceRequired.type },
      },
    });
    const toolNode = new ToolNode([weather]);

    const graph = new StateGraph(MessagesAnnotation)
      .addNode("agent", async (state) => {
        const lastMessage = state.messages.at(-1);
        const response = ToolMessage.isInstance(lastMessage)
          ? await model.invoke(state.messages)
          : await modelWithRequiredTool.invoke(state.messages);
        return { messages: [response] };
      })
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", (state) => {
        const lastMessage = state.messages.at(-1);
        return AIMessage.isInstance(lastMessage) &&
          lastMessage.tool_calls?.length
          ? "tools"
          : "__end__";
      })
      .addEdge("tools", "agent")
      .compile();

    const result = await graph.invoke({
      messages: [
        new HumanMessage("What is the weather in Phoenix? Use the tool."),
      ],
    });

    expect(
      result.messages.some((message) => ToolMessage.isInstance(message))
    ).toBe(true);
    expect(result.messages.at(-1)).toBeInstanceOf(AIMessage);
    expect((result.messages.at(-1) as AIMessage).content).not.toBe("");
  },
  100_000
);
